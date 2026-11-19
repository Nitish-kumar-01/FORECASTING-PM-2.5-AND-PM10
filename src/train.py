"""
Training Engine, Conformal Calibration, and Model Packaging.
Implements PRD Section 7.8 and FR-16:
- Joint multi-station training with Adam, EarlyStopping, and LR scheduling
- Weighted Pinball Loss for quantile optimization
- Post-hoc Conformal Interval Calibration for empirical ~80% coverage
- Unified Model Package Serializer (weights, scalers, schema, station metadata)
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.loss import WeightedPinballLoss
from src.models.lstm import MultiStationLSTMForecaster
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def get_default_device() -> torch.device:
    """Selects best available compute hardware (Apple Silicon MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calibrate_conformal_multiplier(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """
    Computes post-hoc conformal band adjustment factor on validation set.
    Calibrates band width:
        q10_cal = q50 - alpha * (q50 - q10)
        q90_cal = q50 + alpha * (q90 - q50)
    so empirical coverage on validation set equals target_coverage (0.80).
    """
    # y_true: (N, horizon, n_targets)
    # y_quantiles: (N, horizon, n_targets, 3) -> 0: q10, 1: q50, 2: q90
    q10 = y_quantiles[:, :, :, 0]
    q50 = y_quantiles[:, :, :, 1]
    q90 = y_quantiles[:, :, :, 2]

    # Baseline coverage check
    inside = (y_true >= q10) & (y_true <= q90)
    current_cov = float(np.mean(inside))
    logger.info(f"Uncalibrated validation coverage: {current_cov * 100:.2f}% (Target: {target_coverage * 100:.1f}%)")

    if 0.75 <= current_cov <= 0.85:
        return 1.0

    # Search optimal alpha between 0.5 and 2.5
    best_alpha = 1.0
    best_diff = abs(current_cov - target_coverage)

    for alpha in np.linspace(0.5, 2.5, 81):
        q10_adj = q50 - alpha * (q50 - q10)
        q90_adj = q50 + alpha * (q90 - q50)
        cov = np.mean((y_true >= q10_adj) & (y_true <= q90_adj))
        diff = abs(cov - target_coverage)
        if diff < best_diff:
            best_diff = diff
            best_alpha = float(alpha)

    logger.info(f"Calibrated conformal multiplier alpha: {best_alpha:.3f}")
    return best_alpha


def train_single_model(
    model: MultiStationLSTMForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 10,
    gradient_clip: float = 1.0,
    high_pollution_weight: float = 2.0,
    device: Optional[torch.device] = None,
) -> Tuple[MultiStationLSTMForecaster, Dict[str, List[float]]]:
    """
    Executes deep learning training loop with early stopping and loss tracking.
    """
    if device is None:
        device = get_default_device()

    model = model.to(device)
    criterion = WeightedPinballLoss(
        quantiles=model.quantiles,
        high_pollution_weight=high_pollution_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # 1. Training Phase
        model.train()
        train_losses = []
        for x_seq, stn_ids, y_batch in train_loader:
            x_seq = x_seq.to(device)
            stn_ids = stn_ids.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            q_pred, _ = model(x_seq, stn_ids)
            loss = criterion(q_pred, y_batch)
            loss.backward()

            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # 2. Validation Phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_seq, stn_ids, y_batch in val_loader:
                x_seq = x_seq.to(device)
                stn_ids = stn_ids.to(device)
                y_batch = y_batch.to(device)

                q_pred, _ = model(x_seq, stn_ids)
                loss = criterion(q_pred, y_batch)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses))
        scheduler.step(mean_val_loss)

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)

        # Early Stopping Check
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best weights.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def save_model_package(
    model: MultiStationLSTMForecaster,
    scalers: Dict[str, Any],
    station_mapping: Dict[str, int],
    feature_names: List[str],
    target_names: List[str],
    conformal_alpha: float,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Union[str, Path] = "models/model_package.pt",
    status_table: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Serializes complete model package required for independent inference (FR-16).
    """
    package = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_architecture_params": {
            "in_features": model.in_features,
            "num_stations": model.station_embedding.num_embeddings,
            "embedding_dim": model.station_embedding.embedding_dim,
            "conv_filters": model.conv1d.out_channels,
            "conv_kernel_size": model.conv1d.kernel_size[0],
            "bilstm_hidden": model.bilstm.hidden_size,
            "lstm_hidden": model.lstm.hidden_size,
            "horizon": model.horizon,
            "num_targets": model.num_targets,
            "quantiles": model.quantiles,
            "use_attention": model.use_attention,
        },
        "scalers": scalers,
        "station_mapping": station_mapping,
        "feature_names": feature_names,
        "target_names": target_names,
        "conformal_alpha": conformal_alpha,
        "metrics": metrics,
        "status_table": status_table or [],
        "config": config,
        "training_timestamp": datetime.now().isoformat(),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, out_path)
    logger.info(f"Saved complete model package -> {out_path}")
    return out_path


def load_model_package(
    package_path: Union[str, Path] = "models/model_package.pt",
    device: Optional[torch.device] = None,
) -> Tuple[MultiStationLSTMForecaster, Dict[str, Any]]:
    """Loads model package for forecasting or evaluation."""
    if device is None:
        device = get_default_device()

    path = Path(package_path)
    if not path.exists():
        raise FileNotFoundError(f"Model package not found at {path}")

    package = torch.load(path, map_location=device, weights_only=False)

    params = package["model_architecture_params"]
    model = MultiStationLSTMForecaster(
        in_features=params["in_features"],
        num_stations=params.get("num_stations", 16),
        embedding_dim=params.get("embedding_dim", 8),
        conv_filters=params.get("conv_filters", 64),
        conv_kernel_size=params.get("conv_kernel_size", 3),
        bilstm_hidden=params.get("bilstm_hidden", 128),
        lstm_hidden=params.get("lstm_hidden", 64),
        horizon=params.get("horizon", 72),
        num_targets=params.get("num_targets", 2),
        quantiles=params.get("quantiles", [0.10, 0.50, 0.90]),
        use_attention=params.get("use_attention", True),
    )
    model.load_state_dict(package["model_state_dict"])
    model.to(device)
    model.eval()

    return model, package


def train_from_data_dir(
    raw_dir: Union[str, Path] = "data/raw",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_package_path: Union[str, Path] = "models/model_package.pt",
) -> Path:
    """
    CLI / Script pipeline to train model from all station files in raw_dir.
    """
    from src.data.fetch import load_all_stations
    from src.data.clean import run_cleaning_pipeline
    from src.data.features import engineer_station_features, get_model_feature_columns
    from src.data.windows import StationScaler, chronological_split, create_sliding_windows

    raw_path = Path(raw_dir)
    cfg = load_config()

    stations = load_all_stations(raw_path)
    if not stations:
        raise ValueError(f"No station data files found in '{raw_path}'. Place .xlsx/.csv files there first.")

    logger.info(f"Loaded {len(stations)} station(s) for training: {list(stations.keys())}")

    station_mapping = {name: idx for idx, name in enumerate(stations.keys())}
    scalers = {}
    train_x_list, train_stn_list, train_y_list = [], [], []
    val_x_list, val_stn_list, val_y_list = [], [], []
    status_table = []
    feat_cols = []

    for stn_name, df_raw in stations.items():
        logger.info(f"Processing station: {stn_name} ({len(df_raw)} records)...")
        df_clean, _ = run_cleaning_pipeline(df_raw, config=cfg)
        stn_id = station_mapping[stn_name]
        df_feat = engineer_station_features(df_clean, station_id=stn_id, config=cfg)
        feat_cols = get_model_feature_columns(df_feat)

        train_df, val_df, _ = chronological_split(
            df_feat,
            buffer_hours=12 if len(df_feat) < 1000 else 72,
        )

        scaler = StationScaler(target_cols=["PM2.5", "PM10"])
        train_scaled = scaler.fit_transform(train_df)
        val_scaled = scaler.transform(val_df)
        scalers[stn_name] = scaler

        X_tr, S_tr, Y_tr = create_sliding_windows(train_scaled, feat_cols, stride=2 if len(train_scaled) < 1000 else 1, station_id=stn_id)
        X_va, S_val, Y_val = create_sliding_windows(val_scaled, feat_cols, stride=4, station_id=stn_id)

        if len(X_tr) > 0:
            train_x_list.append(X_tr)
            train_stn_list.append(S_tr)
            train_y_list.append(Y_tr)
        if len(X_va) > 0:
            val_x_list.append(X_va)
            val_stn_list.append(S_val)
            val_y_list.append(Y_val)

        status_table.append({
            "station": stn_name,
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "data_range": f"{df_raw.index.min().strftime('%Y-%m-%d')} to {df_raw.index.max().strftime('%Y-%m-%d')}",
            "days_used": round((df_raw.index.max() - df_raw.index.min()).total_seconds() / 86400.0, 1),
            "status": "Approved" if len(df_raw) > 5000 else "Warning (Sample)",
        })

    X_train = np.concatenate(train_x_list, axis=0)
    S_train = np.concatenate(train_stn_list, axis=0)
    Y_train = np.concatenate(train_y_list, axis=0)

    X_val = np.concatenate(val_x_list, axis=0) if val_x_list else X_train[:10]
    S_val = np.concatenate(val_stn_list, axis=0) if val_stn_list else S_train[:10]
    Y_val = np.concatenate(val_y_list, axis=0) if val_y_list else Y_train[:10]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(S_train, dtype=torch.long),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(S_val, dtype=torch.long),
        torch.tensor(Y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = get_default_device()
    logger.info(f"Using compute device: {device}")

    model_inst = MultiStationLSTMForecaster(
        in_features=len(feat_cols),
        num_stations=max(16, len(station_mapping)),
        embedding_dim=8,
        conv_filters=32,
        bilstm_hidden=32,
        lstm_hidden=32,
        dropout=0.2,
        horizon=72,
        num_targets=2,
        quantiles=[0.10, 0.50, 0.90],
        use_attention=True,
    ).to(device)

    logger.info(f"Training neural model for {epochs} epochs...")
    trained_model, history = train_single_model(
        model_inst,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=learning_rate,
        patience=8,
        device=device,
    )

    # Conformal calibration on val
    with torch.no_grad():
        q_val_pred, _ = trained_model(torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(S_val, dtype=torch.long).to(device))
        q_val_np = q_val_pred.cpu().numpy()
    alpha = calibrate_conformal_multiplier(Y_val, q_val_np, target_coverage=0.80)

    # Save Package
    out_pkg = save_model_package(
        model=trained_model,
        scalers=scalers,
        station_mapping=station_mapping,
        feature_names=feat_cols,
        target_names=["PM2.5", "PM10"],
        conformal_alpha=alpha,
        metrics={"final_val_loss": history["val_loss"][-1]},
        config=cfg,
        output_path=output_package_path,
        status_table=status_table,
    )
    logger.info(f"Model training successfully completed! Package saved to: {out_pkg}")
    return out_pkg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Multi-Station LSTM Air Quality Forecaster.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing station files.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--output", type=str, default="models/model_package.pt", help="Output package path.")
    args = parser.parse_args()

    train_from_data_dir(
        raw_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_package_path=args.output,
    )

