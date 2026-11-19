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
from typing import Any, Dict, List, Optional, Tuple, Union

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
