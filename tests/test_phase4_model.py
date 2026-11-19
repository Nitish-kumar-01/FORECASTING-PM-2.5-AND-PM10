"""
Unit Tests for Phase 4: Multi-Station LSTM Neural Architecture,
Weighted Pinball Loss, Conformal Calibration, and Model Packaging.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.loss import WeightedPinballLoss
from src.models.lstm import MultiStationLSTMForecaster
from src.train import (
    calibrate_conformal_multiplier,
    load_model_package,
    save_model_package,
    train_single_model,
)


class TestPhase4DeepLearning(unittest.TestCase):
    """Tests the PyTorch architecture and quantile loss."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.in_features = 10
        self.num_stations = 4
        self.horizon = 72
        self.num_targets = 2
        self.quantiles = [0.10, 0.50, 0.90]

        self.model = MultiStationLSTMForecaster(
            in_features=self.in_features,
            num_stations=self.num_stations,
            embedding_dim=8,
            conv_filters=32,
            conv_kernel_size=3,
            bilstm_hidden=32,
            lstm_hidden=32,
            dropout=0.1,
            horizon=self.horizon,
            num_targets=self.num_targets,
            quantiles=self.quantiles,
            use_attention=True,
        ).to(self.device)

    def test_forward_pass_and_shapes(self):
        batch_size = 4
        seq_len = 360
        x = torch.randn(batch_size, seq_len, self.in_features)
        stn_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        q_pred, attn = self.model(x, stn_ids)

        self.assertEqual(q_pred.shape, (batch_size, self.horizon, self.num_targets, 3))
        self.assertIsNotNone(attn)
        self.assertEqual(attn.shape, (batch_size, seq_len))

    def test_non_crossing_monotonic_quantiles(self):
        batch_size = 8
        seq_len = 360
        x = torch.randn(batch_size, seq_len, self.in_features)
        stn_ids = torch.randint(0, self.num_stations, (batch_size,))

        q_pred, _ = self.model(x, stn_ids)

        q10 = q_pred[:, :, :, 0]
        q50 = q_pred[:, :, :, 1]
        q90 = q_pred[:, :, :, 2]

        # Verify q10 <= q50 <= q90 everywhere
        self.assertTrue(torch.all(q10 <= q50 + 1e-5), "Violated q10 <= q50")
        self.assertTrue(torch.all(q50 <= q90 + 1e-5), "Violated q50 <= q90")

    def test_weighted_pinball_loss(self):
        loss_fn = WeightedPinballLoss(quantiles=[0.10, 0.50, 0.90], high_pollution_weight=2.0)

        batch_size = 4
        y_true = torch.randn(batch_size, self.horizon, self.num_targets)
        y_pred = torch.randn(batch_size, self.horizon, self.num_targets, 3, requires_grad=True)

        loss = loss_fn(y_pred, y_true)
        self.assertGreater(loss.item(), 0.0)

        # Gradient flow test
        loss.backward()
        self.assertIsNotNone(y_pred.grad)
        self.assertGreater(torch.sum(torch.abs(y_pred.grad)).item(), 0.0)

    def test_conformal_calibration(self):
        N = 100
        # Create uncalibrated predictions that are too narrow (e.g. only 50% coverage)
        y_true = np.random.normal(0, 1.0, (N, 72, 2))
        q50 = np.zeros((N, 72, 2))
        q10 = q50 - 0.5  # narrow band
        q90 = q50 + 0.5
        y_quantiles = np.stack([q10, q50, q90], axis=-1)

        alpha = calibrate_conformal_multiplier(y_true, y_quantiles, target_coverage=0.80)
        # Narrow band requires widening (alpha > 1.0)
        self.assertGreater(alpha, 1.0)

    def test_training_loop_and_package_serialization(self):
        # Create tiny synthetic DataLoader
        N = 16
        seq_len = 360
        x_train = torch.randn(N, seq_len, self.in_features)
        stn_train = torch.randint(0, self.num_stations, (N,))
        y_train = torch.randn(N, self.horizon, self.num_targets)

        dataset = TensorDataset(x_train, stn_train, y_train)
        loader = DataLoader(dataset, batch_size=8)

        # Run 2 epochs of training
        trained_model, history = train_single_model(
            self.model,
            train_loader=loader,
            val_loader=loader,
            num_epochs=2,
            learning_rate=1e-3,
            device=self.device,
        )

        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)

        # Test Package Save and Load
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_path = Path(tmp_dir) / "test_model_pkg.pt"

            save_model_package(
                model=trained_model,
                scalers={"Rampur": None},
                station_mapping={"Rampur": 0},
                feature_names=[f"feat_{i}" for i in range(self.in_features)],
                target_names=["PM2.5", "PM10"],
                conformal_alpha=1.05,
                metrics={"Day+1": {"mae": 15.2}},
                config={"batch_size": 64},
                output_path=pkg_path,
            )
            self.assertTrue(pkg_path.exists())

            # Load back
            loaded_model, pkg = load_model_package(pkg_path, device=self.device)
            self.assertEqual(pkg["conformal_alpha"], 1.05)
            self.assertEqual(pkg["target_names"], ["PM2.5", "PM10"])

            # Verify identical inference
            q1, _ = trained_model(x_train[:2], stn_train[:2])
            q2, _ = loaded_model(x_train[:2], stn_train[:2])
            torch.testing.assert_close(q1, q2)


if __name__ == "__main__":
    unittest.main()
