"""
Multi-Station Temporal Deep Learning Architecture for PM2.5 & PM10 Forecasting.
Implements PRD Section 7.2:
- Station Entity Embedding (8 dimensions)
- Conv1D temporal noise filter
- Bidirectional LSTM + Unidirectional LSTM recurrent encoder
- Temporal Attention Layer
- Non-crossing Quantile Output Head (72 hours x 2 targets x 3 quantiles)
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Additive Bahdanau-style attention over temporal sequence timesteps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rnn_outputs: Tensor of shape (N, T, hidden_dim)
        Returns:
            context: Weighted representation of shape (N, hidden_dim)
            weights: Attention distribution of shape (N, T)
        """
        scores = self.attn(rnn_outputs).squeeze(-1)  # (N, T)
        weights = F.softmax(scores, dim=-1)  # (N, T)
        context = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1)  # (N, hidden_dim)
        return context, weights


class MultiStationLSTMForecaster(nn.Module):
    """
    Multi-Station Deep Learning Forecaster for PM2.5 and PM10.
    Combines learned station embeddings, local 1D convolution smoothing,
    stacked BiLSTM-LSTM recurrence, temporal attention, and non-crossing quantile heads.
    """

    def __init__(
        self,
        in_features: int,
        num_stations: int = 16,
        embedding_dim: int = 8,
        conv_filters: int = 64,
        conv_kernel_size: int = 3,
        bilstm_hidden: int = 128,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
        horizon: int = 72,
        num_targets: int = 2,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.horizon = horizon
        self.num_targets = num_targets
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)
        self.use_attention = use_attention

        # 1. Station ID Entity Embedding
        self.station_embedding = nn.Embedding(num_stations, embedding_dim)

        # 2. Local 1D Convolutional Noise Filter
        total_in_dim = in_features + embedding_dim
        self.conv1d = nn.Conv1d(
            in_channels=total_in_dim,
            out_channels=conv_filters,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
        )
        self.conv_act = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout / 2.0)

        # 3. Recurrent Backbone
        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=bilstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=bilstm_hidden * 2,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # 4. Attention mechanism
        if self.use_attention:
            self.attention = TemporalAttention(lstm_hidden)

        # 5. Output Projection Head
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.fc1_act = nn.ReLU()
        self.fc1_dropout = nn.Dropout(dropout)

        # Output raw logits: 72 * 2 * 3
        total_outputs = self.horizon * self.num_targets * self.num_quantiles
        self.fc_out = nn.Linear(128, total_outputs)

    def forward(
        self,
        x_sequence: torch.Tensor,
        station_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_sequence: Temporal observations of shape (N, T=360, in_features)
            station_ids: Categorical station indices of shape (N,)
        Returns:
            quantiles_pred: Shape (N, horizon=72, num_targets=2, num_quantiles=3)
                            Guaranteed monotonic: q10 <= q50 <= q90
            attn_weights: Attention weights of shape (N, T) if enabled
        """
        N, T, _ = x_sequence.shape

        # 1. Station Embedding expansion over time
        emb = self.station_embedding(station_ids)  # (N, embedding_dim)
        emb_tiled = emb.unsqueeze(1).repeat(1, T, 1)  # (N, T, embedding_dim)
        x_cat = torch.cat([x_sequence, emb_tiled], dim=-1)  # (N, T, in_features + embedding_dim)

        # 2. 1D Convolution (expects (N, channels, T))
        x_conv = x_cat.permute(0, 2, 1)
        x_conv = self.conv1d(x_conv)
        x_conv = self.conv_act(x_conv)
        x_conv = self.conv_dropout(x_conv)
        x_conv = x_conv.permute(0, 2, 1)  # (N, T, conv_filters)

        # 3. Stacked BiLSTM -> LSTM
        bilstm_out, _ = self.bilstm(x_conv)
        bilstm_out = self.bilstm_dropout(bilstm_out)

        lstm_out, _ = self.lstm(bilstm_out)
        lstm_out = self.lstm_dropout(lstm_out)  # (N, T, lstm_hidden)

        # 4. Aggregation (Attention or Last Step)
        attn_weights = None
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)  # (N, lstm_hidden)
        else:
            context = lstm_out[:, -1, :]

        # 5. Multi-Quantile Prediction Head
        h = self.fc1(context)
        h = self.fc1_act(h)
        h = self.fc1_dropout(h)

        raw_out = self.fc_out(h)  # (N, 72 * 2 * 3)
        raw_reshaped = raw_out.view(N, self.horizon, self.num_targets, self.num_quantiles)

        # 6. Non-Crossing Monotonic Quantile Guarantee:
        # Index 0: q10, Index 1: q50 (median), Index 2: q90
        # q50 = median
        # q10 = q50 - softplus(delta_10)
        # q90 = q50 + softplus(delta_90)
        q_median = raw_reshaped[:, :, :, 1]
        delta_10 = F.softplus(raw_reshaped[:, :, :, 0])
        delta_90 = F.softplus(raw_reshaped[:, :, :, 2])

        q_10 = q_median - delta_10
        q_50 = q_median
        q_90 = q_median + delta_90

        quantiles_pred = torch.stack([q_10, q_50, q_90], dim=-1)
        return quantiles_pred, attn_weights
