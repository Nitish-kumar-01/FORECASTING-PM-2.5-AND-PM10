"""
Interactive Visualization Utilities for Streamlit Air Quality Dashboard.
Implements FR-12 forecast charts with CPCB regulatory color zones and uncertainty bands.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# CPCB Regulatory Zone Breakpoints and Fill Colors
AQI_ZONES = {
    "PM2.5": [
        (0, 30, "Good", "rgba(46, 204, 113, 0.15)", "#2ECC71"),
        (30, 60, "Satisfactory", "rgba(130, 224, 170, 0.15)", "#82E0AA"),
        (60, 90, "Moderate", "rgba(244, 208, 63, 0.15)", "#F4D03F"),
        (90, 120, "Poor", "rgba(230, 126, 34, 0.15)", "#E67E22"),
        (120, 250, "Very Poor", "rgba(231, 76, 60, 0.15)", "#E74C3C"),
        (250, 500, "Severe", "rgba(146, 43, 33, 0.15)", "#922B21"),
    ],
    "PM10": [
        (0, 50, "Good", "rgba(46, 204, 113, 0.15)", "#2ECC71"),
        (50, 100, "Satisfactory", "rgba(130, 224, 170, 0.15)", "#82E0AA"),
        (100, 250, "Moderate", "rgba(244, 208, 63, 0.15)", "#F4D03F"),
        (250, 350, "Poor", "rgba(230, 126, 34, 0.15)", "#E67E22"),
        (350, 430, "Very Poor", "rgba(231, 76, 60, 0.15)", "#E74C3C"),
        (430, 600, "Severe", "rgba(146, 43, 33, 0.15)", "#922B21"),
    ],
}


def create_72h_forecast_plot(
    future_timestamps: pd.DatetimeIndex,
    y_quantiles: np.ndarray,
    target_name: str = "PM2.5",
    target_idx: int = 0,
    historical_df: Optional[pd.DataFrame] = None,
    station_name: str = "Rampur, Korba - CECB",
) -> go.Figure:
    """
    Creates an interactive Plotly visualization for a 72-hour forecast:
    - Shaded background rectangles for CPCB AQI categories
    - Historical observed readings (last 48h) in dark gray
    - Median forecast line in bold color
    - 10th to 90th percentile calibrated uncertainty shaded region
    """
    fig = go.Figure()

    # Extract quantiles: q10, q50, q90
    q10 = y_quantiles[:, target_idx, 0]
    q50 = y_quantiles[:, target_idx, 1]
    q90 = y_quantiles[:, target_idx, 2]

    # Determine y-axis maximum
    max_val = max(float(np.max(q90)) * 1.15, 150.0)
    if historical_df is not None and target_name in historical_df.columns:
        hist_max = historical_df[target_name].dropna().max()
        if pd.notna(hist_max):
            max_val = max(max_val, float(hist_max) * 1.15)

    # 1. Add CPCB Category Background Zones
    zones = AQI_ZONES.get(target_name, AQI_ZONES["PM2.5"])
    for low, high, label, fill_col, border_col in zones:
        if low > max_val:
            continue
        y_top = min(high, max_val)
        fig.add_hrect(
            y0=low,
            y1=y_top,
            fillcolor=fill_col,
            line_width=0,
            annotation_text=f" {label}",
            annotation_position="top left",
            annotation_font=dict(size=10, color=border_col),
        )

    # 2. Add Historical Past Observations if provided
    if historical_df is not None and target_name in historical_df.columns:
        hist_slice = historical_df.iloc[-48:]  # Last 48 hours
        fig.add_trace(
            go.Scatter(
                x=hist_slice.index,
                y=hist_slice[target_name],
                mode="lines+markers",
                name="Observed History",
                line=dict(color="#34495E", width=2),
                marker=dict(size=4),
            )
        )

    # 3. Add Uncertainty Envelope: q90 top line, then q10 bottom line with fill
    fig.add_trace(
        go.Scatter(
            x=future_timestamps,
            y=q90,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_timestamps,
            y=q10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(41, 128, 185, 0.20)",
            name="10th - 90th Percentile Range",
            hoverinfo="skip",
        )
    )

    # 4. Add Median Forecast Line
    fig.add_trace(
        go.Scatter(
            x=future_timestamps,
            y=q50,
            mode="lines+markers",
            name=f"Median Forecast ({target_name})",
            line=dict(color="#2980B9", width=3),
            marker=dict(size=5, symbol="circle"),
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>Forecast: %{y:.1f} ug/m3<extra></extra>",
        )
    )

    # 5. Day separation vertical dashed lines
    if len(future_timestamps) >= 48:
        fig.add_vline(x=future_timestamps[23].timestamp() * 1000, line_dash="dash", line_color="#7F8C8D", line_width=1)
        fig.add_vline(x=future_timestamps[47].timestamp() * 1000, line_dash="dash", line_color="#7F8C8D", line_width=1)

    fig.update_layout(
        title=dict(
            text=f"<b>72-Hour {target_name} Concentration Forecast</b> ({station_name})",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Date & Time (IST)",
            showgrid=True,
            gridcolor="#E5E8E8",
        ),
        yaxis=dict(
            title=f"{target_name} Concentration (ug/m3)",
            range=[0, max_val],
            showgrid=True,
            gridcolor="#E5E8E8",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=480,
    )

    return fig


def create_training_loss_plot(history: dict) -> go.Figure:
    """Renders real-time training and validation loss curves over epochs."""
    fig = go.Figure()
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    if "train_loss" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_loss"],
                mode="lines+markers",
                name="Training Loss (Pinball)",
                line=dict(color="#2980B9", width=2),
            )
        )

    if "val_loss" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                mode="lines+markers",
                name="Validation Loss",
                line=dict(color="#E74C3C", width=2),
            )
        )

    fig.update_layout(
        title="Training & Validation Loss Convergence",
        xaxis_title="Epoch",
        yaxis_title="Weighted Pinball Loss",
        hovermode="x unified",
        height=320,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig
