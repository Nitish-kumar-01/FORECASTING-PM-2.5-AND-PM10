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
    Creates an interactive Plotly visualization displaying:
    - Complete uploaded historical timeline (all 15 days or 30 days uploaded)
    - Seamless continuous transition into the next 3 days (72-hour) forecast line
    - Calibrated 10th-to-90th percentile prediction intervals
    - CPCB regulatory colored background bands
    - Interactive date range selectors (3 Days, 7 Days, 15 Days, All) & range slider
    """
    fig = go.Figure()

    # Extract quantiles: q10, q50, q90
    q10 = y_quantiles[:, target_idx, 0]
    q50 = y_quantiles[:, target_idx, 1]
    q90 = y_quantiles[:, target_idx, 2]

    # Determine y-axis maximum across historical observations and predicted quantiles
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

    # 2. Add Complete Uploaded Historical Data (Full 15-day / 30-day uploaded dataset)
    last_hist_ts = None
    last_hist_val = None
    if historical_df is not None and target_name in historical_df.columns:
        hist_clean = historical_df[target_name].dropna()
        if len(hist_clean) > 0:
            last_hist_ts = hist_clean.index[-1]
            last_hist_val = float(hist_clean.iloc[-1])
            days_span = (hist_clean.index.max() - hist_clean.index.min()).total_seconds() / 86400.0
            hist_label = f"Uploaded Observed History ({days_span:.1f} Days, {len(hist_clean)} Records)"
            fig.add_trace(
                go.Scatter(
                    x=hist_clean.index,
                    y=hist_clean.values,
                    mode="lines",
                    name=hist_label,
                    line=dict(color="#2C3E50", width=2),
                    hovertemplate="<b>Observed: %{x|%d %b %H:%M}</b><br>Concentration: %{y:.1f} ug/m3<extra></extra>",
                )
            )

    # 3. Seamless connection: anchor forecast line to last observed point
    if last_hist_ts is not None and last_hist_val is not None:
        fc_x = [last_hist_ts] + list(future_timestamps)
        fc_q10 = [last_hist_val] + list(q10)
        fc_q50 = [last_hist_val] + list(q50)
        fc_q90 = [last_hist_val] + list(q90)
    else:
        fc_x = list(future_timestamps)
        fc_q10 = list(q10)
        fc_q50 = list(q50)
        fc_q90 = list(q90)

    # 4. Uncertainty Envelope (10th to 90th percentile)
    fig.add_trace(
        go.Scatter(
            x=fc_x,
            y=fc_q90,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_x,
            y=fc_q10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(41, 128, 185, 0.20)",
            name="10th - 90th Percentile Range",
            hoverinfo="skip",
        )
    )

    # 5. Continuous Median Forecast Line (Next 3 Days)
    fig.add_trace(
        go.Scatter(
            x=fc_x,
            y=fc_q50,
            mode="lines+markers",
            name=f"Next 3 Days Forecast ({target_name})",
            line=dict(color="#2980B9", width=3.5),
            marker=dict(size=4, symbol="circle"),
            hovertemplate="<b>Forecast: %{x|%d %b %H:%M}</b><br>Predicted: %{y:.1f} ug/m3<extra></extra>",
        )
    )

    # 6. Add Forecast Start Divider Line & Day Separation Lines
    if last_hist_ts is not None:
        fig.add_vline(
            x=last_hist_ts.isoformat(),
            line_dash="solid",
            line_color="#2C3E50",
            line_width=2,
            annotation_text=" 🔮 Next 3 Days Forecast Starts",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#2C3E50"),
        )

    if len(future_timestamps) >= 24:
        fig.add_vline(
            x=future_timestamps[23].isoformat(),
            line_dash="dot",
            line_color="#7F8C8D",
            line_width=1,
            annotation_text=" End Day+1",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color="#7F8C8D"),
        )
    if len(future_timestamps) >= 48:
        fig.add_vline(
            x=future_timestamps[47].isoformat(),
            line_dash="dot",
            line_color="#7F8C8D",
            line_width=1,
            annotation_text=" End Day+2",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color="#7F8C8D"),
        )

    # 7. Layout with Range Selector Buttons & Range Slider for 15/30-day navigation
    fig.update_layout(
        title=dict(
            text=f"<b>Complete Observation History & 72-Hour {target_name} Forecast</b> ({station_name})",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Date & Time (IST)",
            showgrid=True,
            gridcolor="#E5E8E8",
            type="date",
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="Forecast Focus (3 Days)", step="day", stepmode="backward"),
                    dict(count=7, label="Last 7 Days + Forecast", step="day", stepmode="backward"),
                    dict(count=15, label="Last 15 Days + Forecast", step="day", stepmode="backward"),
                    dict(count=30, label="Last 1 Month + Forecast", step="day", stepmode="backward"),
                    dict(step="all", label="Full History (All)"),
                ]),
                bgcolor="#EAEDED",
                activecolor="#2980B9",
            ),
            rangeslider=dict(visible=True, thickness=0.07),
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
            y=1.03,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=65, b=40),
        height=520,
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
