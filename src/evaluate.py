"""
Model Evaluation and Benchmark Framework.
Implements PRD Section 7.6 (Evaluation Metrics):
- Day+1, Day+2, Day+3, and Overall (1-72h) horizon metrics
- MAE, RMSE, sMAPE, R-squared, Bias
- Skill score relative to persistence baseline
- Formatted markdown report generator and JSON exporter
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.models.baselines import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    TabularGBDTBaseline,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics across flattened actual vs predicted arrays.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan, "smape": np.nan, "r2": np.nan, "bias": np.nan}

    yt = y_true[mask]
    yp = y_pred[mask]

    diff = yp - yt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))

    # sMAPE: 200 * |y - y_hat| / (|y| + |y_hat| + eps)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    smape = float(np.mean(200.0 * np.abs(diff) / denom))

    # R-squared
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    ss_res = np.sum(diff ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-6))) if ss_tot > 1e-6 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "smape": round(smape, 2),
        "r2": round(r2, 3),
        "bias": round(bias, 2),
    }


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ["PM2.5", "PM10"],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluates multi-horizon forecast across Day+1 (1-24h), Day+2 (25-48h), Day+3 (49-72h),
    and Overall (1-72h) for each pollutant target.
    
    Args:
        y_true: Array of shape (N, 72, n_targets)
        y_pred: Array of shape (N, 72, n_targets)
    Returns:
        Nested dict: {target_name: {horizon_name: metrics_dict}}
    """
    horizons = {
        "Day+1": (0, 24),
        "Day+2": (24, 48),
        "Day+3": (48, 72),
        "Overall (72h)": (0, 72),
    }

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for t_idx, target in enumerate(target_names):
        results[target] = {}
        for h_name, (h_start, h_end) in horizons.items():
            yt_slice = y_true[:, h_start:h_end, t_idx]
            yp_slice = y_pred[:, h_start:h_end, t_idx]
            metrics = compute_metrics(yt_slice, yp_slice)
            results[target][h_name] = metrics

    return results


def calculate_skill_scores(
    model_eval: Dict[str, Dict[str, Dict[str, float]]],
    persistence_eval: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    Computes Skill Score against Persistence:
        Skill Score = 1 - (MAE_model / MAE_persistence)
    Must be positive for all 3 days to pass acceptance criteria.
    """
    skill_scores: Dict[str, Dict[str, float]] = {}

    for target in model_eval.keys():
        skill_scores[target] = {}
        for horizon in ["Day+1", "Day+2", "Day+3", "Overall (72h)"]:
            model_mae = model_eval[target][horizon]["mae"]
            pers_mae = persistence_eval[target][horizon]["mae"]
            if pers_mae > 0:
                score = round(1.0 - (model_mae / pers_mae), 3)
            else:
                score = 0.0
            skill_scores[target][horizon] = score

    return skill_scores


def generate_benchmark_markdown_table(
    eval_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    skill_dict: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> str:
    """
    Generates a clear markdown benchmark comparison table across models.
    eval_dict: {model_name: {target: {horizon: metrics}}}
    """
    lines = [
        "## Forecast Benchmark Comparison Table",
        "",
        "| Model | Target | Horizon | MAE (ug/m3) | RMSE (ug/m3) | sMAPE (%) | R² | Skill Score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for model_name, targets in eval_dict.items():
        for target_name, horizons in targets.items():
            for h_name in ["Day+1", "Day+2", "Day+3", "Overall (72h)"]:
                m = horizons[h_name]
                skill_str = "N/A (Ref)"
                if skill_dict and model_name in skill_dict:
                    skill_val = skill_dict[model_name][target_name].get(h_name, 0.0)
                    skill_badge = "✅" if skill_val > 0 else "❌"
                    skill_str = f"{skill_badge} {skill_val:+.3f}"

                lines.append(
                    f"| {model_name} | {target_name} | {h_name} | {m['mae']} | "
                    f"{m['rmse']} | {m['smape']}% | {m['r2']} | {skill_str} |"
                )

    return "\n".join(lines)


def run_benchmark_comparison(
    X_test: np.ndarray,
    Y_test: np.ndarray,
    target_names: List[str] = ["PM2.5", "PM10"],
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Runs all baseline models on test data, evaluates performance, computes skill scores,
    and returns metrics and formatted markdown.
    """
    # 1. Persistence Baseline
    pers_model = PersistenceForecaster(horizon=72)
    y_pred_pers = pers_model.predict(X_test, target_indices=[0, 1])
    eval_pers = evaluate_forecast(Y_test, y_pred_pers, target_names=target_names)

    # 2. Seasonal Naive Baseline
    sn_model = SeasonalNaiveForecaster(horizon=72)
    y_pred_sn = sn_model.predict(X_test, target_indices=[0, 1])
    eval_sn = evaluate_forecast(Y_test, y_pred_sn, target_names=target_names)

    # 3. Tabular GBDT Baseline
    gbdt_model = TabularGBDTBaseline(horizon=72, target_names=target_names)
    gbdt_model.fit(X_test[: int(len(X_test) * 0.7)], Y_test[: int(len(Y_test) * 0.7)])
    y_pred_gbdt = gbdt_model.predict(X_test)
    eval_gbdt = evaluate_forecast(Y_test, y_pred_gbdt, target_names=target_names)

    # Aggregate evaluation
    all_evals = {
        "Persistence": eval_pers,
        "Seasonal-Naive": eval_sn,
        "LightGBM (GBDT)": eval_gbdt,
    }

    # Compute skill scores relative to persistence
    all_skills = {
        "Persistence": calculate_skill_scores(eval_pers, eval_pers),
        "Seasonal-Naive": calculate_skill_scores(eval_sn, eval_pers),
        "LightGBM (GBDT)": calculate_skill_scores(eval_gbdt, eval_pers),
    }

    report_md = generate_benchmark_markdown_table(all_evals, all_skills)

    benchmark_bundle = {
        "evaluations": all_evals,
        "skill_scores": all_skills,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(benchmark_bundle, f, indent=2)

    return benchmark_bundle, report_md
