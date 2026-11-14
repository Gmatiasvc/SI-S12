"""Script maestro reproducible (SEED=42) para ejecutar las 16 etapas, guardar artefactos y figuras."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import sklearn

# Permitir ejecución directa como `python src/train_pipeline.py` desde la raíz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.modeling import SEED, run_all_experiments
from src.plots import save_static_figures

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_JSON_PATH = ARTIFACTS_DIR / "results_summary.json"
METRICS_CSV_PATH = ARTIFACTS_DIR / "metrics_comparison.csv"
PREDICTIONS_CSV_PATH = ARTIFACTS_DIR / "predictions_test.csv"


def execute_and_save_all(prefer_local_data: bool = True) -> Dict[str, Any]:
    """Ejecuta el pipeline completo, serializa modelos, métricas, predicciones y gráficos."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, metadata = load_dataset(prefer_local=prefer_local_data)
    results = run_all_experiments(df=df, models_dir=MODELS_DIR, seed=SEED)

    # Guardar tablas en CSV
    metrics_df = pd.DataFrame(results["metrics_table"])
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    pred_test_df: pd.DataFrame = results["predictions_test_df"]
    pred_test_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

    # Generar figuras estáticas de alta resolución
    saved_figs = save_static_figures(df=df, results=results, figures_dir=FIGURES_DIR)

    # Guardar resumen JSON limpio (sin objetos sklearn ni DataFrames)
    summary_payload: Dict[str, Any] = {
        "seed": results["seed"],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "dataset_shape": metadata["shape"],
        "split_summary": results["split_summary"],
        "metrics_table": results["metrics_table"],
        "best_mlp": results["best_mlp"],
        "best_overall": results["best_overall"],
        "selection_reason": results["selection_reason"],
        "mlp_architectures": results["mlp_architectures"],
        "learning_curves": results["learning_curves"],
        "loss_curves_official": results["loss_curves_official"],
        "onehot_ablation": results["onehot_ablation"],
        "leakage_experiment": results["leakage_experiment"],
        "subgroups_test": results["subgroups_test"],
        "saved_figures": [str(p.relative_to(PROJECT_ROOT)) for p in saved_figs],
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    return {
        "df": df,
        "metadata": metadata,
        "results": results,
        "summary_payload": summary_payload,
    }


if __name__ == "__main__":
    out = execute_and_save_all(prefer_local_data=True)
    summary = out["summary_payload"]
    print("=" * 72)
    print("EJECUCIÓN COMPLETA DEL PIPELINE MLP — GRUPO 02 (SEED=42)")
    print("=" * 72)
    print(f"Dataset local cargado: {summary['dataset_shape']['rows']} filas x {summary['dataset_shape']['cols']} columnas")
    print(
        f"Split estratificado: Train={summary['split_summary']['train_rows']}, "
        f"Val={summary['split_summary']['val_rows']}, Test={summary['split_summary']['test_rows']}"
    )
    print("\nTabla comparativa (Validación / CV 5-Fold / Test):")
    df_m = pd.DataFrame(summary["metrics_table"])[
        ["code", "name", "n_params", "val_rmse", "cv_rmse_mean", "test_mae", "test_rmse", "test_r2", "train_time_s"]
    ]
    print(df_m.to_string(index=False))
    print("\nSelección MLP:")
    print(summary["selection_reason"])
    print("\nFiguras generadas:")
    for fig_p in summary["saved_figures"]:
        print(f"  - {fig_p}")
    print("=" * 72)
