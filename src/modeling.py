"""Motor de modelado, validación, experimentos MLP y métricas (Etapas 06–16)."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    FEATURE_COLS,
    TARGET_COL,
    VARIABLE_INFO,
    X6_LABELS,
    X8_LABELS,
)

SEED = 42
TEST_SIZE = 116   # ~15% de 768
VAL_SIZE = 115    # ~15% de 768 -> Train = 537 (~70%), Train+Val = 652 (~85%)
SIMPLICITY_TIE_TOLERANCE_RMSE = 0.05


@dataclass(frozen=True)
class ExperimentSpec:
    """Especificación declarativa de un experimento del plan A–D."""

    exp_id: str
    code: str
    name: str
    family: str
    purpose: str
    use_onehot: bool
    estimator_factory: Any


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula MAE, RMSE y R² sobre vectores reales y predichos."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def build_preprocessor(use_onehot: bool = True) -> ColumnTransformer:
    """Construye el transformador de columnas encapsulado para evitar fuga de datos."""
    if use_onehot:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        categories=[[2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, CONTINUOUS_COLS),
                ("cat", cat_pipe, CATEGORICAL_COLS),
            ],
            remainder="drop",
        )

    all_num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("all_num", all_num_pipe, FEATURE_COLS)],
        remainder="drop",
    )


def count_mlp_parameters(mlp: MLPRegressor) -> Dict[str, Any]:
    """Calcula el desglose exacto de pesos, sesgos y dimensiones por capa de un MLP entrenado."""
    layers: List[Dict[str, Any]] = []
    total_weights = 0
    total_biases = 0

    for idx, (w_mat, b_vec) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        in_dim, out_dim = int(w_mat.shape[0]), int(w_mat.shape[1])
        n_weights = int(w_mat.size)
        n_biases = int(b_vec.size)
        is_output = idx == len(mlp.coefs_) - 1
        activation = "Identidad (Lineal)" if is_output else mlp.activation.upper()
        layer_name = "Capa de Salida" if is_output else f"Capa Oculta {idx + 1}"

        layers.append(
            {
                "layer": layer_name,
                "input_dim": in_dim,
                "output_dim": out_dim,
                "activation": activation,
                "weights": n_weights,
                "biases": n_biases,
                "params": n_weights + n_biases,
                "formula": f"({in_dim} × {out_dim}) + {out_dim} = {n_weights + n_biases}",
            }
        )
        total_weights += n_weights
        total_biases += n_biases

    return {
        "total_params": total_weights + total_biases,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "layers": layers,
    }


def split_dataset(df: pd.DataFrame, seed: int = SEED) -> Dict[str, Any]:
    """Divide en Train (537), Validation (115) y Test (116) estratificando por altura y acristalamiento."""
    strata = df["X5"].astype(str) + "_" + df["X7"].astype(str)

    idx_all = df.index.to_numpy()
    idx_train_val, idx_test = train_test_split(
        idx_all,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=strata.loc[idx_all],
    )
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=strata.loc[idx_train_val],
    )

    train_df = df.loc[idx_train].reset_index(drop=True)
    val_df = df.loc[idx_val].reset_index(drop=True)
    test_df = df.loc[idx_test].reset_index(drop=True)
    train_val_df = df.loc[idx_train_val].reset_index(drop=True)

    # Analizar presencia de las 12 formas arquitectónicas base (X1..X5)
    geom_cols = ["X1", "X2", "X3", "X4", "X5"]
    n_geoms_total = int(df[geom_cols].drop_duplicates().shape[0])
    n_geoms_train = int(train_df[geom_cols].drop_duplicates().shape[0])
    n_geoms_val = int(val_df[geom_cols].drop_duplicates().shape[0])
    n_geoms_test = int(test_df[geom_cols].drop_duplicates().shape[0])

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_val_df": train_val_df,
        "summary": {
            "seed": seed,
            "total_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_val_rows": int(len(train_val_df)),
            "stratified_by": "X5 (altura) + X7 (área acristalada)",
            "unique_base_geometries": {
                "total": n_geoms_total,
                "train": n_geoms_train,
                "val": n_geoms_val,
                "test": n_geoms_test,
            },
        },
    }


def get_experiment_specs(seed: int = SEED) -> List[ExperimentSpec]:
    """Define los experimentos oficiales A, B, C y D de la guía."""
    return [
        ExperimentSpec(
            exp_id="exp_d0_dummy",
            code="D0",
            name="DummyRegressor (Media)",
            family="Baseline",
            purpose="Referencia mínima sin aprender relaciones",
            use_onehot=True,
            estimator_factory=lambda: DummyRegressor(strategy="mean"),
        ),
        ExperimentSpec(
            exp_id="exp_d1_linear",
            code="D1",
            name="Regresión Lineal (OneHot)",
            family="Baseline",
            purpose="Baseline lineal con codificación categórica",
            use_onehot=True,
            estimator_factory=lambda: LinearRegression(),
        ),
        ExperimentSpec(
            exp_id="exp_d2_rf",
            code="D2",
            name="Random Forest (200 árboles)",
            family="Baseline",
            purpose="Baseline no lineal basado en árboles",
            use_onehot=True,
            estimator_factory=lambda: RandomForestRegressor(
                n_estimators=200, random_state=seed, n_jobs=-1
            ),
        ),
        ExperimentSpec(
            exp_id="exp_a_mlp_16",
            code="A",
            name="MLP (16), ReLU",
            family="MLP",
            purpose="Arquitectura pequeña (1 capa oculta)",
            use_onehot=True,
            estimator_factory=lambda: MLPRegressor(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.005,
                max_iter=2000,
                tol=1e-5,
                n_iter_no_change=50,
                random_state=seed,
            ),
        ),
        ExperimentSpec(
            exp_id="exp_b_mlp_32_16",
            code="B",
            name="MLP (32, 16), ReLU",
            family="MLP",
            purpose="Efecto de profundidad (2 capas ocultas)",
            use_onehot=True,
            estimator_factory=lambda: MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.005,
                max_iter=2000,
                tol=1e-5,
                n_iter_no_change=50,
                random_state=seed,
            ),
        ),
        ExperimentSpec(
            exp_id="exp_c1_mlp_64_32_noreg",
            code="C1",
            name="MLP (64, 32), alpha=1e-6",
            family="MLP",
            purpose="Alta capacidad con regularización mínima",
            use_onehot=True,
            estimator_factory=lambda: MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-6,
                learning_rate_init=0.005,
                max_iter=2000,
                tol=1e-6,
                n_iter_no_change=100,
                random_state=seed,
            ),
        ),
        ExperimentSpec(
            exp_id="exp_c2_mlp_64_32_l2",
            code="C2",
            name="MLP (64, 32), L2 alpha=0.05",
            family="MLP",
            purpose="Alta capacidad + regularización L2",
            use_onehot=True,
            estimator_factory=lambda: MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.05,
                learning_rate_init=0.005,
                max_iter=2000,
                tol=1e-5,
                n_iter_no_change=50,
                random_state=seed,
            ),
        ),
        ExperimentSpec(
            exp_id="exp_c3_mlp_64_32_es",
            code="C3",
            name="MLP (64, 32), L2 + Early Stopping",
            family="MLP",
            purpose="Capacidad alta + L2 + parada temprana",
            use_onehot=True,
            estimator_factory=lambda: MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.05,
                learning_rate_init=0.01,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=60,
                max_iter=2000,
                tol=1e-4,
                random_state=seed,
            ),
        ),
    ]


def trace_mlp_learning_curves(
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    max_epochs: int = 600,
    eval_every: int = 10,
) -> Dict[str, Any]:
    """Registra la evolución época a época de RMSE en Train y Validation sin tocar Test."""
    preprocessor = build_preprocessor(use_onehot=spec.use_onehot)
    X_tr_proc = preprocessor.fit_transform(X_train)
    X_va_proc = preprocessor.transform(X_val)
    y_tr_arr = y_train.to_numpy(dtype=float)
    y_va_arr = y_val.to_numpy(dtype=float)

    base_mlp = spec.estimator_factory()
    tracer = MLPRegressor(
        hidden_layer_sizes=base_mlp.hidden_layer_sizes,
        activation=base_mlp.activation,
        solver="adam",
        alpha=base_mlp.alpha,
        learning_rate_init=base_mlp.learning_rate_init,
        early_stopping=False,
        warm_start=True,
        max_iter=1,
        random_state=SEED,
    )

    epochs: List[int] = []
    train_rmse: List[float] = []
    val_rmse: List[float] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            tracer.fit(X_tr_proc, y_tr_arr)
            if epoch == 1 or epoch % eval_every == 0 or epoch == max_epochs:
                pred_tr = tracer.predict(X_tr_proc)
                pred_va = tracer.predict(X_va_proc)
                epochs.append(epoch)
                train_rmse.append(float(np.sqrt(mean_squared_error(y_tr_arr, pred_tr))))
                val_rmse.append(float(np.sqrt(mean_squared_error(y_va_arr, pred_va))))

    best_val_idx = int(np.argmin(val_rmse))
    return {
        "epochs": epochs,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "best_epoch_val": epochs[best_val_idx],
        "best_val_rmse": val_rmse[best_val_idx],
        "final_train_rmse": train_rmse[-1],
        "final_val_rmse": val_rmse[-1],
        "overfit_gap_end": float(val_rmse[-1] - train_rmse[-1]),
    }


def evaluate_onehot_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Compara el impacto real de usar OneHotEncoder en X6 y X8 frente a tratarlas como continuas."""
    candidates = [
        ("Regresión Lineal", lambda: LinearRegression()),
        (
            "Random Forest (200)",
            lambda: RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        ),
        (
            "MLP (32, 16), ReLU",
            lambda: MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.005,
                max_iter=1500,
                tol=1e-5,
                n_iter_no_change=50,
                random_state=seed,
            ),
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for model_label, factory in candidates:
        for use_ohe in [False, True]:
            pipe = Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(use_onehot=use_ohe)),
                    ("model", factory()),
                ]
            )
            pipe.fit(X_train, y_train)
            val_pred = pipe.predict(X_val)
            m_val = compute_metrics(y_val.to_numpy(), val_pred)
            n_features_out = int(pipe.named_steps["preprocess"].transform(X_train.iloc[:1]).shape[1])
            rows.append(
                {
                    "modelo": model_label,
                    "codificacion": "Con OneHotEncoder (X6, X8)" if use_ohe else "Sin OneHot (numéricas crudas)",
                    "use_onehot": use_ohe,
                    "n_features_codificadas": n_features_out,
                    "val_mae": round(m_val["mae"], 4),
                    "val_rmse": round(m_val["rmse"], 4),
                    "val_r2": round(m_val["r2"], 4),
                }
            )
    return rows


def evaluate_y2_leakage_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Demuestra la fuga de datos (data leakage) si se incluye Y2 como predictor de Y1."""
    cols_with_y2 = FEATURE_COLS + ["Y2"]
    X_tr = train_df[cols_with_y2]
    y_tr = train_df[TARGET_COL]
    X_va = val_df[cols_with_y2]
    y_va = val_df[TARGET_COL]

    pre_leak = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_COLS + ["Y2"]),
            (
                "cat",
                OneHotEncoder(
                    categories=[[2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )

    results: Dict[str, Any] = {}
    for name, est in [
        ("Regresión Lineal + Y2 (fuga)", LinearRegression()),
        (
            "MLP (32,16) + Y2 (fuga)",
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                learning_rate_init=0.005,
                max_iter=1500,
                random_state=seed,
            ),
        ),
    ]:
        pipe = Pipeline([("preprocess", clone(pre_leak)), ("model", est)])
        pipe.fit(X_tr, y_tr)
        m = compute_metrics(y_va.to_numpy(), pipe.predict(X_va))
        results[name] = {k: round(v, 4) for k, v in m.items()}

    corr_y1_y2 = float(train_df["Y1"].corr(train_df["Y2"]))
    results["pearson_y1_y2_train"] = round(corr_y1_y2, 4)
    return results


def compute_subgroup_metrics(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
) -> Dict[str, List[Dict[str, Any]]]:
    """Calcula MAE, RMSE y R² en Test por subgrupos de Altura, Orientación y Área Acristalada."""
    eval_df = test_df.copy()
    eval_df["y_pred"] = y_pred
    eval_df["abs_err"] = np.abs(eval_df[TARGET_COL] - eval_df["y_pred"])
    eval_df["err"] = eval_df[TARGET_COL] - eval_df["y_pred"]

    subgroup_defs = [
        ("X5", "Altura total (m)", lambda v: f"{v:.1f} m"),
        ("X6", "Orientación", lambda v: X6_LABELS.get(int(v), str(v))),
        ("X7", "Área acristalada", lambda v: f"{int(round(v * 100))}% ({v:.2f})"),
    ]

    output: Dict[str, List[Dict[str, Any]]] = {}
    for col, group_title, fmt in subgroup_defs:
        rows = []
        for val, grp in eval_df.groupby(col):
            m = compute_metrics(grp[TARGET_COL].to_numpy(), grp["y_pred"].to_numpy())
            rows.append(
                {
                    "subgrupo": fmt(val),
                    "valor_raw": float(val),
                    "n_test": int(len(grp)),
                    "y1_medio": round(float(grp[TARGET_COL].mean()), 3),
                    "mae": round(m["mae"], 4),
                    "rmse": round(m["rmse"], 4),
                    "r2": round(m["r2"], 4),
                    "sesgo_medio": round(float(grp["err"].mean()), 4),
                }
            )
        output[group_title] = rows
    return output


def select_best_mlp(results_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """Selecciona el mejor MLP por RMSE validado con criterio explícito de desempate por simplicidad."""
    mlp_rows = [r for r in results_list if r["family"] == "MLP"]
    mlp_sorted = sorted(mlp_rows, key=lambda r: (r["val_rmse"], r["n_params"]))
    best_raw = mlp_sorted[0]

    # Buscar si algún MLP más simple (menos parámetros) está dentro de la tolerancia de empate
    tied_candidates = [
        r
        for r in mlp_rows
        if (r["val_rmse"] - best_raw["val_rmse"]) <= SIMPLICITY_TIE_TOLERANCE_RMSE
    ]
    tied_candidates_sorted = sorted(tied_candidates, key=lambda r: (r["n_params"], r["val_rmse"]))
    winner = tied_candidates_sorted[0]

    if winner["exp_id"] == best_raw["exp_id"]:
        reason = (
            f"Seleccionado {winner['name']} por obtener el menor RMSE en Validación "
            f"({winner['val_rmse']:.4f} kWh/m²) y CV ({winner['cv_rmse_mean']:.4f} kWh/m²)."
        )
    else:
        reason = (
            f"Empate técnico dentro de ΔRMSE ≤ {SIMPLICITY_TIE_TOLERANCE_RMSE} kWh/m² "
            f"({best_raw['name']}: {best_raw['val_rmse']:.4f} vs {winner['name']}: {winner['val_rmse']:.4f}); "
            f"se elige {winner['name']} por simplicidad ({winner['n_params']} vs {best_raw['n_params']} parámetros)."
        )
    return winner, reason


def run_all_experiments(
    df: pd.DataFrame,
    models_dir: Optional[Any] = None,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Ejecuta el protocolo completo de experimentación (Etapas 06–15) y devuelve resultados reales."""
    split = split_dataset(df, seed=seed)
    train_df = split["train_df"]
    val_df = split["val_df"]
    test_df = split["test_df"]
    train_val_df = split["train_val_df"]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    X_train_val, y_train_val = train_val_df[FEATURE_COLS], train_val_df[TARGET_COL]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    specs = get_experiment_specs(seed=seed)

    results_rows: List[Dict[str, Any]] = []
    pipelines: Dict[str, Pipeline] = {}
    mlp_architectures: Dict[str, Any] = {}
    learning_curves: Dict[str, Any] = {}
    loss_curves_official: Dict[str, List[float]] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    for spec in specs:
        pre = build_preprocessor(use_onehot=spec.use_onehot)
        est = spec.estimator_factory()
        pipe = Pipeline(steps=[("preprocess", pre), ("model", est)])

        # 1) Validación Cruzada 5-Fold sobre Train+Val (sin tocar Test)
        cv_scores = cross_validate(
            clone(pipe),
            X_train_val,
            y_train_val,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )

        # 2) Entrenamiento en Train y evaluación en Train y Validation
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time_s = float(time.perf_counter() - t0)

        pred_train = pipe.predict(X_train)
        pred_val = pipe.predict(X_val)
        m_train = compute_metrics(y_train.to_numpy(), pred_train)
        m_val = compute_metrics(y_val.to_numpy(), pred_val)

        # 3) Evaluación en Test (solo para registro final de Etapa 15; jamás usada para seleccionar)
        pred_test = pipe.predict(X_test)
        m_test = compute_metrics(y_test.to_numpy(), pred_test)
        test_predictions[spec.exp_id] = pred_test

        model_obj = pipe.named_steps["model"]
        n_features_in = int(pipe.named_steps["preprocess"].transform(X_train.iloc[:1]).shape[1])

        if spec.family == "MLP":
            arch_info = count_mlp_parameters(model_obj)
            n_params = arch_info["total_params"]
            n_iter = int(model_obj.n_iter_)
            mlp_architectures[spec.exp_id] = {
                "exp_id": spec.exp_id,
                "code": spec.code,
                "name": spec.name,
                "hidden_layers": list(model_obj.hidden_layer_sizes),
                "alpha": float(model_obj.alpha),
                "learning_rate_init": float(model_obj.learning_rate_init),
                "early_stopping": bool(model_obj.early_stopping),
                "n_iter": n_iter,
                **arch_info,
            }
            loss_curves_official[spec.exp_id] = [float(v) for v in model_obj.loss_curve_]
            learning_curves[spec.exp_id] = trace_mlp_learning_curves(
                spec=spec,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                max_epochs=600,
                eval_every=10,
            )
        else:
            if isinstance(model_obj, LinearRegression):
                n_params = int(model_obj.coef_.size + 1)
            elif isinstance(model_obj, DummyRegressor):
                n_params = 1
            else:
                n_params = int(sum(t.tree_.node_count for t in model_obj.estimators_))
            n_iter = 1

        pipelines[spec.exp_id] = pipe
        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, models_dir / f"{spec.exp_id}.joblib")

        results_rows.append(
            {
                "exp_id": spec.exp_id,
                "code": spec.code,
                "name": spec.name,
                "family": spec.family,
                "purpose": spec.purpose,
                "n_features_encoded": n_features_in,
                "n_params": n_params,
                "n_iter": n_iter,
                "train_time_s": round(train_time_s, 4),
                "train_mae": round(m_train["mae"], 4),
                "train_rmse": round(m_train["rmse"], 4),
                "train_r2": round(m_train["r2"], 4),
                "val_mae": round(m_val["mae"], 4),
                "val_rmse": round(m_val["rmse"], 4),
                "val_r2": round(m_val["r2"], 4),
                "cv_mae_mean": round(float(-cv_scores["test_mae"].mean()), 4),
                "cv_mae_std": round(float(cv_scores["test_mae"].std()), 4),
                "cv_rmse_mean": round(float(-cv_scores["test_rmse"].mean()), 4),
                "cv_rmse_std": round(float(cv_scores["test_rmse"].std()), 4),
                "cv_r2_mean": round(float(cv_scores["test_r2"].mean()), 4),
                "test_mae": round(m_test["mae"], 4),
                "test_rmse": round(m_test["rmse"], 4),
                "test_r2": round(m_test["r2"], 4),
                "diff_test_val_rmse": round(m_test["rmse"] - m_val["rmse"], 4),
                "diff_test_val_mae": round(m_test["mae"] - m_val["mae"], 4),
            }
        )

    best_mlp, selection_reason = select_best_mlp(results_rows)
    best_overall = sorted(results_rows, key=lambda r: r["val_rmse"])[0]

    # Guardar predicciones en test para todos los modelos
    pred_test_df = test_df.copy()
    for exp_id, preds in test_predictions.items():
        pred_test_df[f"pred_{exp_id}"] = np.round(preds, 4)
        pred_test_df[f"res_{exp_id}"] = np.round(pred_test_df[TARGET_COL] - preds, 4)

    # Análisis por subgrupos para el mejor MLP y para el mejor modelo no neuronal (RF y Linear)
    subgroups = {
        "best_mlp": {
            "exp_id": best_mlp["exp_id"],
            "name": best_mlp["name"],
            "groups": compute_subgroup_metrics(test_df, test_predictions[best_mlp["exp_id"]]),
        },
        "exp_d2_rf": {
            "exp_id": "exp_d2_rf",
            "name": "Random Forest (200 árboles)",
            "groups": compute_subgroup_metrics(test_df, test_predictions["exp_d2_rf"]),
        },
        "exp_d1_linear": {
            "exp_id": "exp_d1_linear",
            "name": "Regresión Lineal (OneHot)",
            "groups": compute_subgroup_metrics(test_df, test_predictions["exp_d1_linear"]),
        },
    }

    onehot_ablation = evaluate_onehot_ablation(X_train, y_train, X_val, y_val, seed=seed)
    leakage_test = evaluate_y2_leakage_experiment(train_df, val_df, seed=seed)

    return {
        "seed": seed,
        "split_summary": split["summary"],
        "metrics_table": results_rows,
        "best_mlp": best_mlp,
        "best_overall": best_overall,
        "selection_reason": selection_reason,
        "mlp_architectures": mlp_architectures,
        "learning_curves": learning_curves,
        "loss_curves_official": loss_curves_official,
        "onehot_ablation": onehot_ablation,
        "leakage_experiment": leakage_test,
        "subgroups_test": subgroups,
        "predictions_test_df": pred_test_df,
        "pipelines": pipelines,
    }


def check_domain_and_geometry(inputs: Dict[str, float]) -> Dict[str, Any]:
    """Verifica si una entrada está dentro del dominio del entrenamiento y es geométricamente coherente."""
    warnings: List[str] = []
    out_of_range_vars: List[str] = []

    for col in FEATURE_COLS:
        val = float(inputs[col])
        v_min = float(VARIABLE_INFO[col]["min"])
        v_max = float(VARIABLE_INFO[col]["max"])
        if val < v_min or val > v_max:
            out_of_range_vars.append(col)
            warnings.append(
                f"{col} ({VARIABLE_INFO[col]['name']}) = {val:.3g} está FUERA del rango "
                f"de entrenamiento [{v_min}, {v_max}]. La predicción es una extrapolación."
            )

    # Valores discretos observados en el diseño factorial de UCI Energy Efficiency
    if float(inputs["X5"]) not in (3.5, 7.0):
        warnings.append(
            f"X5 (Altura) = {inputs['X5']:.2f} m es un valor intermedio; el dataset solo "
            "contiene edificios de 3.5 m (1 planta) y 7.0 m (2 plantas)."
        )

    if int(round(inputs["X6"])) not in X6_LABELS:
        warnings.append(
            f"X6 (Orientación) = {inputs['X6']} no corresponde a los códigos válidos (2, 3, 4, 5)."
        )

    if int(round(inputs["X8"])) not in X8_LABELS:
        warnings.append(
            f"X8 (Distribución de acristalamiento) = {inputs['X8']} no corresponde a (0–5)."
        )

    if float(inputs["X7"]) == 0.0 and int(round(inputs["X8"])) != 0:
        warnings.append(
            "Inconsistencia de diseño: X7 = 0.0 (sin área acristalada) suele corresponder a X8 = 0."
        )
    elif float(inputs["X7"]) > 0.0 and int(round(inputs["X8"])) == 0:
        warnings.append(
            "Inconsistencia de diseño: X7 > 0.0 (con ventanas), pero X8 = 0 indica 'Sin acristalamiento'."
        )

    # Identidad geométrica exacta de los edificios simulados: X2 = X3 + 2 * X4
    expected_x2 = float(inputs["X3"]) + 2.0 * float(inputs["X4"])
    diff_x2 = abs(float(inputs["X2"]) - expected_x2)
    if diff_x2 > 5.0:
        warnings.append(
            f"Coherencia geométrica: en los edificios del dataset se cumple X2 ≈ X3 + 2·X4 "
            f"(esperado ≈ {expected_x2:.1f} m², ingresado {inputs['X2']:.1f} m²; Δ = {diff_x2:.1f} m²)."
        )

    return {
        "is_in_domain": len(out_of_range_vars) == 0,
        "has_warnings": len(warnings) > 0,
        "out_of_range_vars": out_of_range_vars,
        "warnings": warnings,
        "expected_x2_from_geometry": round(expected_x2, 2),
    }
