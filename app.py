"""Dashboard moderno en Streamlit para el proyecto MLP de Carga Térmica (Grupo 02)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    FEATURE_COLS,
    TARGET_COL,
    VARIABLE_INFO,
    X6_LABELS,
    X8_LABELS,
    get_variable_summary_df,
    load_dataset,
)
from src.modeling import SEED, check_domain_and_geometry
from src.plots import (
    PALETTE,
    plotly_correlation_heatmap,
    plotly_eda_categorical,
    plotly_eda_scatter_compactness,
    plotly_learning_curves,
    plotly_mlp_architecture_diagram,
    plotly_model_comparison,
    plotly_real_vs_pred_and_residuals,
    plotly_y1_distribution,
)
from src.train_pipeline import (
    MODELS_DIR,
    PREDICTIONS_CSV_PATH,
    RESULTS_JSON_PATH,
    execute_and_save_all,
)

st.set_page_config(
    page_title="Grupo 02 · Carga Térmica con MLP",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main {
        background-color: #F8FAFC;
    }
    .hero-banner {
        background: linear-gradient(120deg, #0F5E4D 0%, #166F5C 55%, #1E3A5F 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -10px rgba(15, 94, 77, 0.45);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 750;
        margin: 0.15rem 0 0.35rem 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.98rem;
        opacity: 0.92;
        margin: 0;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #0F5E4D;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .kpi-card.accent { border-left-color: #E07A5F; }
    .kpi-card.navy { border-left-color: #1E3A5F; }
    .kpi-card.amber { border-left-color: #D99B26; }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
    }
    .kpi-note {
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.25rem;
    }
    .info-box {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #0F5E4D;
        padding: 0.95rem 1.15rem;
        border-radius: 10px;
        color: #064E3B;
        margin: 0.6rem 0;
    }
    .warning-box {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        padding: 0.95rem 1.15rem;
        border-radius: 10px;
        color: #78350F;
        margin: 0.6rem 0;
    }
    .danger-box {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        padding: 0.95rem 1.15rem;
        border-radius: 10px;
        color: #7F1D1D;
        margin: 0.6rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def ensure_artifacts_ready() -> None:
    """Garantiza carga autocontenida: si faltan artefactos locales, los genera automáticamente."""
    if not RESULTS_JSON_PATH.exists() or not PREDICTIONS_CSV_PATH.exists():
        with st.spinner("Inicializando por primera vez artefactos y modelos locales (SEED=42)..."):
            execute_and_save_all(prefer_local_data=True)


@st.cache_data(show_spinner=False)
def load_all_local_state() -> Dict[str, Any]:
    """Carga datos, metadatos, resultados y predicciones en caché para respuesta instantánea."""
    df, metadata = load_dataset(prefer_local=True)
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    pred_test_df = pd.read_csv(PREDICTIONS_CSV_PATH)
    return {
        "df": df,
        "metadata": metadata,
        "summary": summary,
        "pred_test_df": pred_test_df,
    }


@st.cache_resource(show_spinner=False)
def load_trained_pipelines() -> Dict[str, Any]:
    """Carga los pipelines serializados con joblib desde artifacts/models/."""
    pipelines: Dict[str, Any] = {}
    for model_file in sorted(MODELS_DIR.glob("*.joblib")):
        pipelines[model_file.stem] = joblib.load(model_file)
    return pipelines


def kpi_card(label: str, value: str, note: str, variant: str = "") -> str:
    cls = f"kpi-card {variant}".strip()
    return (
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div>'
        f"</div>"
    )


def render_hero(badge: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-badge">{badge}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


ensure_artifacts_ready()
state = load_all_local_state()
pipelines = load_trained_pipelines()

df: pd.DataFrame = state["df"]
metadata: Dict[str, Any] = state["metadata"]
summary: Dict[str, Any] = state["summary"]
pred_test_df: pd.DataFrame = state["pred_test_df"]

best_mlp = summary["best_mlp"]
best_overall = summary["best_overall"]

# Para gráficos reutilizamos la estructura esperada por src/plots.py
results_for_plots = {
    **summary,
    "predictions_test_df": pred_test_df,
}

# ---------------------------------------------------------------------------
# SIDEBAR DE NAVEGACIÓN Y TRAZABILIDAD
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Grupo 02 · MLP Térmico")
    st.caption("Sistemas Inteligentes y Machine Learning")

    pages = [
        "🏠 Inicio & Contexto (Etapas 01–04)",
        "📊 Exploración de Datos - EDA (Etapa 05)",
        "🧠 Arquitectura MLP & Preprocesamiento (Etapas 06–10)",
        "🔬 Experimentos, Curvas y Selección (Etapas 11–14)",
        "🎯 Evaluación Final & Residuos (Etapa 15)",
        "🏢 Simulador & Predicción en Vivo (Etapa 16)",
        "🎓 Sustentación & Preguntas del Docente",
    ]
    selected_page = st.radio("Navegación por etapas", pages, label_visibility="collapsed")

    st.divider()
    st.markdown("**Trazabilidad local**")
    st.markdown(
        f"- Dataset: `data/ENB2012_data.xlsx` ({metadata['shape']['rows']}×{metadata['shape']['cols']})\n"
        f"- Semilla global: `SEED = {summary['seed']}`\n"
        f"- Mejor MLP: **{best_mlp['name']}**\n"
        f"- RMSE Test (MLP): `{best_mlp['test_rmse']:.4f} kWh/m²`\n"
        f"- R² Test (MLP): `{best_mlp['test_r2']:.4f}`\n"
        f"- Python `{summary['environment']['python']}` · sklearn `{summary['environment']['scikit_learn']}`"
    )
    st.divider()
    if st.button("🔄 Re-entrenar todo (SEED=42)", width="stretch"):
        with st.spinner("Re-ejecutando las 16 etapas desde cero..."):
            execute_and_save_all(prefer_local_data=True)
            st.cache_data.clear()
            st.cache_resource.clear()
        st.success("Entrenamiento regenerado y guardado.")
        st.rerun()


# ===========================================================================
# VISTA 1: INICIO & CONTEXTO (ETAPAS 01–04)
# ===========================================================================
if selected_page == pages[0]:
    render_hero(
        "GUÍA APLICATIVA · GRUPO 02 · AUTOCONTENIDO",
        "Estimación de Carga Térmica de Edificios con Red Neuronal Multicapa (MLP)",
        "Consultora de eficiencia energética · Dataset UCI Energy Efficiency (ID 242) · 16 etapas reproducibles",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Ejemplos simulados", f"{len(df)}", "12 formas base en Ecotect (2012)"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Variables de entrada", "8 (X1–X8)", "6 continuas + 2 nominales (X6, X8)", "navy"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Objetivo principal", "Y1 (kWh/m²)", f"Rango [{df['Y1'].min():.2f}, {df['Y1'].max():.2f}]", "amber"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Mejor MLP en Test", f"{best_mlp['test_rmse']:.3f} RMSE", f"{best_mlp['name']} · R²={best_mlp['test_r2']:.4f}", "accent"), unsafe_allow_html=True)

    st.markdown("#### 1. Situación problemática y tipo de aprendizaje (Etapas 01 y 02)")
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.markdown(
            """
            <div class="info-box">
            <b>Contexto de negocio:</b> Una consultora de eficiencia energética necesita estimar tempranamente
            la <b>carga de calefacción (Y1, en kWh/m²)</b> a partir de parámetros geométricos y de envolvente
            antes de ejecutar simulaciones térmicas costosas.<br><br>
            <b>Formulación supervisada:</b> Problema de <b>regresión supervisada tabular</b> (<i>f(X1..X8) → Y1</i>).
            Los datos provienen de simulaciones físicas en <i>Autodesk Ecotect</i> (Tsanas & Xifara, 2012), por lo que
            representan condiciones físicas idealizadas y no mediciones reales con ocupación humana o clima estocástico.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        leak = summary["leakage_experiment"]
        lin_leak = leak["Regresión Lineal + Y2 (fuga)"]
        mlp_leak = leak["MLP (32,16) + Y2 (fuga)"]
        st.markdown(
            f"""
            <div class="danger-box">
            <b>Regla crítica — Exclusión de Y2 (Etapa 04):</b><br>
            <code>Y2</code> (carga de refrigeración) tiene una correlación de Pearson de
            <b>r = {leak['pearson_y1_y2_train']:.4f}</b> con <code>Y1</code>. Incluirla como predictor produce
            <b>fuga de datos (data leakage)</b>: en Validación artificialmente bajaría el RMSE lineal a
            <b>{lin_leak['rmse']:.3f}</b> (R²={lin_leak['r2']:.4f}) y el del MLP a <b>{mlp_leak['rmse']:.3f}</b>,
            algo imposible en diseño real donde aún no conocemos Y2.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 2. Trazabilidad y estructura de variables (Etapas 03 y 04)")
    var_df = get_variable_summary_df(df)
    st.dataframe(var_df, width="stretch", hide_index=True)

    st.markdown("#### 3. Créditos, fuente oficial y licencia")
    st.markdown(
        f"- **Fuente oficial:** {metadata['uci']['name']} (UCI ID {metadata['uci']['uci_id']}) — "
        f"[Enlace UCI]({metadata['uci']['repository_url']})\n"
        f"- **Autores del dataset:** {', '.join(metadata['uci']['creators'])} ({metadata['uci']['year_of_dataset_creation']})\n"
        f"- **Archivos locales incluidos en el repositorio:** `{metadata['local_files']['xlsx']}` y `{metadata['local_files']['csv']}` (0 valores nulos, {metadata['duplicate_rows']} filas duplicadas).\n"
        f"- **Licencia:** {metadata['uci']['license']}."
    )


# ===========================================================================
# VISTA 2: EXPLORACIÓN DE DATOS - EDA (ETAPA 05)
# ===========================================================================
elif selected_page == pages[1]:
    render_hero(
        "ETAPA 05 · ANÁLISIS EXPLORATORIO DE DATOS (EDA)",
        "Distribución de Carga de Calefacción (Y1) y Relaciones Estructurales",
        "Visualización interactiva de compacidad, altura, orientación, área acristalada y correlaciones",
    )

    tab_dist, tab_corr, tab_rel, tab_data = st.tabs(
        ["📈 Distribución de Y1", "🔥 Mapa de Correlaciones", "🔎 Relaciones Clave", "🗂️ Datos Crudos"]
    )

    with tab_dist:
        st.plotly_chart(plotly_y1_distribution(df), width="stretch")
        st.markdown(
            """
            <div class="info-box">
            <b>Hallazgo bimodal:</b> La distribución de <code>Y1</code> presenta dos grupos claramente separados
            gobernados por la <b>altura total (X5)</b>: los edificios de 1 planta (3.5 m) concentran cargas entre
            6 y 20 kWh/m², mientras que los de 2 plantas (7.0 m) concentran cargas entre 23 y 43 kWh/m².
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_corr:
        st.plotly_chart(plotly_correlation_heatmap(df), width="stretch")
        st.markdown(
            "- **Mayor correlación positiva con Y1:** Altura `X5` (`r = +0.89`) y Área de techo `X4` (`r = -0.86`, inversa porque menor techo implica edificios más altos de 2 plantas en este diseño factorial).\n"
            "- **Relación lineal engañosa en X6/X8:** La correlación de Pearson con `X6` (`r ≈ 0.00`) y `X8` (`r ≈ 0.09`) asume orden numérico artificial; por eso se evalúa `OneHotEncoder`."
        )

    with tab_rel:
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(plotly_eda_scatter_compactness(df), width="stretch")
        with c_right:
            cat_choice = st.selectbox(
                "Variable de diseño para comparar Y1 por altura:",
                ["X6", "X7", "X8"],
                format_func=lambda c: f"{c} — {VARIABLE_INFO[c]['name']}",
            )
            st.plotly_chart(plotly_eda_categorical(df, cat_choice), width="stretch")

    with tab_data:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            h_filter = st.multiselect("Filtrar por Altura X5 (m)", [3.5, 7.0], default=[3.5, 7.0])
        with col_f2:
            x7_filter = st.multiselect("Filtrar por Área Acristalada X7", [0.0, 0.1, 0.25, 0.4], default=[0.0, 0.1, 0.25, 0.4])
        filtered = df[df["X5"].isin(h_filter) & df["X7"].isin(x7_filter)]
        st.caption(f"Mostrando {len(filtered)} de {len(df)} edificios del archivo local `data/energy_efficiency.csv`.")
        st.dataframe(filtered, width="stretch", height=380)


# ===========================================================================
# VISTA 3: ARQUITECTURA MLP & PREPROCESAMIENTO (ETAPAS 06–10)
# ===========================================================================
elif selected_page == pages[2]:
    render_hero(
        "ETAPAS 06–10 · DIVISIÓN, PREPROCESAMIENTO Y ARQUITECTURA MLP",
        "Diseño Feedforward, Codificación Categórica y Conteo de Parámetros",
        "División estratificada sin fuga, estandarización + OneHotEncoder y anatomía exacta de pesos y sesgos",
    )

    sp = summary["split_summary"]
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(kpi_card("Train (70%)", f"{sp['train_rows']} filas", f"Semilla SEED = {sp['seed']}"), unsafe_allow_html=True)
    with s2:
        st.markdown(kpi_card("Validation (15%)", f"{sp['val_rows']} filas", "Selección de hiperparámetros", "navy"), unsafe_allow_html=True)
    with s3:
        st.markdown(kpi_card("Test (15%)", f"{sp['test_rows']} filas", "Aislado para evaluación final", "accent"), unsafe_allow_html=True)
    with s4:
        st.markdown(kpi_card("Formas base", f"{sp['unique_base_geometries']['total']} únicas", "Representadas en los 3 splits", "amber"), unsafe_allow_html=True)

    st.markdown("#### 1. Diagrama y conteo exacto de parámetros por arquitectura (Etapas 09 y 10)")
    mlp_archs = summary["mlp_architectures"]
    arch_id = st.selectbox(
        "Selecciona una arquitectura MLP entrenada para inspeccionar sus tensores y parámetros:",
        list(mlp_archs.keys()),
        format_func=lambda k: f"Exp. {mlp_archs[k]['code']} · {mlp_archs[k]['name']} ({mlp_archs[k]['total_params']} parámetros)",
    )
    chosen_arch = mlp_archs[arch_id]

    col_diag, col_tbl = st.columns([1.2, 1])
    with col_diag:
        layer_dims = [16] + chosen_arch["hidden_layers"] + [1]
        st.plotly_chart(plotly_mlp_architecture_diagram(layer_dims), width="stretch")
    with col_tbl:
        st.markdown(
            f"""
            **Operación por capa:** $z = Wx + b$, activación oculta $\\text{{ReLU}}(z) = \\max(0, z)$, capa final **lineal**.<br>
            **Aprendizaje:** Pérdida MSE (`squared_error`), optimizador **Adam** (`lr={chosen_arch['learning_rate_init']}`),
            regularización L2 `alpha={chosen_arch['alpha']}`, `early_stopping={chosen_arch['early_stopping']}`.
            """
        )
        layers_df = pd.DataFrame(chosen_arch["layers"])[
            ["layer", "input_dim", "output_dim", "activation", "weights", "biases", "params", "formula"]
        ]
        layers_df.columns = ["Capa", "Entrada", "Salida", "Activación", "Pesos (W)", "Sesgos (b)", "Total", "Cálculo"]
        st.dataframe(layers_df, width="stretch", hide_index=True)
        st.success(
            f"**Total entrenable:** {chosen_arch['total_weights']} pesos + {chosen_arch['total_biases']} sesgos = "
            f"**{chosen_arch['total_params']} parámetros** (convergió en {chosen_arch['n_iter']} épocas)."
        )

    st.markdown("#### 2. Evaluación de OneHotEncoder en variables nominales X6 y X8 (Etapa 07)")
    ohe_df = pd.DataFrame(summary["onehot_ablation"])
    col_o1, col_o2 = st.columns([1.1, 1])
    with col_o1:
        fig_ohe = px.bar(
            ohe_df,
            x="modelo",
            y="val_rmse",
            color="codificacion",
            barmode="group",
            color_discrete_map={
                "Sin OneHot (numéricas crudas)": PALETTE["accent"],
                "Con OneHotEncoder (X6, X8)": PALETTE["primary"],
            },
            title="RMSE en Validación: Con vs Sin OneHotEncoder (X6, X8)",
            labels={"val_rmse": "RMSE Validación (kWh/m²)", "modelo": "Modelo"},
        )
        fig_ohe.update_layout(template="plotly_white", height=370)
        st.plotly_chart(fig_ohe, width="stretch")
    with col_o2:
        st.dataframe(ohe_df, width="stretch", hide_index=True)
        st.markdown(
            """
            <div class="info-box">
            <b>Interpretación:</b> Codificar <code>X6</code> (4 categorías) y <code>X8</code> (6 categorías)
            expande la entrada de 8 a <b>16 dimensiones</b> sin imponer una falsa jerarquía numérica
            (por ejemplo, asumir que <i>Oeste=5</i> es "mayor" que <i>Norte=2</i>). En el MLP esto evita
            sesgos ordinales artificiales y mejora la representación de orientaciones.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# VISTA 4: EXPERIMENTOS, CURVAS Y SELECCIÓN (ETAPAS 11–14)
# ===========================================================================
elif selected_page == pages[3]:
    render_hero(
        "ETAPAS 11–14 · EXPERIMENTOS A–D, CURVAS Y VALIDACIÓN CRUZADA",
        "Comparación Sistemática de Arquitecturas MLP y Baselines Tabulares",
        "Selección rigurosa por RMSE de Validación y CV 5-Fold encapsulada en Pipeline (sin tocar Test)",
    )

    st.markdown(
        f"""
        <div class="info-box">
        <b>Decisión de Selección (Etapa 14):</b> {summary['selection_reason']}<br>
        Nota de protocolo: El conjunto <b>Test</b> no participó en ningún ajuste, búsqueda de hiperparámetros ni selección.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_df = pd.DataFrame(summary["metrics_table"])
    display_cols = [
        "code",
        "name",
        "family",
        "n_params",
        "n_iter",
        "train_time_s",
        "train_rmse",
        "val_mae",
        "val_rmse",
        "val_r2",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
    ]
    tbl_show = metrics_df[display_cols].copy()
    tbl_show.columns = [
        "Exp.",
        "Modelo",
        "Familia",
        "Parámetros",
        "Épocas",
        "Tiempo (s)",
        "Train RMSE",
        "Val MAE",
        "Val RMSE",
        "Val R²",
        "CV5 RMSE",
        "CV5 Std",
        "CV5 R²",
    ]
    st.dataframe(tbl_show, width="stretch", hide_index=True)

    st.plotly_chart(plotly_learning_curves(results_for_plots), width="stretch")
    st.plotly_chart(plotly_model_comparison(results_for_plots), width="stretch")

    with st.expander("Análisis detallado de sobreajuste y efecto de regularización (Exp A vs B vs C1–C3)"):
        lc_rows = []
        for exp_id, lc in summary["learning_curves"].items():
            lc_rows.append(
                {
                    "Modelo": summary["mlp_architectures"][exp_id]["name"],
                    "Mejor época (Val)": lc["best_epoch_val"],
                    "Mejor Val RMSE": round(lc["best_val_rmse"], 4),
                    f"Train RMSE (época {lc['epochs'][-1]})": round(lc["final_train_rmse"], 4),
                    f"Val RMSE (época {lc['epochs'][-1]})": round(lc["final_val_rmse"], 4),
                    "Brecha Val - Train": round(lc["overfit_gap_end"], 4),
                }
            )
        st.dataframe(pd.DataFrame(lc_rows), width="stretch", hide_index=True)


# ===========================================================================
# VISTA 5: EVALUACIÓN FINAL & RESIDUOS (ETAPA 15)
# ===========================================================================
elif selected_page == pages[4]:
    render_hero(
        "ETAPA 15 · EVALUACIÓN FINAL EN TEST Y ANÁLISIS DE RESIDUOS",
        "Desempeño Definitivo en Datos No Vistos (n = 116) y Errores por Subgrupos",
        "Confrontación justa entre el MLP seleccionado, Regresión Lineal y Random Forest",
    )

    metrics_df = pd.DataFrame(summary["metrics_table"])
    lin_row = metrics_df[metrics_df["exp_id"] == "exp_d1_linear"].iloc[0]
    rf_row = metrics_df[metrics_df["exp_id"] == "exp_d2_rf"].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            kpi_card(
                f"Test RMSE · {best_mlp['code']}",
                f"{best_mlp['test_rmse']:.4f}",
                f"MAE={best_mlp['test_mae']:.4f} · R²={best_mlp['test_r2']:.4f}",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            kpi_card(
                "Test RMSE · Reg. Lineal",
                f"{lin_row['test_rmse']:.4f}",
                f"MAE={lin_row['test_mae']:.4f} · R²={lin_row['test_r2']:.4f}",
                "accent",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            kpi_card(
                "Test RMSE · Random Forest",
                f"{rf_row['test_rmse']:.4f}",
                f"MAE={rf_row['test_mae']:.4f} · R²={rf_row['test_r2']:.4f}",
                "navy",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            kpi_card(
                "Brecha Test - Val (MLP)",
                f"{best_mlp['diff_test_val_rmse']:+.4f}",
                f"Diferencia MAE: {best_mlp['diff_test_val_mae']:+.4f} kWh/m²",
                "amber",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("#### 1. Tabla completa de evaluación final en Test (n = 116)")
    test_cols = [
        "code",
        "name",
        "val_rmse",
        "test_mae",
        "test_rmse",
        "test_r2",
        "diff_test_val_mae",
        "diff_test_val_rmse",
    ]
    t_df = metrics_df[test_cols].copy()
    t_df.columns = ["Exp.", "Modelo", "Val RMSE", "Test MAE", "Test RMSE", "Test R²", "Δ MAE (Test-Val)", "Δ RMSE (Test-Val)"]
    st.dataframe(t_df, width="stretch", hide_index=True)

    st.markdown("#### 2. Gráficos de dispersión y análisis de residuos")
    model_options = {r["exp_id"]: f"{r['code']} · {r['name']}" for r in summary["metrics_table"] if r["exp_id"] != "exp_d0_dummy"}
    chosen_test_id = st.selectbox(
        "Inspeccionar dispersión y residuos en Test para:",
        list(model_options.keys()),
        index=list(model_options.keys()).index(best_mlp["exp_id"]),
        format_func=lambda k: model_options[k],
    )
    st.plotly_chart(
        plotly_real_vs_pred_and_residuals(results_for_plots, chosen_test_id, model_options[chosen_test_id]),
        width="stretch",
    )

    col_hist, col_err = st.columns([1, 1.2])
    with col_hist:
        res_series = pred_test_df[f"res_{chosen_test_id}"]
        fig_res_hist = px.histogram(
            res_series,
            nbins=22,
            title=f"Distribución de residuos ({model_options[chosen_test_id]})",
            labels={"value": "Residuo Y1_real - Y1_pred (kWh/m²)", "count": "Frecuencia"},
            color_discrete_sequence=[PALETTE["primary"]],
        )
        fig_res_hist.add_vline(x=0, line_dash="dash", line_color=PALETTE["accent"])
        fig_res_hist.update_layout(template="plotly_white", height=360, showlegend=False)
        st.plotly_chart(fig_res_hist, width="stretch")

    with col_err:
        st.markdown("**Top 8 edificios con mayor error absoluto en Test (diagnóstico)**")
        diag_df = pred_test_df[FEATURE_COLS + [TARGET_COL]].copy()
        diag_df["Y1_Pred"] = pred_test_df[f"pred_{chosen_test_id}"]
        diag_df["Error_Abs"] = np.abs(pred_test_df[f"res_{chosen_test_id}"]).round(4)
        diag_df = diag_df.sort_values("Error_Abs", ascending=False).head(8)
        st.dataframe(diag_df, width="stretch", hide_index=True)

    st.markdown("#### 3. Desempeño por subgrupos en Test (Altura X5, Orientación X6, Acristalamiento X7)")
    sg_data = summary["subgroups_test"]["best_mlp"]["groups"]
    sg_tabs = st.tabs(list(sg_data.keys()))
    for tab, g_name in zip(sg_tabs, sg_data.keys()):
        with tab:
            c_sg1, c_sg2 = st.columns([1.1, 1])
            df_mlp_g = pd.DataFrame(summary["subgroups_test"]["best_mlp"]["groups"][g_name])
            df_lin_g = pd.DataFrame(summary["subgroups_test"]["exp_d1_linear"]["groups"][g_name])
            df_rf_g = pd.DataFrame(summary["subgroups_test"]["exp_d2_rf"]["groups"][g_name])

            comp_g = pd.concat(
                [
                    df_lin_g[["subgrupo", "mae"]].assign(Modelo="Regresión Lineal"),
                    df_mlp_g[["subgrupo", "mae"]].assign(Modelo=best_mlp["name"]),
                    df_rf_g[["subgrupo", "mae"]].assign(Modelo="Random Forest"),
                ],
                ignore_index=True,
            )
            with c_sg1:
                fig_sg = px.bar(
                    comp_g,
                    x="subgrupo",
                    y="mae",
                    color="Modelo",
                    barmode="group",
                    color_discrete_map={
                        "Regresión Lineal": "#94A3B8",
                        best_mlp["name"]: PALETTE["primary"],
                        "Random Forest": PALETTE["navy"],
                    },
                    title=f"MAE en Test por {g_name}",
                    labels={"subgrupo": g_name, "mae": "MAE (kWh/m²)"},
                )
                fig_sg.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_sg, width="stretch")
            with c_sg2:
                st.caption(f"Métricas del MLP seleccionado ({best_mlp['name']}) por {g_name}:")
                st.dataframe(df_mlp_g, width="stretch", hide_index=True)


# ===========================================================================
# VISTA 6: SIMULADOR & PREDICCIÓN EN VIVO (ETAPA 16)
# ===========================================================================
elif selected_page == pages[5]:
    render_hero(
        "ETAPA 16 · DEMOSTRACIÓN INTERACTIVA E INFERENCIA EN VIVO",
        "Simulador de Carga de Calefacción (Y1) con Control de Dominio",
        f"Inferencia instantánea desde pipelines serializados · Modelo principal: {best_mlp['name']}",
    )

    presets = {
        "Edificio compacto 2 plantas (caso real del dataset)": {
            "X1": 0.98, "X2": 514.5, "X3": 294.0, "X4": 110.25,
            "X5": 7.0, "X6": 2, "X7": 0.25, "X8": 1,
        },
        "Edificio alargado 1 planta (caso real del dataset)": {
            "X1": 0.62, "X2": 808.5, "X3": 367.5, "X4": 220.5,
            "X5": 3.5, "X6": 4, "X7": 0.40, "X8": 3,
        },
        "Prueba de extrapolación (fuera del rango entrenado)": {
            "X1": 0.50, "X2": 920.0, "X3": 450.0, "X4": 235.0,
            "X5": 10.5, "X6": 3, "X7": 0.60, "X8": 2,
        },
    }

    preset_choice = st.selectbox("Cargar perfil predefinido (o ajustar manualmente abajo):", list(presets.keys()))
    p_vals = presets[preset_choice]

    col_form, col_pred = st.columns([1.15, 1])

    with col_form:
        st.markdown("#### Características de diseño del edificio")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            x1 = st.number_input("X1 · Compacidad relativa [0.62–0.98]", min_value=0.20, max_value=1.20, value=float(p_vals["X1"]), step=0.01)
            x2 = st.number_input("X2 · Área superficial (m²) [514.5–808.5]", min_value=200.0, max_value=1500.0, value=float(p_vals["X2"]), step=2.5)
            x3 = st.number_input("X3 · Área de paredes (m²) [245.0–416.5]", min_value=100.0, max_value=800.0, value=float(p_vals["X3"]), step=2.5)
            x4 = st.number_input("X4 · Área de techo (m²) [110.25–220.5]", min_value=50.0, max_value=500.0, value=float(p_vals["X4"]), step=2.25)
        with f_col2:
            x5 = st.number_input("X5 · Altura total (m) [3.5 o 7.0]", min_value=2.0, max_value=20.0, value=float(p_vals["X5"]), step=0.5)
            x6 = st.selectbox("X6 · Orientación (nominal)", [2, 3, 4, 5], index=[2, 3, 4, 5].index(int(p_vals["X6"])), format_func=lambda v: X6_LABELS[v])
            x7 = st.slider("X7 · Área acristalada (proporción) [0.00–0.40]", min_value=0.0, max_value=0.80, value=float(p_vals["X7"]), step=0.05)
            x8 = st.selectbox("X8 · Distribución de acristalamiento (nominal)", [0, 1, 2, 3, 4, 5], index=int(p_vals["X8"]), format_func=lambda v: X8_LABELS[v])

        avail_models = {r["exp_id"]: f"{r['code']} · {r['name']}" for r in summary["metrics_table"] if r["exp_id"] != "exp_d0_dummy"}
        selected_sim_model = st.selectbox(
            "Modelo para inferencia principal:",
            list(avail_models.keys()),
            index=list(avail_models.keys()).index(best_mlp["exp_id"]),
            format_func=lambda k: avail_models[k],
        )

    user_inputs = {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "X6": x6, "X7": x7, "X8": x8}
    input_df = pd.DataFrame([user_inputs])
    domain_check = check_domain_and_geometry(user_inputs)

    pred_selected = float(pipelines[selected_sim_model].predict(input_df)[0])
    pred_mlp_best = float(pipelines[best_mlp["exp_id"]].predict(input_df)[0])
    pred_rf = float(pipelines["exp_d2_rf"].predict(input_df)[0])
    pred_lin = float(pipelines["exp_d1_linear"].predict(input_df)[0])

    selected_row = [r for r in summary["metrics_table"] if r["exp_id"] == selected_sim_model][0]
    rmse_ref = float(selected_row["test_rmse"])

    with col_pred:
        st.markdown("#### Estimación de Carga Térmica (Y1)")
        st.markdown(
            kpi_card(
                f"Predicción · {avail_models[selected_sim_model]}",
                f"{pred_selected:.2f} kWh/m²",
                f"Banda orientativa (±1 RMSE Test): [{max(0.0, pred_selected - rmse_ref):.2f}, {pred_selected + rmse_ref:.2f}] kWh/m²",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("")
        c_cmp1, c_cmp2, c_cmp3 = st.columns(3)
        c_cmp1.metric("Mejor MLP", f"{pred_mlp_best:.2f} kWh/m²")
        c_cmp2.metric("Random Forest", f"{pred_rf:.2f} kWh/m²")
        c_cmp3.metric("Reg. Lineal", f"{pred_lin:.2f} kWh/m²")

        if domain_check["has_warnings"]:
            warn_html = "<br>".join([f"• {w}" for w in domain_check["warnings"]])
            st.markdown(
                f"""
                <div class="warning-box">
                <b>⚠️ Advertencia de Extrapolación / Consistencia Geométrica:</b><br>{warn_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="info-box">
                <b>✅ Entrada dentro del dominio de entrenamiento:</b> Todas las variables están dentro
                de los rangos observados y cumplen la relación geométrica <code>X2 ≈ X3 + 2·X4</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            **Interpretación y limitaciones:**
            - **Datos admitidos:** 8 atributos arquitectónicos (`X1`–`X8`), sin valores nulos; `X6` y `X8` se codifican vía `OneHotEncoder`.
            - **Cómo interpretar la salida:** Demanda anual estimada de calefacción por unidad de área útil (`kWh/m²`).
            - **Limitación clave:** Modelo entrenado sobre 768 configuraciones simuladas en *Ecotect* (climas/materiales fijos y solo alturas de 3.5 m y 7.0 m). No extrapola con garantía a climas distintos ni rascacielos.
            """
        )

    st.markdown("#### Sensibilidad local: efecto de variar Área Acristalada (X7) manteniendo fijo el edificio")
    x7_grid = np.linspace(0.0, 0.4, 21)
    sens_df = pd.DataFrame([user_inputs] * len(x7_grid))
    sens_df["X7"] = x7_grid
    sens_df["MLP"] = pipelines[best_mlp["exp_id"]].predict(sens_df[FEATURE_COLS])
    sens_df["Random Forest"] = pipelines["exp_d2_rf"].predict(sens_df[FEATURE_COLS])
    sens_df["Reg. Lineal"] = pipelines["exp_d1_linear"].predict(sens_df[FEATURE_COLS])
    sens_melt = sens_df.melt(id_vars=["X7"], value_vars=["MLP", "Random Forest", "Reg. Lineal"], var_name="Modelo", value_name="Y1_pred")
    fig_sens = px.line(
        sens_melt,
        x="X7",
        y="Y1_pred",
        color="Modelo",
        markers=True,
        title="Curva de respuesta estimada al variar X7 (0% a 40% de acristalamiento)",
        labels={"X7": "Área acristalada X7", "Y1_pred": "Carga estimada Y1 (kWh/m²)"},
    )
    fig_sens.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig_sens, width="stretch")


# ===========================================================================
# VISTA 7: SUSTENTACIÓN & PREGUNTAS DEL DOCENTE
# ===========================================================================
elif selected_page == pages[6]:
    render_hero(
        "PREPARACIÓN PARA LA SUSTENTACIÓN · 16 ETAPAS & 8 PREGUNTAS",
        "Verificación Metodológica y Respuestas Respaldadas por Evidencia Real",
        "Checklist completo de la guía académica y guía rápida de defensa oral",
    )

    st.markdown("#### Checklist de las 16 etapas obligatorias")
    stages = [
        ("01", "Definición del problema", "Predicción energética temprana en consultoría sobre datos simulados en Ecotect."),
        ("02", "Tipo de aprendizaje", "Regresión supervisada tabular f(X1..X8) → Y1 en kWh/m²."),
        ("03", "Obtención y trazabilidad", "Archivo oficial data/ENB2012_data.xlsx y data/energy_efficiency.csv locales (768×10, 0 nulos)."),
        ("04", "Entradas, salidas y objetivo", "X = X1..X8, y = Y1; exclusión explícita de Y2 (r = 0.976 causaría data leakage)."),
        ("05", "Análisis exploratorio", "Distribución bimodal por altura X5, correlaciones y gráficos frente a X1, X5, X6, X7."),
        ("06", "Separación experimental", f"Split estratificado Train={summary['split_summary']['train_rows']}, Val={summary['split_summary']['val_rows']}, Test={summary['split_summary']['test_rows']} (SEED={SEED})."),
        ("07", "Preprocesamiento", "Imputación + StandardScaler para continuas y comparación explícita de OneHotEncoder en X6 y X8."),
        ("08", "Modelo de referencia", "Baselines DummyRegressor, Regresión Lineal y Random Forest (200 árboles)."),
        ("09", "Diseño de arquitectura", "8 entradas → 16 codificadas, capas densas con ReLU y salida escalar lineal."),
        ("10", "Configuración aprendizaje", "Pérdida MSE, optimizador Adam (lr=0.005) y conteo exacto de parámetros por capa."),
        ("11", "Plan de hiperparámetros", "Comparación de Exp A (16), Exp B (32,16) y Exp C (64,32) con L2 y Early Stopping."),
        ("12", "Ejecución y seguimiento", "Curvas oficiales de pérdida MSE y seguimiento Train vs Validación por época."),
        ("13", "Validación", "Validación cruzada 5-Fold sobre Train+Val con todo el preprocesamiento encapsulado en Pipeline."),
        ("14", "Selección de configuración", summary["selection_reason"]),
        ("15", "Evaluación final", "MAE, RMSE, R², brecha Test–Val, residuos y métricas por subgrupos en Test (n=116)."),
        ("16", "Interpretación y demostración", "Simulador en vivo con alertas de extrapolación fuera de rango e incoherencia geométrica."),
    ]
    st.dataframe(pd.DataFrame(stages, columns=["Etapa", "Título", "Evidencia implementada en el proyecto"]), width="stretch", hide_index=True)

    st.markdown("#### Respuestas a las 8 preguntas de sustentación")
    qa_items = [
        (
            "1. ¿Por qué un MLP necesita funciones de activación no lineales?",
            "Sin activaciones no lineales (como ReLU), apilar varias capas densas equivale matemáticamente a una única multiplicación matricial lineal W_eq·x + b_eq. Por lo tanto, el MLP nunca podría superar a una Regresión Lineal ni capturar interacciones no lineales entre compacidad, altura y acristalamiento.",
        ),
        (
            "2. ¿Qué aprende una capa oculta que no representa una regresión lineal simple?",
            "Cada neurona oculta aprende una combinación de características activada por umbral (ReLU) que actúa como una frontera o rasgo latente por tramos (por ejemplo, combinar compacidad baja con altura de 7 m y alto acristalamiento). La capa de salida combina estas representaciones no lineales.",
        ),
        (
            "3. ¿Por qué X6 y X8 requieren revisar su interpretación?",
            "Aunque en el archivo ENB2012_data.xlsx vienen como números enteros (X6: 2–5; X8: 0–5), no representan magnitudes ordenadas sino códigos nominales de orientación cardinal (Norte, Este, Sur, Oeste) y distribución de ventanas. Tratarlas como números continuos impone un orden falso; usar OneHotEncoder resuelve esa ambigüedad.",
        ),
        (
            "4. ¿Qué sucede si incluimos Y2 como predictor de Y1?",
            f"Ocurre fuga de datos (data leakage). En nuestro experimento demostrativo, Y2 (carga de refrigeración) tiene correlación r = {summary['leakage_experiment']['pearson_y1_y2_train']:.4f} con Y1; al incluirla, incluso una regresión lineal parece tener un error artificialmente bajo. Es inválido porque en la práctica ambas cargas se desconocen al diseñar un edificio nuevo.",
        ),
        (
            "5. ¿Qué es early stopping y cómo se utiliza sin tocar el test?",
            "Early stopping detiene el entrenamiento cuando el error en un conjunto de validación interno deja de mejorar tras cierto número de épocas (n_iter_no_change=25 en Exp. C3). Así se evita memorizar ruido (sobreajuste). En scikit-learn toma validation_fraction=0.15 desde dentro del propio conjunto Train, manteniendo Validation y Test completamente intactos.",
        ),
        (
            "6. ¿Qué diferencias hay entre MAE y RMSE?",
            "MAE (Error Absoluto Medio) mide el error promedio en kWh/m² tratando todas las desviaciones de forma lineal, lo que facilita su interpretación práctica. RMSE (Raíz del Error Cuadrático Medio) eleva los errores al cuadrado antes de promediar, penalizando con mucha mayor fuerza los errores grandes o atípicos.",
        ),
        (
            "7. ¿Por qué Random Forest podría superar al MLP en un conjunto pequeño?",
            "Con solo 768 observaciones (537 en entrenamiento) generadas mediante un diseño factorial discreto (12 geometrías, 4 niveles de X7, 2 alturas), los árboles de decisión particionan directamente esos escalones discretos con muy baja varianza sin necesitar gradientes suaves ni ajustar cientos o miles de pesos.",
        ),
        (
            "8. ¿Qué conclusiones respaldan los residuos?",
            "Al comparar residuos (y_real - y_pred) vs predicción en Test, la Regresión Lineal exhibe curvatura y agrupamientos sistemáticos (subajuste no lineal), mientras que el MLP seleccionado y Random Forest muestran residuos centrados cerca de cero y con menor dispersión, aunque con errores puntuales concentrados en configuraciones extremas de altura o acristalamiento.",
        ),
    ]
    for q_title, q_ans in qa_items:
        with st.expander(q_title):
            st.write(q_ans)
