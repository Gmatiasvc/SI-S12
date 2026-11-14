"""Generadores de gráficos estáticos (Matplotlib/Seaborn para LaTeX/diapositivas) e interactivos (Plotly para Streamlit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from src.data_loader import ALL_COLS, FEATURE_COLS, TARGET_COL, X6_LABELS, X8_LABELS

# Paleta corporativa consistente con la guía académica (verde bosque + teal + acentos ámbar/coral)
PALETTE = {
    "primary": "#0F5E4D",
    "secondary": "#1D8A73",
    "accent": "#E07A5F",
    "amber": "#D99B26",
    "navy": "#1E3A5F",
    "slate": "#475569",
    "light_bg": "#F8FAFC",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFDFC",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
    }
)


# ---------------------------------------------------------------------------
# 1. FIGURAS ESTÁTICAS DE ALTA RESOLUCIÓN PARA LATEX, NOTEBOOK Y DIAPOSITIVAS
# ---------------------------------------------------------------------------

def save_static_figures(
    df: pd.DataFrame,
    results: Dict[str, Any],
    figures_dir: Path,
) -> List[Path]:
    """Genera y guarda las 7 figuras estáticas oficiales en alta resolución."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    saved.append(_plot_static_y1_distribution(df, figures_dir / "fig01_distribucion_y1.png"))
    saved.append(_plot_static_correlations(df, figures_dir / "fig02_correlaciones.png"))
    saved.append(_plot_static_eda_relations(df, figures_dir / "fig03_eda_relaciones.png"))
    saved.append(_plot_static_learning_curves(results, figures_dir / "fig04_curvas_perdida_mlp.png"))
    saved.append(_plot_static_model_comparison(results, figures_dir / "fig05_comparacion_modelos.png"))
    saved.append(_plot_static_real_pred_residuals(results, figures_dir / "fig06_real_vs_pred_y_residuos.png"))
    saved.append(_plot_static_subgroup_errors(results, figures_dir / "fig07_errores_subgrupos.png"))
    return saved


def _plot_static_y1_distribution(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    sns.histplot(df[TARGET_COL], bins=24, kde=True, color=PALETTE["primary"], ax=axes[0])
    axes[0].axvline(df[TARGET_COL].mean(), color=PALETTE["accent"], linestyle="--", lw=1.8, label=f"Media: {df[TARGET_COL].mean():.2f}")
    axes[0].axvline(df[TARGET_COL].median(), color=PALETTE["navy"], linestyle=":", lw=1.8, label=f"Mediana: {df[TARGET_COL].median():.2f}")
    axes[0].set_title("Distribución bimodal de Y1 (Carga de Calefacción)")
    axes[0].set_xlabel("Y1 - Carga de calefacción (kWh/m²)")
    axes[0].set_ylabel("Frecuencia (edificios)")
    axes[0].legend()

    df_plot = df.copy()
    df_plot["Altura"] = df_plot["X5"].map({3.5: "3.5 m (1 planta)", 7.0: "7.0 m (2 plantas)"})
    sns.boxplot(
        data=df_plot,
        x="Altura",
        y=TARGET_COL,
        hue="Altura",
        palette=[PALETTE["secondary"], PALETTE["navy"]],
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Separación de Y1 por altura del edificio (X5)")
    axes[1].set_xlabel("Altura total X5")
    axes[1].set_ylabel("Y1 (kWh/m²)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_static_correlations(df: pd.DataFrame, out_path: Path) -> Path:
    corr = df[ALL_COLS].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Correlación de Pearson (r)"},
        ax=ax,
    )
    ax.set_title("Mapa de correlaciones (X1–X8, Y1 e Y2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_static_eda_relations(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    df_plot = df.copy()
    df_plot["Altura"] = df_plot["X5"].map({3.5: "3.5 m", 7.0: "7.0 m"})
    df_plot["Orientación"] = df_plot["X6"].map(X6_LABELS)

    # 1. Y1 vs Compacidad X1
    sns.scatterplot(
        data=df_plot,
        x="X1",
        y=TARGET_COL,
        hue="Altura",
        palette={"3.5 m": PALETTE["secondary"], "7.0 m": PALETTE["navy"]},
        alpha=0.75,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Y1 vs Compacidad relativa (X1)")
    axes[0, 0].set_xlabel("X1 (Compacidad relativa)")
    axes[0, 0].set_ylabel("Y1 (kWh/m²)")

    # 2. Y1 vs Altura X5
    sns.violinplot(
        data=df_plot,
        x="Altura",
        y=TARGET_COL,
        hue="Altura",
        palette={"3.5 m": PALETTE["secondary"], "7.0 m": PALETTE["navy"]},
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Y1 según Altura total (X5)")
    axes[0, 1].set_xlabel("Altura X5")
    axes[0, 1].set_ylabel("Y1 (kWh/m²)")

    # 3. Y1 vs Orientación X6
    sns.boxplot(
        data=df_plot,
        x="Orientación",
        y=TARGET_COL,
        hue="Altura",
        palette={"3.5 m": PALETTE["secondary"], "7.0 m": PALETTE["navy"]},
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Y1 según Orientación (X6) y Altura")
    axes[1, 0].set_xlabel("Orientación (código nominal X6)")
    axes[1, 0].set_ylabel("Y1 (kWh/m²)")

    # 4. Y1 vs Área acristalada X7
    df_plot["Acristalamiento"] = (df_plot["X7"] * 100).round(0).astype(int).astype(str) + "%"
    sns.boxplot(
        data=df_plot,
        x="Acristalamiento",
        y=TARGET_COL,
        order=["0%", "10%", "25%", "40%"],
        hue="Altura",
        palette={"3.5 m": PALETTE["secondary"], "7.0 m": PALETTE["navy"]},
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Y1 según Área acristalada (X7) y Altura")
    axes[1, 1].set_xlabel("Área acristalada X7 (% del piso)")
    axes[1, 1].set_ylabel("Y1 (kWh/m²)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_static_learning_curves(results: Dict[str, Any], out_path: Path) -> Path:
    curves = results["learning_curves"]
    loss_official = results["loss_curves_official"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

    colors = {
        "exp_a_mlp_16": "#2A9D8F",
        "exp_b_mlp_32_16": "#0F5E4D",
        "exp_c1_mlp_64_32_noreg": "#E76F51",
        "exp_c2_mlp_64_32_l2": "#1E3A5F",
        "exp_c3_mlp_64_32_es": "#D99B26",
    }
    labels = {
        "exp_a_mlp_16": "Exp A: MLP (16)",
        "exp_b_mlp_32_16": "Exp B: MLP (32,16)",
        "exp_c1_mlp_64_32_noreg": "Exp C1: MLP (64,32) alpha=1e-6",
        "exp_c2_mlp_64_32_l2": "Exp C2: MLP (64,32) L2=0.05",
        "exp_c3_mlp_64_32_es": "Exp C3: MLP (64,32) L2 + EarlyStop",
    }

    for exp_id, loss_vals in loss_official.items():
        axes[0].plot(
            range(1, len(loss_vals) + 1),
            loss_vals,
            label=f"{labels.get(exp_id, exp_id)} ({len(loss_vals)} ep)",
            color=colors.get(exp_id, "#333333"),
            lw=1.8,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Curva oficial de pérdida de entrenamiento (MLPRegressor)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida MSE/2 + L2 (escala log)")
    axes[0].legend(fontsize=8.2)

    for exp_id, lc in curves.items():
        axes[1].plot(
            lc["epochs"],
            lc["val_rmse"],
            label=f"{labels.get(exp_id, exp_id)} (Val)",
            color=colors.get(exp_id, "#333333"),
            lw=1.9,
        )
        axes[1].plot(
            lc["epochs"],
            lc["train_rmse"],
            linestyle="--",
            alpha=0.55,
            color=colors.get(exp_id, "#333333"),
            lw=1.2,
        )
    axes[1].set_ylim(0, 6.0)
    axes[1].set_title("RMSE por época: Validación (sólida) vs Train (punteada)")
    axes[1].set_xlabel("Época (seguimiento controlado)")
    axes[1].set_ylabel("RMSE (kWh/m²)")
    axes[1].legend(fontsize=8.0, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_static_model_comparison(results: Dict[str, Any], out_path: Path) -> Path:
    df_m = pd.DataFrame(results["metrics_table"])
    df_m = df_m[df_m["exp_id"] != "exp_d0_dummy"].copy()
    df_m["short_label"] = df_m["code"] + ": " + df_m["name"].str.replace("Regresión Lineal (OneHot)", "Reg. Lineal", regex=False).str.replace("Random Forest (200 árboles)", "Random Forest", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    x = np.arange(len(df_m))
    width = 0.27
    axes[0].bar(x - width, df_m["val_rmse"], width, label="Validación", color=PALETTE["secondary"])
    axes[0].bar(x, df_m["cv_rmse_mean"], width, label="CV 5-Fold (Train+Val)", color=PALETTE["navy"])
    axes[0].bar(x + width, df_m["test_rmse"], width, label="Test final", color=PALETTE["accent"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_m["short_label"], rotation=25, ha="right", fontsize=8.5)
    axes[0].set_ylabel("RMSE (kWh/m²) — menor es mejor")
    axes[0].set_title("Comparación de arquitecturas MLP y baselines")
    axes[0].legend()

    df_ohe = pd.DataFrame(results["onehot_ablation"])
    sns.barplot(
        data=df_ohe,
        x="modelo",
        y="val_rmse",
        hue="codificacion",
        palette=[PALETTE["accent"], PALETTE["primary"]],
        ax=axes[1],
    )
    axes[1].set_title("Impacto de codificar X6 y X8 con OneHotEncoder (Validación)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE en Validación (kWh/m²)")
    axes[1].legend(title="Preprocesamiento", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_static_real_pred_residuals(results: Dict[str, Any], out_path: Path) -> Path:
    pred_df = results["predictions_test_df"]
    best_mlp_id = results["best_mlp"]["exp_id"]
    best_mlp_name = results["best_mlp"]["name"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))

    y_true = pred_df[TARGET_COL].to_numpy()
    y_lin = pred_df["pred_exp_d1_linear"].to_numpy()
    y_mlp = pred_df[f"pred_{best_mlp_id}"].to_numpy()
    y_rf = pred_df["pred_exp_d2_rf"].to_numpy()

    axes[0].scatter(y_true, y_lin, alpha=0.55, s=28, color="#94A3B8", label="Regresión Lineal")
    axes[0].scatter(y_true, y_rf, alpha=0.65, s=30, color=PALETTE["navy"], marker="^", label="Random Forest")
    axes[0].scatter(y_true, y_mlp, alpha=0.80, s=34, color=PALETTE["primary"], label=best_mlp_name)
    lims = [min(y_true.min(), y_mlp.min()) - 1, max(y_true.max(), y_mlp.max()) + 1]
    axes[0].plot(lims, lims, "r--", lw=1.6, label="Predicción ideal (y = x)")
    axes[0].set_title("Conjunto Test: y_real vs y_predicho")
    axes[0].set_xlabel("Y1 Real (kWh/m²)")
    axes[0].set_ylabel("Y1 Predicho (kWh/m²)")
    axes[0].legend(fontsize=8.5)

    res_lin = y_true - y_lin
    res_mlp = y_true - y_mlp
    res_rf = y_true - y_rf
    axes[1].scatter(y_lin, res_lin, alpha=0.45, s=25, color="#94A3B8", label="Regresión Lineal")
    axes[1].scatter(y_rf, res_rf, alpha=0.65, s=28, color=PALETTE["navy"], marker="^", label="Random Forest")
    axes[1].scatter(y_mlp, res_mlp, alpha=0.80, s=32, color=PALETTE["primary"], label=best_mlp_name)
    axes[1].axhline(0.0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title("Conjunto Test: Residuos (y_real - y_pred) vs Predicción")
    axes[1].set_xlabel("Y1 Predicho (kWh/m²)")
    axes[1].set_ylabel("Residuo (kWh/m²)")
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_static_subgroup_errors(results: Dict[str, Any], out_path: Path) -> Path:
    sg_mlp = results["subgroups_test"]["best_mlp"]["groups"]
    sg_lin = results["subgroups_test"]["exp_d1_linear"]["groups"]
    sg_rf = results["subgroups_test"]["exp_d2_rf"]["groups"]
    mlp_label = results["subgroups_test"]["best_mlp"]["name"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    group_keys = list(sg_mlp.keys())

    for idx, g_key in enumerate(group_keys):
        rows = []
        for src_name, src_dict in [("Reg. Lineal", sg_lin), (mlp_label, sg_mlp), ("Random Forest", sg_rf)]:
            for item in src_dict[g_key]:
                rows.append({"Subgrupo": item["subgrupo"], "MAE": item["mae"], "Modelo": src_name})
        df_g = pd.DataFrame(rows)
        sns.barplot(
            data=df_g,
            x="Subgrupo",
            y="MAE",
            hue="Modelo",
            palette=["#94A3B8", PALETTE["primary"], PALETTE["navy"]],
            ax=axes[idx],
        )
        axes[idx].set_title(f"MAE en Test por {g_key}")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("MAE (kWh/m²)")
        axes[idx].tick_params(axis="x", rotation=18)
        if idx > 0 and axes[idx].get_legend():
            axes[idx].get_legend().remove()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. GRÁFICOS INTERACTIVOS EN PLOTLY PARA EL DASHBOARD STREAMLIT
# ---------------------------------------------------------------------------

def plotly_y1_distribution(df: pd.DataFrame) -> go.Figure:
    df_p = df.copy()
    df_p["Altura"] = df_p["X5"].map({3.5: "3.5 m (1 planta)", 7.0: "7.0 m (2 plantas)"})
    fig = px.histogram(
        df_p,
        x=TARGET_COL,
        color="Altura",
        nbins=26,
        marginal="box",
        barmode="overlay",
        color_discrete_map={"3.5 m (1 planta)": PALETTE["secondary"], "7.0 m (2 plantas)": PALETTE["navy"]},
        title="Distribución de Y1 (Carga de Calefacción) segmentada por Altura X5",
        labels={TARGET_COL: "Carga de calefacción Y1 (kWh/m²)", "count": "Edificios"},
    )
    fig.update_layout(template="plotly_white", height=420, legend_title="Altura X5")
    return fig


def plotly_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df[ALL_COLS].corr(numeric_only=True).round(3)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        title="Matriz de Correlaciones de Pearson (X1–X8, Y1, Y2)",
    )
    fig.update_layout(template="plotly_white", height=520)
    return fig


def plotly_eda_scatter_compactness(df: pd.DataFrame) -> go.Figure:
    df_p = df.copy()
    df_p["Altura"] = df_p["X5"].map({3.5: "3.5 m (1 planta)", 7.0: "7.0 m (2 plantas)"})
    df_p["Orientación"] = df_p["X6"].map(X6_LABELS)
    fig = px.scatter(
        df_p,
        x="X1",
        y=TARGET_COL,
        color="Altura",
        size="X7",
        hover_data=["X2", "X3", "X4", "Orientación", "X7", "X8", "Y2"],
        color_discrete_map={"3.5 m (1 planta)": PALETTE["secondary"], "7.0 m (2 plantas)": PALETTE["navy"]},
        title="Y1 vs Compacidad Relativa (X1), tamaño = Área Acristalada (X7)",
        labels={"X1": "Compacidad relativa X1", TARGET_COL: "Y1 (kWh/m²)"},
    )
    fig.update_layout(template="plotly_white", height=420)
    return fig


def plotly_eda_categorical(df: pd.DataFrame, col: str) -> go.Figure:
    df_p = df.copy()
    df_p["Altura"] = df_p["X5"].map({3.5: "3.5 m (1 planta)", 7.0: "7.0 m (2 plantas)"})
    if col == "X6":
        df_p["Categoría"] = df_p["X6"].map(X6_LABELS)
        title = "Y1 según Orientación del edificio (X6)"
    elif col == "X8":
        df_p["Categoría"] = df_p["X8"].map(X8_LABELS)
        title = "Y1 según Distribución del acristalamiento (X8)"
    else:
        df_p["Categoría"] = (df_p["X7"] * 100).round(0).astype(int).astype(str) + "%"
        title = "Y1 según Área acristalada (X7)"

    fig = px.box(
        df_p,
        x="Categoría",
        y=TARGET_COL,
        color="Altura",
        color_discrete_map={"3.5 m (1 planta)": PALETTE["secondary"], "7.0 m (2 plantas)": PALETTE["navy"]},
        title=title,
        labels={"Categoría": col, TARGET_COL: "Y1 (kWh/m²)"},
    )
    fig.update_layout(template="plotly_white", height=420)
    return fig


def plotly_mlp_architecture_diagram(layer_sizes: List[int]) -> go.Figure:
    """Dibuja un diagrama limpio de la arquitectura MLP (capas, dimensiones y conexiones)."""
    fig = go.Figure()
    n_layers = len(layer_sizes)
    x_positions = np.linspace(0.08, 0.92, n_layers)

    layer_titles = ["Entrada codificada\n(16 dims)"]
    for i, s in enumerate(layer_sizes[1:-1], start=1):
        layer_titles.append(f"Oculta {i}\n({s} neuronas, ReLU)")
    layer_titles.append("Salida\n(1 neurona, Lineal)")

    node_coords: List[List[tuple[float, float]]] = []
    for x, size in zip(x_positions, layer_sizes):
        display_n = min(size, 8)
        ys = np.linspace(0.18, 0.82, display_n) if display_n > 1 else np.array([0.5])
        node_coords.append([(float(x), float(y)) for y in ys])

    # Conexiones entre capas representativas
    for l_idx in range(n_layers - 1):
        for x0, y0 in node_coords[l_idx]:
            for x1, y1 in node_coords[l_idx + 1]:
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color="rgba(15, 94, 77, 0.16)", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Nodos de cada capa
    palette_nodes = [PALETTE["navy"]] + [PALETTE["primary"]] * max(0, n_layers - 2) + [PALETTE["accent"]]
    for l_idx, (coords, size, title, col) in enumerate(zip(node_coords, layer_sizes, layer_titles, palette_nodes)):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=18, color=col, line=dict(color="white", width=2)),
                name=title.replace("\n", " "),
                hovertemplate=f"{title.replace(chr(10), ' ')}<br>Dimensión real: {size}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=x_positions[l_idx],
            y=0.95,
            text=f"<b>{title.replace(chr(10), '<br>')}</b>",
            showarrow=False,
            font=dict(size=12, color="#0F172A"),
        )
        if size > 8:
            fig.add_annotation(
                x=x_positions[l_idx],
                y=0.08,
                text=f"<i>(mostrando 8 de {size} nodos)</i>",
                showarrow=False,
                font=dict(size=10, color="#64748B"),
            )

    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1.05]),
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )
    return fig


def plotly_learning_curves(results: Dict[str, Any]) -> go.Figure:
    curves = results["learning_curves"]
    labels = {
        "exp_a_mlp_16": "Exp A: MLP (16)",
        "exp_b_mlp_32_16": "Exp B: MLP (32,16)",
        "exp_c1_mlp_64_32_noreg": "Exp C1: MLP (64,32) alpha=1e-6",
        "exp_c2_mlp_64_32_l2": "Exp C2: MLP (64,32) L2=0.05",
        "exp_c3_mlp_64_32_es": "Exp C3: MLP (64,32) L2+EarlyStop",
    }
    colors = {
        "exp_a_mlp_16": "#2A9D8F",
        "exp_b_mlp_32_16": "#0F5E4D",
        "exp_c1_mlp_64_32_noreg": "#E76F51",
        "exp_c2_mlp_64_32_l2": "#1E3A5F",
        "exp_c3_mlp_64_32_es": "#D99B26",
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Curva oficial de pérdida (MLPRegressor)",
            "Evolución de RMSE: Validación (sólida) vs Train (punteada)",
        ),
    )

    for exp_id, loss_vals in results["loss_curves_official"].items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(loss_vals) + 1)),
                y=loss_vals,
                name=labels.get(exp_id, exp_id),
                line=dict(color=colors.get(exp_id, "#333333"), width=2),
            ),
            row=1,
            col=1,
        )

    for exp_id, lc in curves.items():
        fig.add_trace(
            go.Scatter(
                x=lc["epochs"],
                y=lc["val_rmse"],
                name=f"{labels.get(exp_id, exp_id)} (Val)",
                line=dict(color=colors.get(exp_id, "#333333"), width=2.2),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=lc["epochs"],
                y=lc["train_rmse"],
                name=f"{labels.get(exp_id, exp_id)} (Train)",
                line=dict(color=colors.get(exp_id, "#333333"), width=1.3, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(type="log", title_text="Pérdida (escala log)", row=1, col=1)
    fig.update_xaxes(title_text="Época", row=1, col=1)
    fig.update_yaxes(range=[0, 6.0], title_text="RMSE (kWh/m²)", row=1, col=2)
    fig.update_xaxes(title_text="Época", row=1, col=2)
    fig.update_layout(template="plotly_white", height=440, legend=dict(orientation="h", y=-0.22))
    return fig


def plotly_model_comparison(results: Dict[str, Any]) -> go.Figure:
    df_m = pd.DataFrame(results["metrics_table"])
    df_m = df_m[df_m["exp_id"] != "exp_d0_dummy"].copy()
    df_m["Modelo"] = df_m["code"] + " · " + df_m["name"]
    melted = df_m.melt(
        id_vars=["Modelo"],
        value_vars=["val_rmse", "cv_rmse_mean", "test_rmse"],
        var_name="Conjunto",
        value_name="RMSE",
    )
    melted["Conjunto"] = melted["Conjunto"].map(
        {"val_rmse": "Validación", "cv_rmse_mean": "CV 5-Fold (Train+Val)", "test_rmse": "Test final"}
    )
    fig = px.bar(
        melted,
        x="Modelo",
        y="RMSE",
        color="Conjunto",
        barmode="group",
        color_discrete_map={
            "Validación": PALETTE["secondary"],
            "CV 5-Fold (Train+Val)": PALETTE["navy"],
            "Test final": PALETTE["accent"],
        },
        title="Comparación de RMSE en Validación, CV 5-Fold y Test (menor es mejor)",
    )
    fig.update_layout(template="plotly_white", height=450, xaxis_tickangle=-20)
    return fig


def plotly_real_vs_pred_and_residuals(results: Dict[str, Any], selected_exp_id: str, selected_name: str) -> go.Figure:
    pred_df = results["predictions_test_df"]
    y_true = pred_df[TARGET_COL]
    y_pred = pred_df[f"pred_{selected_exp_id}"]
    residuals = pred_df[f"res_{selected_exp_id}"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"y_real vs y_predicho ({selected_name})",
            f"Residuos (y_real - y_pred) vs Predicción ({selected_name})",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color=PALETTE["primary"], size=8, opacity=0.8),
            name="Observaciones Test",
            hovertemplate="Real: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    min_v = float(min(y_true.min(), y_pred.min()) - 1)
    max_v = float(max(y_true.max(), y_pred.max()) + 1)
    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            line=dict(color=PALETTE["accent"], dash="dash", width=2),
            name="Ideal (y = x)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=PALETTE["navy"], size=8, opacity=0.8),
            name="Residuos",
            hovertemplate="Pred: %{x:.2f}<br>Residuo: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color=PALETTE["accent"], row=1, col=2)

    fig.update_xaxes(title_text="Y1 Real (kWh/m²)", row=1, col=1)
    fig.update_yaxes(title_text="Y1 Predicho (kWh/m²)", row=1, col=1)
    fig.update_xaxes(title_text="Y1 Predicho (kWh/m²)", row=1, col=2)
    fig.update_yaxes(title_text="Residuo (kWh/m²)", row=1, col=2)
    fig.update_layout(template="plotly_white", height=440, showlegend=False)
    return fig
