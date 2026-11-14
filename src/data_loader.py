"""Carga, trazabilidad y metadatos del dataset UCI Energy Efficiency (ID 242)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
XLSX_PATH = DATA_DIR / "ENB2012_data.xlsx"
CSV_PATH = DATA_DIR / "energy_efficiency.csv"
METADATA_PATH = DATA_DIR / "metadata.json"

UCI_ID = 242
UCI_PAGE_URL = "https://archive.ics.uci.edu/dataset/242/energy+efficiency"
UCI_XLSX_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

FEATURE_COLS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
CONTINUOUS_COLS = ["X1", "X2", "X3", "X4", "X5", "X7"]
CATEGORICAL_COLS = ["X6", "X8"]
TARGET_COL = "Y1"
SECONDARY_TARGET_COL = "Y2"
ALL_COLS = FEATURE_COLS + [TARGET_COL, SECONDARY_TARGET_COL]

X6_LABELS: Dict[int, str] = {
    2: "Norte (2)",
    3: "Este (3)",
    4: "Sur (4)",
    5: "Oeste (5)",
}

X8_LABELS: Dict[int, str] = {
    0: "Sin acristalamiento (0)",
    1: "Uniforme (1)",
    2: "Norte (2)",
    3: "Este (3)",
    4: "Sur (4)",
    5: "Oeste (5)",
}

VARIABLE_INFO: Dict[str, Dict[str, Any]] = {
    "X1": {
        "name": "Compacidad relativa",
        "role": "Entrada (Continua)",
        "type": "Continua",
        "unit": "Adimensional (ratio)",
        "description": "Relación entre el área superficial de un cubo de igual volumen y el área del edificio.",
        "min": 0.62,
        "max": 0.98,
    },
    "X2": {
        "name": "Área superficial",
        "role": "Entrada (Continua)",
        "type": "Continua",
        "unit": "m²",
        "description": "Superficie total exterior del edificio (muros, techo y base).",
        "min": 514.5,
        "max": 808.5,
    },
    "X3": {
        "name": "Área de paredes",
        "role": "Entrada (Continua)",
        "type": "Continua",
        "unit": "m²",
        "description": "Superficie total de las paredes verticales exteriores.",
        "min": 245.0,
        "max": 416.5,
    },
    "X4": {
        "name": "Área de techo",
        "role": "Entrada (Continua)",
        "type": "Continua",
        "unit": "m²",
        "description": "Superficie de la cubierta del edificio (proyección horizontal).",
        "min": 110.25,
        "max": 220.5,
    },
    "X5": {
        "name": "Altura total",
        "role": "Entrada (Continua)",
        "type": "Continua",
        "unit": "m",
        "description": "Altura total de la edificación (3.5 m para 1 planta o 7.0 m para 2 plantas).",
        "min": 3.5,
        "max": 7.0,
    },
    "X6": {
        "name": "Orientación",
        "role": "Entrada (Categórica nominal)",
        "type": "Nominal (codificada 2–5)",
        "unit": "Código nominal",
        "description": "Orientación cardinal principal del edificio: 2=Norte, 3=Este, 4=Sur, 5=Oeste.",
        "min": 2,
        "max": 5,
    },
    "X7": {
        "name": "Área acristalada",
        "role": "Entrada (Continua)",
        "type": "Continua",
        "unit": "Proporción del piso (0.00–0.40)",
        "description": "Proporción de área de ventanas respecto al área total del suelo (0%, 10%, 25%, 40%).",
        "min": 0.0,
        "max": 0.4,
    },
    "X8": {
        "name": "Distribución del acristalamiento",
        "role": "Entrada (Categórica nominal)",
        "type": "Nominal (codificada 0–5)",
        "unit": "Código nominal",
        "description": "Distribución direccional de ventanas: 0=Sin ventanas, 1=Uniforme, 2=Norte, 3=Este, 4=Sur, 5=Oeste.",
        "min": 0,
        "max": 5,
    },
    "Y1": {
        "name": "Carga de calefacción (Heating Load)",
        "role": "Objetivo principal (y)",
        "type": "Continua",
        "unit": "kWh/m²",
        "description": "Energía térmica requerida para calefacción anual simulada en Ecotect.",
        "min": 6.01,
        "max": 43.1,
    },
    "Y2": {
        "name": "Carga de refrigeración (Cooling Load)",
        "role": "Objetivo secundario (excluida de X)",
        "type": "Continua",
        "unit": "kWh/m²",
        "description": "Energía requerida para enfriamiento. No debe usarse como predictor de Y1 (evita data leakage).",
        "min": 10.9,
        "max": 48.03,
    },
}


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, elimina filas vacías y garantiza tipos numéricos esperados."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing_cols = [c for c in ALL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing_cols}")

    df = df[ALL_COLS].dropna(how="all").reset_index(drop=True)
    for col in ALL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=ALL_COLS).reset_index(drop=True)
    df["X6"] = df["X6"].astype(int)
    df["X8"] = df["X8"].astype(int)
    return df


def download_and_save_dataset(force: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Descarga el archivo oficial ENB2012_data.xlsx y metadatos UCI (solo si hace falta)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if force or not XLSX_PATH.exists():
        try:
            urllib.request.urlretrieve(UCI_XLSX_URL, XLSX_PATH)
        except Exception:
            # Respaldo mediante ucimlrepo si la descarga directa del xlsx falla
            from ucimlrepo import fetch_ucirepo

            ds = fetch_ucirepo(id=UCI_ID)
            df_uciml = pd.concat([ds.data.features, ds.data.targets], axis=1)
            df_uciml = _clean_dataframe(df_uciml)
            df_uciml.to_excel(XLSX_PATH, index=False)

    df = pd.read_excel(XLSX_PATH)
    df = _clean_dataframe(df)
    df.to_csv(CSV_PATH, index=False)

    uci_meta: Dict[str, Any] = {}
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=UCI_ID)
        uci_meta = {
            "name": ds.metadata.get("name", "Energy Efficiency"),
            "uci_id": ds.metadata.get("uci_id", UCI_ID),
            "num_instances": int(len(df)),
            "num_features": len(FEATURE_COLS),
            "abstract": ds.metadata.get("abstract", ""),
            "creators": ds.metadata.get("creators", ["Athanasios Tsanas", "Angeliki Xifara"]),
            "year_of_dataset_creation": ds.metadata.get("year_of_dataset_creation", 2012),
            "repository_url": ds.metadata.get("repository_url", UCI_PAGE_URL),
            "data_url": UCI_XLSX_URL,
            "license": "CC BY 4.0 (UCI Machine Learning Repository)",
        }
    except Exception:
        uci_meta = {
            "name": "Energy Efficiency",
            "uci_id": UCI_ID,
            "num_instances": int(len(df)),
            "num_features": len(FEATURE_COLS),
            "abstract": (
                "Conjunto de 768 edificios simulados en Ecotect con 12 formas de edificación "
                "variadas en área acristalada, distribución del acristalamiento y orientación."
            ),
            "creators": ["Athanasios Tsanas", "Angeliki Xifara"],
            "year_of_dataset_creation": 2012,
            "repository_url": UCI_PAGE_URL,
            "data_url": UCI_XLSX_URL,
            "license": "CC BY 4.0 (UCI Machine Learning Repository)",
        }

    # Estadísticas descriptivas locales para trazabilidad y control de extrapolación
    ranges: Dict[str, Dict[str, float]] = {}
    for col in ALL_COLS:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "median": float(df[col].median()),
            "unique_count": int(df[col].nunique()),
        }

    metadata: Dict[str, Any] = {
        "uci": uci_meta,
        "local_files": {
            "xlsx": str(XLSX_PATH.relative_to(PROJECT_ROOT)),
            "csv": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        },
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "missing_values": {col: int(df[col].isna().sum()) for col in ALL_COLS},
        "duplicate_rows": int(df.duplicated().sum()),
        "ranges": ranges,
        "variables": VARIABLE_INFO,
        "x6_labels": {str(k): v for k, v in X6_LABELS.items()},
        "x8_labels": {str(k): v for k, v in X8_LABELS.items()},
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return df, metadata


def load_dataset(prefer_local: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga el dataset desde archivos locales del repositorio (100% autocontenido).

    Si los archivos locales aún no existen, intenta descargarlos y guardarlos en data/.
    """
    if prefer_local and CSV_PATH.exists() and METADATA_PATH.exists():
        df = _clean_dataframe(pd.read_csv(CSV_PATH))
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return df, metadata

    if prefer_local and XLSX_PATH.exists():
        df = _clean_dataframe(pd.read_excel(XLSX_PATH))
        df.to_csv(CSV_PATH, index=False)
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return df, metadata

    return download_and_save_dataset(force=False)


def get_variable_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una tabla descriptiva completa de las 10 variables para EDA y reporte."""
    rows = []
    for col in ALL_COLS:
        info = VARIABLE_INFO[col]
        series = df[col]
        rows.append(
            {
                "Variable": col,
                "Nombre": info["name"],
                "Rol": info["role"],
                "Tipo": info["type"],
                "Unidad": info["unit"],
                "Mín": round(float(series.min()), 4),
                "Mediana": round(float(series.median()), 4),
                "Media": round(float(series.mean()), 4),
                "Máx": round(float(series.max()), 4),
                "Desv. Est.": round(float(series.std()), 4),
                "Únicos": int(series.nunique()),
                "Nulos": int(series.isna().sum()),
            }
        )
    return pd.DataFrame(rows)
