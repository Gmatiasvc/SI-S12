# Estimación de Carga Térmica de Edificios con MLP (Grupo 02)

Proyecto aplicativo autocontenido de **Sistemas Inteligentes y Machine Learning** para predecir la carga de calefacción (`Y1`, en kWh/m²) de edificios a partir de 8 características arquitectónicas (`X1`–`X8`) del conjunto oficial **UCI Energy Efficiency (ID 242)**, utilizando **Perceptrones Multicapa (MLP)** en `scikit-learn` y comparándolos contra baselines tabulares (`DummyRegressor`, `Regresión Lineal` y `Random Forest`).

---

## Características principales (100 % autocontenido)

Al clonar este repositorio (`git clone`), **no necesitas descargar datasets externos ni entrenar manualmente desde cero para explorar**:
- El dataset oficial (`data/ENB2012_data.xlsx` y `data/energy_efficiency.csv`) ya está incluido localmente junto con `data/metadata.json`.
- Los modelos serializados (`artifacts/models/*.joblib`), métricas reales (`artifacts/results_summary.json`, `artifacts/metrics_comparison.csv`) y predicciones (`artifacts/predictions_test.csv`) están listos para inferencia instantánea.
- Las 7 figuras de alta resolución para el informe y las diapositivas están en `figures/`.
- El notebook didáctico con las **16 etapas obligatorias** ejecutadas está en `notebooks/MLP_Carga_Termica_16_Etapas.ipynb`.
- La documentación técnica completa en **LaTeX** (en términos sencillos + guía de 15 diapositivas) está en `docs/documentacion_proyecto_mlp.tex`.

---

## Estructura del proyecto

```text
├── app.py                                      # Dashboard interactivo en Streamlit (7 vistas)
├── requirements.txt                            # Dependencias con versiones fijas
├── run.sh                                      # Script rápido de inicio (Linux/macOS)
├── README.md                                   # Esta guía
├── data/
│   ├── ENB2012_data.xlsx                       # Excel oficial UCI Energy Efficiency (ID 242)
│   ├── energy_efficiency.csv                   # Copia CSV limpia (768 × 10)
│   └── metadata.json                           # Trazabilidad, rangos y diccionario de variables
├── src/
│   ├── __init__.py
│   ├── data_loader.py                          # Carga local offline y metadata UCI
│   ├── modeling.py                             # Pipelines, MLP (A, B, C), baselines (D), CV 5-Fold, subgrupos
│   ├── plots.py                                # Gráficos estáticos (Matplotlib/Seaborn) e interactivos (Plotly)
│   └── train_pipeline.py                       # Script maestro reproducible (SEED=42)
├── artifacts/
│   ├── results_summary.json                    # Métricas completas (Train, Val, CV5, Test, tiempos, ablación)
│   ├── metrics_comparison.csv                  # Tabla resumen de experimentos A–D
│   ├── predictions_test.csv                    # Predicciones y residuos en Test (n=116)
│   └── models/                                 # Pipelines entrenados (.joblib)
├── figures/                                    # 7 gráficos PNG de alta resolución (220 DPI)
├── notebooks/
│   └── MLP_Carga_Termica_16_Etapas.ipynb       # Notebook didáctico ejecutado paso a paso
└── docs/
    └── documentacion_proyecto_mlp.tex          # Documento LaTeX listo para compilar a PDF
```

---

## Cómo ejecutar tras `git clone`

### Opción rápida (Linux / macOS)
```bash
git clone <URL_DEL_REPOSITORIO>
cd SI-S12
./run.sh
```
El script crea `.venv` automáticamente si no existe, instala `requirements.txt` (solo la primera vez) y abre el dashboard en `http://localhost:8501`.

### Opción manual paso a paso (Linux / macOS / Windows)
```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # En Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Instalar dependencias fijadas
pip install -r requirements.txt

# 3. (Opcional) Re-entrenar todo desde cero con SEED=42
python src/train_pipeline.py

# 4. Ejecutar el dashboard Streamlit
streamlit run app.py
```

---

## Reproducibilidad y licencia

- **Semilla global fija:** `SEED = 42` en todas las particiones estratificadas, validación cruzada `KFold(n_splits=5)` y modelos (`MLPRegressor`, `RandomForestRegressor`).
- **Preprocesamiento sin fuga de datos:** `ColumnTransformer` + `StandardScaler` + `OneHotEncoder` encapsulados dentro de `sklearn.pipeline.Pipeline`.
- **Fuente y licencia del dataset:** Athanasios Tsanas & Angeliki Xifara (2012), *UCI Machine Learning Repository — Energy Efficiency (ID 242)*, bajo licencia **CC BY 4.0**.
