#!/usr/bin/env bash
set -euo pipefail

# Script autocontenido para levantar la aplicación Streamlit del Grupo 02
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "Creando entorno virtual .venv..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

if [ ! -f "artifacts/results_summary.json" ]; then
  echo "Generando artefactos locales de entrenamiento..."
  .venv/bin/python src/train_pipeline.py
fi

echo "Iniciando dashboard Streamlit en http://localhost:8501 ..."
exec .venv/bin/streamlit run app.py --server.port 8501
