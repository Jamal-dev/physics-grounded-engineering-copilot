#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Create the environment with: conda env create -f environment.yml" >&2
  exit 1
fi

env_name="${CONDA_ENV_NAME:-local-rag}"
exec conda run --no-capture-output -n "$env_name" streamlit run app.py
