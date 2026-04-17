#!/usr/bin/env bash
set -euo pipefail

# Installs private mlflow_oauth2_client into cct_corrected_py312.
# Usage:
#   scripts/install_mlflow_oauth2_client.sh /absolute/path/to/mlflow-ouath2
#
# The source repo must already be available locally (e.g. cloned manually
# with your own GitLab credentials).

ENV_NAME="cct_corrected_py312"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/mlflow-ouath2"
  exit 1
fi

PKG_PATH="$1"

if [[ ! -d "$PKG_PATH" ]]; then
  echo "Directory does not exist: $PKG_PATH"
  exit 1
fi

if [[ ! -f "$PKG_PATH/pyproject.toml" && ! -f "$PKG_PATH/setup.py" ]]; then
  echo "No pyproject.toml or setup.py found in: $PKG_PATH"
  exit 1
fi

echo "Installing mlflow_oauth2_client from local source: $PKG_PATH"
mamba run -n "$ENV_NAME" python -m pip install "$PKG_PATH"

echo "Installed packages:"
mamba run -n "$ENV_NAME" python -c "import importlib.metadata as m; print('mlflow', m.version('mlflow')); print('mlflow_oauth2_client', m.version('mlflow_oauth2_client'))"
