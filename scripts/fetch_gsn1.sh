#!/usr/bin/env bash
set -euo pipefail

URL="https://www.mimuw.edu.pl/~ciebie/gsn-2021-1.zip"
DATA_DIR="data"

usage() {
  cat <<'EOF'
Usage:
  scripts/fetch_gsn1.sh [--force]

Downloads https://www.mimuw.edu.pl/~ciebie/gsn-2021-1.zip, unzips it, and places
its contents under ./data/.

Options:
  --force   re-download and re-extract even if data seems present
EOF
}

FORCE=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi
if [[ -n "${1:-}" && "${1:-}" != "--force" ]]; then
  echo "Unknown argument: ${1}" >&2
  usage >&2
  exit 2
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

download() {
  local url="$1"
  local out="$2"
  local ua="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

  if command -v curl >/dev/null 2>&1; then
    if curl -L --fail --retry 3 --retry-delay 1 -A "$ua" -o "$out" "$url"; then
      return 0
    fi

    # Some hosts block "generic" clients or behave differently on HTTP vs HTTPS.
    if [[ "$url" == https://* ]]; then
      local http_url="http://${url#https://}"
      echo "Download failed; retrying over HTTP: $http_url" >&2
      curl -L --fail --retry 3 --retry-delay 1 -A "$ua" -o "$out" "$http_url"
      return 0
    fi
    return 1
  elif command -v wget >/dev/null 2>&1; then
    if wget -O "$out" --user-agent="$ua" "$url"; then
      return 0
    fi
    if [[ "$url" == https://* ]]; then
      local http_url="http://${url#https://}"
      echo "Download failed; retrying over HTTP: $http_url" >&2
      wget -O "$out" --user-agent="$ua" "$http_url"
      return 0
    fi
    return 1
  else
    echo "Need either curl or wget to download files." >&2
    exit 1
  fi
}

require_cmd unzip

mkdir -p "$DATA_DIR"

if [[ $FORCE -eq 0 ]]; then
  if [[ -f "$DATA_DIR/gsn1/data/labels.csv" ]]; then
    echo "Looks like dataset is already present at $DATA_DIR/gsn1/data/labels.csv"
    echo "Run with --force to re-download/re-extract."
    exit 0
  fi
fi

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

zip_path="$tmp_dir/gsn-2021-1.zip"
extract_dir="$tmp_dir/extract"
mkdir -p "$extract_dir"

echo "Downloading: $URL"
download "$URL" "$zip_path"

echo "Extracting zip"
unzip -q "$zip_path" -d "$extract_dir"

echo "Placing into ./$DATA_DIR/"

# Expected by code: data/gsn1/data/labels.csv
# The zip currently contains: data/labels.csv (and images)
if [[ -f "$extract_dir/data/labels.csv" ]]; then
  rm -rf "$DATA_DIR/gsn1"
  mkdir -p "$DATA_DIR/gsn1"
  rm -rf "$DATA_DIR/gsn1/data"
  mv "$extract_dir/data" "$DATA_DIR/gsn1/data"
else
  shopt -s dotglob nullglob
  for item in "$extract_dir"/*; do
    base="$(basename "$item")"
    rm -rf "$DATA_DIR/$base"
    mv "$item" "$DATA_DIR/"
  done
  shopt -u dotglob nullglob
fi

echo "Done."
echo "Expected file: $DATA_DIR/gsn1/data/labels.csv"
