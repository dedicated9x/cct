#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/profile_one_epoch.sh local
#   ./scripts/profile_one_epoch.sh cluster

TAG="${1:-local}"
OUT_DIR="${2:-.EXCLUDED/profilowanie}"
OUT_LOG="${OUT_DIR}/profiler_${TAG}.log"
SYS_LOG="${OUT_DIR}/system_${TAG}.log"

mkdir -p "${OUT_DIR}"

echo "===> Writing logs to:"
echo "     ${OUT_LOG}"
echo "     ${SYS_LOG}"

{
  echo "timestamp: $(date -Is)"
  echo "hostname: $(hostname)"
  echo "uname: $(uname -a)"
  echo "pwd: $(pwd)"
  echo

  echo "--- cpu ---"
  command -v lscpu >/dev/null 2>&1 && lscpu || true
  echo

  echo "--- cgroups (cpu) ---"
  if [[ -f /sys/fs/cgroup/cpu.max ]]; then
    echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
  fi
  if [[ -f /sys/fs/cgroup/cpu.stat ]]; then
    echo "--- cpu.stat ---"
    sed -n '1,200p' /sys/fs/cgroup/cpu.stat || true
  fi
  if [[ -f /sys/fs/cgroup/cpuset.cpus.effective ]]; then
    echo "cpuset.cpus.effective: $(cat /sys/fs/cgroup/cpuset.cpus.effective)"
  fi
  if [[ -f /sys/fs/cgroup/cpuset.cpus ]]; then
    echo "cpuset.cpus: $(cat /sys/fs/cgroup/cpuset.cpus)"
  fi
  echo

  echo "--- torch cpu backend ---"
  python - <<'PY'
import os, torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("mkldnn_available:", torch.backends.mkldnn.is_available())
print("mkldnn_enabled:", torch.backends.mkldnn.enabled)
print("num_threads:", torch.get_num_threads())
print("num_interop_threads:", torch.get_num_interop_threads())
keys = [
    "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
    "MKL_DYNAMIC","KMP_AFFINITY","KMP_BLOCKTIME",
    "TORCH_DISABLE_MKLDNN",
    "CCT_ENABLE_PROFILER","CCT_MAX_EPOCHS","CCT_PROFILER_TYPE",
]
print("selected_env:", {k: os.environ.get(k) for k in keys})
PY

} | tee "${SYS_LOG}"

echo
echo "===> Running: profiling 1 epoch"

export CCT_ENABLE_PROFILER=1
export CCT_MAX_EPOCHS=1
export CCT_PROFILER_TYPE="${CCT_PROFILER_TYPE:-advanced}"

set +e
python main.py 2>&1 | tee "${OUT_LOG}"
exit_code=${PIPESTATUS[0]}
set -e

echo "===> Done (exit_code=${exit_code})"

