#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${NIGHT7A_PYTHON:-/root/miniconda3/envs/SpaLORA/bin/python}"

if [[ "${CUDA_VISIBLE_DEVICES+x}" != x || "${CUDA_VISIBLE_DEVICES}" != "" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be explicitly set to the empty string" >&2
  exit 70
fi
for key in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS; do
  if [[ "${!key:-}" != "1" ]]; then
    echo "${key} must equal 1" >&2
    exit 71
  fi
done

cd "${repo}"
"${python_bin}" scripts/night7a_p0.py --source-only

for dataset in a1 tonsil d1 p22; do
  if [[ "${dataset}" == "a1" || "${dataset}" == "tonsil" ]]; then
    last_seed=4
  else
    last_seed=9
  fi
  for seed in $(seq 0 "${last_seed}"); do
    "${python_bin}" scripts/night7a_p0.py \
      --cell-dataset "${dataset}" --cell-seed "${seed}"
  done
done

"${python_bin}" scripts/night7a_p0.py --aggregate-only
