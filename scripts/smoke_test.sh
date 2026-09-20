#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python -u src/train.py \
  --species yeast \
  --data-root data \
  --output-dir outputs/smoke_test_yeast \
  --seed 45 \
  --folds 2 \
  --epochs 1

python -u src/test.py \
  --species yeast \
  --data-root data \
  --checkpoint-dir outputs/smoke_test_yeast

echo "PASS: training and testing smoke test completed."
