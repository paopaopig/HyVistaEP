#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for species in yeast human melanogaster coli; do
  echo
  echo "===== Testing ${species} ====="
  python -u src/test.py \
    --species "$species" \
    --data-root data \
    --checkpoint-dir "checkpoints/$species"
done
