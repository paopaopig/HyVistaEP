# HyVistaEP

HyVistaEP predicts essential proteins in yeast, human, fruit fly, and *E. coli*.

## Installation

Python 3.10 and a CUDA-enabled GPU are recommended.

```bash
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Data preparation

Prepare the data for all species:

```bash
python -u src/prepare_data.py --data-root data
```

The same preparation is performed automatically when training or testing starts.

## Test

Test one species:

```bash
python -u src/test.py \
  --species yeast \
  --data-root data \
  --checkpoint-dir checkpoints/yeast
```

Test all species:

```bash
bash scripts/test_all_species.sh
```

Valid species names are `yeast`, `human`, `melanogaster`, and `coli`.

## Train

Run a short training and testing example:

```bash
bash scripts/smoke_test.sh
```

Run ten-fold training:

```bash
python -u src/train.py \
  --species yeast \
  --data-root data \
  --output-dir outputs/yeast \
  --seed 45 \
  --folds 10
```

Test the trained models:

```bash
python -u src/test.py \
  --species yeast \
  --data-root data \
  --checkpoint-dir outputs/yeast
```
