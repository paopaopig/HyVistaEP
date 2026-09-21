# HyVistaEP

**Multi-view hypergraph learning for essential protein prediction**

HyVistaEP integrates temporal gene expression, subcellular localization,
orthology, protein sequence representations, and PPI topology within a unified
hypergraph framework. The repository provides the complete training and
evaluation workflow for *S. cerevisiae*, *H. sapiens*, *D. melanogaster*, and
*E. coli*, together with the processed inputs and ten-fold pretrained models.

## Environment

Python 3.10 and a CUDA-enabled GPU are recommended.

```bash
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Data preparation

Prepare the graph data for all four species:

```bash
python -u src/prepare_data.py --data-root data
```

Data preparation is also invoked automatically when training or evaluation
starts, so this command may be omitted in the standard workflow.

## Evaluate pretrained models

Evaluate one species with the supplied ten-fold checkpoints:

```bash
python -u src/test.py \
  --species yeast \
  --data-root data \
  --checkpoint-dir checkpoints/yeast
```

Evaluate all four species:

```bash
bash scripts/test_all_species.sh
```

Accepted species names are `yeast`, `human`, `melanogaster`, and `coli`.

## Train and evaluate

Run a short end-to-end training and evaluation test:

```bash
bash scripts/smoke_test.sh
```

Run ten-fold training for one species:

```bash
python -u src/train.py \
  --species yeast \
  --data-root data \
  --output-dir outputs/yeast \
  --seed 45 \
  --folds 10
```

Evaluate the resulting checkpoints:

```bash
python -u src/test.py \
  --species yeast \
  --data-root data \
  --checkpoint-dir outputs/yeast
```
