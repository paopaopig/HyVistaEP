import argparse
import os
from pathlib import Path

import numpy as np


SPECIES = ("yeast", "human", "melanogaster", "coli")


def prepare_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    names = np.load(dataset_dir / "protein_names.npy", allow_pickle=True)
    node_count = len(names)
    edge_path = dataset_dir / "ppi_edges.npz"
    spin_path = dataset_dir / "SPIN.npy"

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing PPI edge data: {edge_path}")

    if spin_path.exists():
        spin = np.load(spin_path, mmap_mode="r")
        if spin.shape != (node_count, node_count):
            raise ValueError(
                f"Unexpected graph shape in {spin_path}: {spin.shape}"
            )
        return spin_path

    with np.load(edge_path, allow_pickle=False) as edge_data:
        edge_index = edge_data["edge_index"]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Unexpected edge_index shape: {edge_index.shape}")
    if edge_index.size:
        if edge_index.min() < 0 or edge_index.max() >= node_count:
            raise ValueError(f"PPI edge index is out of range for {node_count} nodes")
        if np.any(edge_index[0] >= edge_index[1]):
            raise ValueError("PPI edges must be unique upper-triangle pairs")

    temporary_path = Path(f"{spin_path}.tmp")
    try:
        spin = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float64,
            shape=(node_count, node_count),
        )
        spin[edge_index[0], edge_index[1]] = 1.0
        spin[edge_index[1], edge_index[0]] = 1.0
        spin.flush()
        del spin
        os.replace(temporary_path, spin_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return spin_path


def prepare_all(data_root, species="all", dataset="BIOGRID"):
    selected = SPECIES if species == "all" else (species,)
    outputs = []
    for name in selected:
        dataset_dir = Path(data_root) / name / "PPI" / dataset
        outputs.append(prepare_dataset(dataset_dir))
    return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--species", choices=("all",) + SPECIES, default="all")
    parser.add_argument("--dataset", default="BIOGRID")
    return parser.parse_args()


def main():
    args = parse_args()
    for output in prepare_all(args.data_root, args.species, args.dataset):
        print(output)


if __name__ == "__main__":
    main()
