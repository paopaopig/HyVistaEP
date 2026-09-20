import argparse
import os
from pathlib import Path
import re

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from config import get_config
from data_load import load_gat_data, load_gat_data_coli
from model import DyHTSAGCN


def configure_graph_environment(cfg):
    if cfg["subcell_k"] is not None:
        os.environ["SUBCELL_K"] = str(cfg["subcell_k"])
    os.environ["HOMOLOGY_Q"] = str(cfg["homology_q"])


def load_graph(cfg):
    loader = load_gat_data_coli if cfg["species"] == "coli" else load_gat_data
    return loader(
        cfg["species"], cfg["dataset"], cfg["k_value"], cfg["q_th"],
        esm_path=cfg["esm_path"],
    )


def build_model(cfg, device):
    return DyHTSAGCN(
        cfg["gene_dim"], cfg["subcell_dim"], cfg["homology_dim"],
        cfg["conv_hidden1"], cfg["conv_hidden2"], cfg["species"],
        cfg["dataset"], cfg["k_value"], cfg["drop_mlp"], cfg["drop_conv"],
        cfg["mlp_hidden"], cfg["classifier_hidden1"], cfg["classifier_hidden2"],
        cfg["drop_classifier"], cfg["T"], esm_layer_input=cfg["esm_dim"],
        use_subcellular=cfg["use_subcellular"],
        attention_heads=cfg["attention_heads"],
    ).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one saved 10-fold model group"
    )
    parser.add_argument("--species", required=True,
                        choices=("human", "yeast", "coli", "melanogaster"))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Root containing <species>/PPI/BIOGRID.",
    )
    parser.add_argument("--T", type=int)
    parser.add_argument("--homology-q", type=float)
    parser.add_argument("--subcell-k", type=float)
    args = parser.parse_args()
    os.environ["HYVISTAEP_DATA_ROOT"] = str(
        Path(args.data_root).expanduser().resolve()
    )
    cfg = get_config(args.species)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_paths = list(checkpoint_dir.glob("best_model_fold_*.pth"))
    fold_pattern = re.compile(r"best_model_fold_(\d+)\.pth$")
    checkpoint_paths = [
        path for path in checkpoint_paths if fold_pattern.match(path.name)
    ]
    checkpoint_paths.sort(
        key=lambda path: int(fold_pattern.match(path.name).group(1))
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No best_model_fold_*.pth files found in {checkpoint_dir}"
        )

    first_checkpoint = torch.load(
        checkpoint_paths[0],
        map_location="cpu", weights_only=False,
    )
    saved_cfg = first_checkpoint.get("config", {})
    cfg["T"] = args.T if args.T is not None else saved_cfg.get("T", cfg["T"])
    cfg["q_th"] = saved_cfg.get("q_th", cfg["q_th"])
    cfg["subcell_k"] = (
        args.subcell_k if args.subcell_k is not None
        else saved_cfg.get("subcell_k", cfg["subcell_k"])
    )
    cfg["homology_q"] = (
        args.homology_q if args.homology_q is not None
        else saved_cfg.get("homology_q", cfg["homology_q"])
    )
    configure_graph_environment(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = load_graph(cfg)
    for graph in graphs:
        graph.to(device)

    aucs, auprs = [], []
    for fold, path in enumerate(checkpoint_paths, start=1):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        indices = torch.as_tensor(
            checkpoint["test_indices"], dtype=torch.long, device=device
        )
        with torch.no_grad():
            probabilities = torch.sigmoid(model(graphs)[indices]).cpu().numpy()
        labels = graphs[0].y[indices].cpu().numpy()
        auc = roc_auc_score(labels, probabilities)
        aupr = average_precision_score(labels, probabilities)
        aucs.append(auc)
        auprs.append(aupr)
        saved_fold = checkpoint.get("fold", fold)
        print(f"Fold {saved_fold}: AUC={auc:.6f}, AUPR={aupr:.6f}")

    mean_auc = float(np.mean(aucs))
    mean_aupr = float(np.mean(auprs))
    print(f"{args.species} AUC: {mean_auc:.6f} +/- {np.std(aucs):.6f}")
    print(f"{args.species} AUPR: {mean_aupr:.6f} +/- {np.std(auprs):.6f}")


if __name__ == "__main__":
    main()
