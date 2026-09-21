from copy import deepcopy
import os
from pathlib import Path


CONFIGS = {
    "human": dict(
        dataset="BIOGRID", gene_dim=64, subcell_dim=1024, homology_dim=162,
        use_subcellular=True, attention_heads=8, k_value=3.0, q_th=30,
        T=3, lr=0.003, weight_decay=2e-4, drop_mlp=0.3, drop_conv=0.3,
        drop_classifier=0.2, mlp_hidden=256,
        pos_weight_scale=1.0, subcell_k=1.0, homology_q=100,
    ),
    "yeast": dict(
        dataset="BIOGRID", gene_dim=36, subcell_dim=1024, homology_dim=62,
        use_subcellular=True, attention_heads=8, k_value=3.0, q_th=30,
        T=5, lr=0.005, weight_decay=2e-4, drop_mlp=0.2, drop_conv=0.2,
        drop_classifier=0.2, mlp_hidden=256,
        pos_weight_scale=1.0, subcell_k=1.0, homology_q=15,
    ),
    "melanogaster": dict(
        dataset="BIOGRID", gene_dim=66, subcell_dim=1024, homology_dim=203,
        use_subcellular=True, attention_heads=8, k_value=3.0, q_th=50,
        T=6, lr=0.002, weight_decay=5e-4, drop_mlp=0.3, drop_conv=0.3,
        drop_classifier=0.2, mlp_hidden=64,
        pos_weight_scale=0.75, subcell_k=1.0, homology_q=80,
    ),
    "coli": dict(
        dataset="BIOGRID", gene_dim=24, subcell_dim=0, homology_dim=189,
        use_subcellular=False, attention_heads=2, k_value=3.0, q_th=25,
        T=7, lr=0.005, weight_decay=2e-4, drop_mlp=0.3, drop_conv=0.3,
        drop_classifier=0.2, mlp_hidden=256,
        pos_weight_scale=0.65, subcell_k=None, homology_q=35,
    ),
}


def get_config(species, data_root=None):
    if species not in CONFIGS:
        raise ValueError(f"Unsupported species: {species}; choose {sorted(CONFIGS)}")
    cfg = deepcopy(CONFIGS[species])
    cfg["species"] = species
    if data_root is None:
        data_root = os.environ.get(
            "HYVISTAEP_DATA_ROOT",
            str(Path(__file__).resolve().parents[1] / "data"),
        )
    cfg["data_root"] = str(Path(data_root).expanduser().resolve())
    cfg["esm_path"] = str(
        Path(cfg["data_root"]) / species / "PPI" / cfg["dataset"]
        / "features" / "esm3_embeddings.pt"
    )
    cfg.update(conv_hidden1=64, conv_hidden2=32, classifier_hidden1=64,
               classifier_hidden2=32, esm_dim=1536, epochs=40, folds=10, seed=45)
    return cfg
