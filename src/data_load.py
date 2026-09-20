import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
import os

from prepare_data import prepare_dataset

torch.backends.cudnn.enabled = False


def dataset_root(species, dataset):
    data_root = os.environ.get("HYVISTAEP_DATA_ROOT")
    if not data_root:
        raise RuntimeError(
            "HYVISTAEP_DATA_ROOT is not set. Pass --data-root to train.py/test.py."
        )
    root = os.path.join(os.path.abspath(os.path.expanduser(data_root)), species,
                        "PPI", dataset)
    prepare_dataset(root)
    return root


def build_subcell_hyperedge_index(
    fea_sub_csv,
    ppi_neighbors,
    k=1.0,
    min_size=3,
    max_ratio=None,
    use_positive_only=True,
    min_score_floor=None,
):
    frame = pd.read_csv(fea_sub_csv)
    if frame.shape[1] <= 2:
        raise ValueError("Subcellular CSV must contain metadata and feature columns")

    feature_frame = frame.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_matrix = feature_frame.to_numpy(dtype=float)
    num_nodes = len(feature_frame)
    unique_edges = []
    seen_edges = set()
    stats = []

    for column_id, column in enumerate(feature_frame.columns):
        scores = feature_matrix[:, column_id]
        positive_scores = scores[scores > 0]
        if len(positive_scores) == 0:
            stats.append({"column": column, "kept": False, "reason": "all_zero"})
            continue

        stat_scores = positive_scores if use_positive_only else scores
        threshold = float(np.mean(stat_scores) + k * np.std(stat_scores))
        valid_mask = scores >= threshold
        if min_score_floor is not None:
            valid_mask &= scores > min_score_floor
        members = np.flatnonzero(valid_mask).tolist()

        if len(members) < min_size:
            stats.append({
                "column": column,
                "kept": False,
                "reason": "size_below_min",
                "threshold": threshold,
                "size": len(members),
            })
            continue
        if max_ratio is not None and len(members) > max_ratio * num_nodes:
            stats.append({
                "column": column,
                "kept": False,
                "reason": "size_above_max_ratio",
                "threshold": threshold,
                "size": len(members),
            })
            continue

        member_set = set(members)
        unvisited = set(members)
        added_count = 0
        duplicate_count = 0
        while unvisited:
            start = unvisited.pop()
            component = [start]
            stack = [start]
            while stack:
                node = stack.pop()
                connected = [
                    int(neighbor)
                    for neighbor in ppi_neighbors[node]
                    if int(neighbor) in member_set
                    and int(neighbor) in unvisited
                ]
                for neighbor in connected:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)

            if len(component) < min_size:
                continue
            component.sort()
            edge_key = tuple(component)
            if edge_key in seen_edges:
                duplicate_count += 1
                continue
            seen_edges.add(edge_key)
            unique_edges.append(component)
            added_count += 1

        stats.append({
            "column": column,
            "kept": added_count > 0,
            "reason": "ppi_connected_components",
            "threshold": threshold,
            "size": len(members),
            "hyperedges_added": added_count,
            "duplicates": duplicate_count,
        })

    node_indices = []
    hyperedge_ids = []
    for hyperedge_id, members in enumerate(unique_edges):
        node_indices.extend(members)
        hyperedge_ids.extend([hyperedge_id] * len(members))
    if not node_indices:
        raise ValueError("No valid subcellular hyperedges were constructed")

    hyperedge_index = torch.tensor(
        [node_indices, hyperedge_ids], dtype=torch.long
    )
    return hyperedge_index, stats


def build_homology_hyperedge_index(
    homology_feature_path,
    ppi_neighbors,
    q,
    min_size=3,
):
    feature_matrix = np.load(homology_feature_path)
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"Expected 2-D homology features, got {feature_matrix.shape}"
        )

    unique_edges = []
    seen_edges = set()
    stats = []
    node_homology_scores = feature_matrix.sum(axis=1)
    q_active = node_homology_scores >= float(q)
    for column_id in range(feature_matrix.shape[1]):
        scores = feature_matrix[:, column_id]
        if not np.any(scores > 0):
            stats.append({
                "column": column_id,
                "kept": False,
                "reason": "all_zero",
            })
            continue

        threshold = None
        members = np.flatnonzero((scores > 0) & q_active).tolist()
        if len(members) < min_size:
            stats.append({
                "column": column_id,
                "kept": False,
                "reason": "size_below_min",
                "q": float(q),
                "threshold": threshold,
                "size": len(members),
            })
            continue

        member_set = set(members)
        unvisited = set(members)
        added_count = 0
        while unvisited:
            start = unvisited.pop()
            component = [start]
            stack = [start]
            while stack:
                node = stack.pop()
                connected = [
                    int(neighbor)
                    for neighbor in ppi_neighbors[node]
                    if int(neighbor) in member_set
                    and int(neighbor) in unvisited
                ]
                for neighbor in connected:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)

            if len(component) < min_size:
                continue
            component.sort()
            edge_key = tuple(component)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            unique_edges.append(component)
            added_count += 1

        stats.append({
            "column": column_id,
            "kept": added_count > 0,
            "reason": "ppi_connected_components",
            "q": float(q),
            "threshold": threshold,
            "size": len(members),
            "hyperedges_added": added_count,
        })

    node_indices = []
    hyperedge_ids = []
    for hyperedge_id, members in enumerate(unique_edges):
        node_indices.extend(members)
        hyperedge_ids.extend([hyperedge_id] * len(members))
    if not node_indices:
        raise ValueError("No valid homology hyperedges were constructed")

    return (
        torch.tensor([node_indices, hyperedge_ids], dtype=torch.long),
        stats,
        feature_matrix.shape,
    )


def build_ppi_multiscale_community_hyperedge_index(
    ppi_matrix,
    resolutions=(0.5, 1.0, 1.5, 2.0),
    min_community_size=3,
    seed=45,
):
    num_nodes = ppi_matrix.shape[0]
    undirected = (ppi_matrix != 0) | (ppi_matrix.T != 0)
    source, target = np.nonzero(np.triu(undirected, k=1))

    ppi_graph = nx.Graph()
    ppi_graph.add_nodes_from(range(num_nodes))
    ppi_graph.add_edges_from(zip(source.tolist(), target.tolist()))
    print("PPI undirected edges for community detection:", ppi_graph.number_of_edges())

    if not hasattr(nx.community, "louvain_communities"):
        raise RuntimeError(
            "This experiment requires networkx.community.louvain_communities. "
            "Please upgrade NetworkX to version 2.8 or newer."
        )

    unique_hyperedges = set()
    counts_by_resolution = {}
    for resolution in resolutions:
        communities = nx.community.louvain_communities(
            ppi_graph,
            weight=None,
            resolution=float(resolution),
            threshold=1e-7,
            seed=seed,
        )
        retained = 0
        for community in communities:
            if len(community) < min_community_size:
                continue
            unique_hyperedges.add(tuple(sorted(int(node) for node in community)))
            retained += 1
        counts_by_resolution[float(resolution)] = retained
        print(
            f"Louvain resolution={resolution}: "
            f"communities={len(communities)}, retained(size>=3)={retained}"
        )

    hyperedges = sorted(unique_hyperedges)
    if not hyperedges:
        raise ValueError("No PPI communities with size >= 3")

    node_indices = []
    hyperedge_ids = []
    for hyperedge_id, members in enumerate(hyperedges):
        node_indices.extend(members)
        hyperedge_ids.extend([hyperedge_id] * len(members))
    hyperedge_index = torch.tensor(
        [node_indices, hyperedge_ids], dtype=torch.long
    )
    return hyperedge_index, {
        "ppi_edges": ppi_graph.number_of_edges(),
        "resolutions": tuple(float(value) for value in resolutions),
        "counts_by_resolution": counts_by_resolution,
        "unique_communities": len(hyperedges),
    }


def attach_esm_to_ppi(data_gat_all, species, dataset, esm_path):
    A_static = None
    root = dataset_root(species, dataset)
    nodes_name = np.load(
        os.path.join(root, "gene_name.npy"), allow_pickle=True
    ).astype(str).tolist()

    if esm_path is not None:
        esm_data = torch.load(esm_path, map_location="cpu", weights_only=False)
        esm_x = esm_data["embeddings"].float()
        completed = esm_data.get("completed")
        if esm_x.ndim != 2 or esm_x.shape[1] != 1536:
            raise ValueError(f"Expected ESM3 matrix [N, 1536], got {tuple(esm_x.shape)}")
        if esm_x.shape[0] != len(nodes_name):
            raise ValueError(
                f"ESM3 node count {esm_x.shape[0]} != graph node count {len(nodes_name)}"
            )
        if completed is not None and not bool(torch.as_tensor(completed).all()):
            raise ValueError("ESM3 file contains unfinished nodes")
        if esm_data.get("failures"):
            raise ValueError(f"ESM3 file contains failures: {esm_data['failures']}")
        if "proteins" in esm_data and list(map(str, esm_data["proteins"])) != nodes_name:
            raise ValueError("ESM3 protein order does not match gene_name.npy")

        A_static = np.load(os.path.join(root, "SPIN.npy"))
        if A_static.shape != (len(nodes_name), len(nodes_name)):
            raise ValueError(f"Unexpected SPIN shape: {A_static.shape}")
        ppi_source, ppi_target = np.nonzero(A_static)
        ppi_edge_index = torch.tensor(
            np.vstack([ppi_source, ppi_target]), dtype=torch.long
        )
        (
            esm_community_hyperedge_index,
            esm_community_stats,
        ) = (
            build_ppi_multiscale_community_hyperedge_index(
                A_static,
                resolutions=(0.5, 1.0, 1.5, 2.0),
                min_community_size=3,
                seed=45,
            )
        )
        for graph in data_gat_all:
            graph.esm_x = esm_x
            graph.ppi_edge_index = ppi_edge_index
            graph.esm_community_hyperedge_index = esm_community_hyperedge_index
        print("ESM3 features:", tuple(esm_x.shape))
        print("PPI edge_index:", tuple(ppi_edge_index.shape))
        esm_community_sizes = torch.bincount(esm_community_hyperedge_index[1])
        print(
            "ESM3 multi-resolution PPI-community hyperedges:",
            int(esm_community_hyperedge_index[1].max().item()) + 1,
        )
        print(
            "ESM3 community size (min/mean/max):",
            int(esm_community_sizes.min()),
            f"{esm_community_sizes.float().mean().item():.2f}",
            int(esm_community_sizes.max()),
        )

    subcellular_csv = os.path.join(
        root, "refined_net_and_fea", "fea_sub_1024.csv"
    )
    if os.path.exists(subcellular_csv):
        ppi_matrix = A_static
        if ppi_matrix is None:
            ppi_matrix = np.load(os.path.join(root, "SPIN.npy"))
        if ppi_matrix.shape != (len(nodes_name), len(nodes_name)):
            raise ValueError(f"Unexpected SPIN shape: {ppi_matrix.shape}")
        ppi_neighbors = [
            np.flatnonzero(
                (ppi_matrix[node_id] != 0)
                | (ppi_matrix[:, node_id] != 0)
            )
            for node_id in range(ppi_matrix.shape[0])
        ]
        subcell_hyperedge_index, subcell_stats = build_subcell_hyperedge_index(
            subcellular_csv,
            ppi_neighbors,
            k=float(os.environ.get("SUBCELL_K", "1.0")),
            min_size=3,
            max_ratio=None,
            use_positive_only=True,
        )
        for graph in data_gat_all:
            graph.subcell_hyperedge_index = subcell_hyperedge_index
        kept_count = int(subcell_hyperedge_index[1].max().item()) + 1
        oversized_count = sum(
            item.get("reason") == "size_above_max_ratio"
            for item in subcell_stats
        )
        hyperedge_sizes = torch.bincount(subcell_hyperedge_index[1])
        print("Subcellular hyperedges:", kept_count)
        print("Subcellular threshold k:", float(os.environ.get("SUBCELL_K", "1.0")))
        print("Subcellular threshold comparison: score >= threshold")
        print("Subcellular construction: PPI connected components inside each S_j")
        print("Subcellular max_ratio: None (large hyperedges retained)")
        print("Subcellular oversized hyperedges removed:", oversized_count)
        print("Subcellular memberships:", subcell_hyperedge_index.shape[1])
        print(
            "Subcellular hyperedge size (min/mean/max):",
            int(hyperedge_sizes.min()),
            f"{hyperedge_sizes.float().mean().item():.2f}",
            int(hyperedge_sizes.max()),
        )

    if "ppi_neighbors" not in locals():
        ppi_matrix = A_static
        if ppi_matrix is None:
            ppi_matrix = np.load(os.path.join(root, "SPIN.npy"))
        ppi_neighbors = [
            np.flatnonzero(
                (ppi_matrix[node_id] != 0)
                | (ppi_matrix[:, node_id] != 0)
            )
            for node_id in range(ppi_matrix.shape[0])
        ]

    homology_feature_path = os.path.join(
        root, "refined_net_and_fea", "fea_orth.npy"
    )
    if os.path.exists(homology_feature_path):
        homology_q_text = os.environ.get("HOMOLOGY_Q")
        if homology_q_text in (None, "", "none"):
            raise ValueError("HOMOLOGY_Q is required for unified homology hypergraph construction")
        homology_q = float(homology_q_text)
        (
            homology_hyperedge_index,
            homology_stats,
            homology_shape,
        ) = build_homology_hyperedge_index(
            homology_feature_path,
            ppi_neighbors,
            q=homology_q,
            min_size=3,
        )
        for graph in data_gat_all:
            graph.homology_hyperedge_index = homology_hyperedge_index
        homology_sizes = torch.bincount(homology_hyperedge_index[1])
        print("Homology features:", homology_shape)
        print(
            "Homology hyperedges:",
            int(homology_hyperedge_index[1].max().item()) + 1,
        )
        print("Homology node-score q:", homology_q)
        print(
            "Homology construction: PPI connected components "
            "inside each homology score dimension"
        )
        print(
            "Homology hyperedge size (min/mean/max):",
            int(homology_sizes.min()),
            f"{homology_sizes.float().mean().item():.2f}",
            int(homology_sizes.max()),
        )
    return data_gat_all

def _load_base_node_data(species, dataset, use_subcellular):
    root = dataset_root(species, dataset)
    node_names = np.load(
        os.path.join(root, "gene_name.npy"), allow_pickle=True
    )
    num_nodes = len(node_names)

    feature_root = os.path.join(root, "refined_net_and_fea")
    gene_features = np.load(os.path.join(feature_root, "fea_gene.npy"))
    homology_features = np.load(os.path.join(feature_root, "fea_orth.npy"))
    feature_blocks = [gene_features]

    if use_subcellular:
        subcell_frame = pd.read_csv(
            os.path.join(feature_root, "fea_sub_1024.csv")
        )
        subcell_features = subcell_frame.iloc[:, 2:].to_numpy()
        feature_blocks.append(subcell_features)
        print("fea_sub:", subcell_features.shape)

    feature_blocks.append(homology_features)
    for name, values in (
        ("fea_gene", gene_features),
        ("fea_orthologous", homology_features),
    ):
        if values.ndim != 2 or values.shape[0] != num_nodes:
            raise ValueError(
                f"{name} shape {values.shape} is incompatible with "
                f"{num_nodes} nodes"
            )
        print(f"{name}:", values.shape)

    if use_subcellular and subcell_features.shape[0] != num_nodes:
        raise ValueError(
            f"fea_sub shape {subcell_features.shape} is incompatible with "
            f"{num_nodes} nodes"
        )

    labels = np.load(os.path.join(root, "labels.npy")).reshape(-1)
    if labels.shape[0] != num_nodes:
        raise ValueError(
            f"labels length {labels.shape[0]} != node count {num_nodes}"
        )

    features = torch.as_tensor(
        np.hstack(feature_blocks), dtype=torch.float
    )
    label_tensor = torch.as_tensor(labels, dtype=torch.float)
    return [Data(x=features, y=label_tensor)]


def _load_or_build_base_data(species, dataset, use_subcellular):
    cache_path = os.path.join(
        os.path.join(dataset_root(species, dataset), "data_and_model"),
        "base_node_features_hypergraph_v2.pt",
    )
    if os.path.exists(cache_path):
        print(f"{cache_path} The file exists.")
        return torch.load(cache_path, weights_only=False)

    print(f"{cache_path} The file does not exist.")
    data_gat_all = _load_base_node_data(
        species, dataset, use_subcellular=use_subcellular
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data_gat_all, cache_path)
    print("The base-node feature cache has been saved!")
    return data_gat_all


def load_gat_data(species, dataset, k_value, q_th, esm_path=None):
    del k_value, q_th
    data_gat_all = _load_or_build_base_data(
        species, dataset, use_subcellular=True
    )
    return attach_esm_to_ppi(
        data_gat_all, species, dataset, esm_path
    )


def load_gat_data_coli(species, dataset, k_value, q_th, esm_path=None):
    del k_value, q_th
    data_gat_all = _load_or_build_base_data(
        species, dataset, use_subcellular=False
    )
    return attach_esm_to_ppi(
        data_gat_all, species, dataset, esm_path
    )
