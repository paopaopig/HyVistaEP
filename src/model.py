import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv
from torch_geometric.data import Data
import numpy as np
import os


_DYNAMIC_HYPERGRAPH_CACHE = {}


def dataset_root(species, dataset):
    data_root = os.environ.get("HYVISTAEP_DATA_ROOT")
    if not data_root:
        raise RuntimeError(
            "HYVISTAEP_DATA_ROOT is not set. Pass --data-root to train.py/test.py."
        )
    return os.path.join(os.path.abspath(os.path.expanduser(data_root)), species,
                        "PPI", dataset)


def build_dynamic_snapshots(root):
    feature_root = os.path.join(root, "features")
    gene_expression = np.load(
        os.path.join(feature_root, "gene_expression.npy")
    )
    dynamic_activity = np.load(
        os.path.join(feature_root, "dynamic_activity.npy")
    )
    with np.load(os.path.join(root, "ppi_edges.npz"), allow_pickle=False) as data:
        ppi_edges = data["edge_index"]

    if gene_expression.ndim != 2 or dynamic_activity.ndim != 2:
        raise ValueError("Dynamic feature matrices must be two-dimensional")
    if gene_expression.shape[0] != dynamic_activity.shape[0]:
        raise ValueError("Dynamic feature matrices have inconsistent node counts")
    if ppi_edges.ndim != 2 or ppi_edges.shape[0] != 2:
        raise ValueError(f"Unexpected PPI edge shape: {ppi_edges.shape}")

    snapshot_x = torch.as_tensor(
        np.hstack([gene_expression, dynamic_activity]), dtype=torch.float
    )
    snapshots = []
    for timepoint in range(dynamic_activity.shape[1]):
        active = dynamic_activity[:, timepoint] != 0
        keep = active[ppi_edges[0]] & active[ppi_edges[1]]
        active_edges = ppi_edges[:, keep]
        directed_edges = np.concatenate(
            [active_edges, active_edges[::-1]], axis=1
        )
        snapshots.append(
            Data(
                x=snapshot_x,
                edge_index=torch.as_tensor(directed_edges, dtype=torch.long),
            )
        )
    return snapshots


def build_sliding_window_dynamic_hypergraph(
    snapshots, window_size=5, min_size=3, max_size=None
):
    num_snapshots = len(snapshots)
    if num_snapshots == 0:
        raise ValueError("No dynamic snapshots were loaded")
    num_nodes = snapshots[0].num_nodes
    starts = list(range(max(1, num_snapshots - window_size + 1)))
    hyperedge_frequency = {}

    for start in starts:
        end = min(num_snapshots, start + window_size)
        neighbor_counts = [dict() for _ in range(num_nodes)]
        for snapshot in snapshots[start:end]:
            edge_index = snapshot.edge_index.detach().cpu()
            snapshot_pairs = set()
            for source, target in zip(
                edge_index[0].tolist(), edge_index[1].tolist()
            ):
                if source == target:
                    continue
                pair = (
                    (source, target)
                    if source < target
                    else (target, source)
                )
                snapshot_pairs.add(pair)
            for left, right in snapshot_pairs:
                neighbor_counts[left][right] = (
                    neighbor_counts[left].get(right, 0) + 1
                )
                neighbor_counts[right][left] = (
                    neighbor_counts[right].get(left, 0) + 1
                )

        window_edges = set()
        for center, counts in enumerate(neighbor_counts):
            if len(counts) < min_size - 1:
                continue
            ordered = sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if max_size is not None:
                ordered = ordered[: max_size - 1]
            edge_key = tuple(
                sorted([center] + [neighbor for neighbor, _ in ordered])
            )
            if len(edge_key) >= min_size:
                window_edges.add(edge_key)
        for edge_key in window_edges:
            hyperedge_frequency[edge_key] = (
                hyperedge_frequency.get(edge_key, 0) + 1
            )

    if not hyperedge_frequency:
        raise ValueError("No dynamic hyperedges were constructed")

    node_ids, edge_ids, edge_weights = [], [], []
    for edge_id, (members, frequency) in enumerate(
        hyperedge_frequency.items()
    ):
        node_ids.extend(members)
        edge_ids.extend([edge_id] * len(members))
        edge_weights.append(frequency / len(starts))

    hyperedge_index = torch.tensor(
        [node_ids, edge_ids], dtype=torch.long
    )
    hyperedge_weight = torch.tensor(edge_weights, dtype=torch.float)

    feature_sum = None
    for snapshot in snapshots:
        snapshot_x = snapshot.x
        if not isinstance(snapshot_x, torch.Tensor):
            snapshot_x = torch.tensor(snapshot_x, dtype=torch.float)
        snapshot_x = snapshot_x.float()
        feature_sum = (
            snapshot_x.clone()
            if feature_sum is None
            else feature_sum + snapshot_x
        )
    dynamic_x = feature_sum / float(num_snapshots)
    return dynamic_x, hyperedge_index, hyperedge_weight, len(starts)


class DynamicHypergraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, drop_conv):
        super().__init__()
        self.conv1 = HypergraphConv(in_dim, hidden_dim1)
        self.conv2 = HypergraphConv(hidden_dim1, hidden_dim2)
        self.bn1 = nn.BatchNorm1d(hidden_dim1, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim2, track_running_stats=False)
        self.dropout = nn.Dropout(drop_conv)

    def forward(self, x, hyperedge_index, hyperedge_weight):
        x = self.conv1(
            x, hyperedge_index, hyperedge_weight=hyperedge_weight
        )
        x = self.dropout(F.relu(self.bn1(x)))
        x = self.conv2(
            x, hyperedge_index, hyperedge_weight=hyperedge_weight
        )
        x = self.dropout(F.relu(self.bn2(x)))
        return x


class DyHTSAGCN(nn.Module):
    def __init__(self, gene_layer_input, sub_layer_input, orth_layer_input, conv_hidden1, conv_hidden2, species, dataset, k_value, drop_mlp, drop_conv, mlp_hidden, classifier_hidden_1, classifier_hidden_2, drop_classifier, T, esm_layer_input=1536, use_subcellular=None, attention_heads=8):
        super(DyHTSAGCN, self).__init__()
        self.species = species
        self.gene_layer_input = gene_layer_input
        self.sub_layer_input = sub_layer_input
        self.orth_layer_input = orth_layer_input
        self.esm_layer_input = esm_layer_input
        self.use_subcellular = species != 'coli' if use_subcellular is None else use_subcellular
        self.use_fusion_attention = True
        branch_names = ["dynamic"]
        if self.use_subcellular:
            branch_names.append("subcellular")
        branch_names.extend(("homology", "raw_mlp", "esm3"))
        self.active_branch_names = tuple(branch_names)
        self.num_active_branches = len(self.active_branch_names)
        self.has_dynamic_branch = True
        self.has_subcellular_branch = self.use_subcellular
        self.has_homology_branch = True
        self.has_raw_mlp_branch = True
        self.has_esm3_branch = True
        self.use_esm_community_residual = True

        raw_feature_blocks = ["expression"]
        if self.use_subcellular:
            raw_feature_blocks.append("subcellular")
        raw_feature_blocks.append("homology")
        self.raw_feature_blocks = tuple(raw_feature_blocks)
        raw_feature_dims = {
            "expression": gene_layer_input,
            "subcellular": sub_layer_input,
            "homology": orth_layer_input,
        }
        self.raw_feature_input_dim = sum(
            raw_feature_dims[name] for name in self.raw_feature_blocks
        )
        if self.has_dynamic_branch:
            self.dynamic_hypergraph_encoder = DynamicHypergraphEncoder(
                in_dim=gene_layer_input * 2,
                hidden_dim1=conv_hidden1,
                hidden_dim2=conv_hidden2,
                drop_conv=drop_conv,
            )
        self.T = T
        self.dropout_mlp = nn.Dropout(drop_mlp)
        self.dropout_cov = nn.Dropout(drop_conv)
        self.classifier_drop = nn.Dropout(drop_classifier)
        self._shape_debug_printed = False
        if self.has_homology_branch:
            self.conv_4th_1 = HypergraphConv(orth_layer_input, conv_hidden1)
            self.conv_4th_2 = HypergraphConv(conv_hidden1, conv_hidden2)
            self.bn_4th_1 = nn.BatchNorm1d(conv_hidden1, track_running_stats=False)
            self.bn_4th_2 = nn.BatchNorm1d(conv_hidden2, track_running_stats=False)
        if self.has_esm3_branch:
            self.esm_gcn_1 = GCNConv(esm_layer_input, conv_hidden1)
            self.esm_gcn_2 = GCNConv(conv_hidden1, conv_hidden2)
            self.bn_esm_gcn_1 = nn.BatchNorm1d(conv_hidden1, track_running_stats=False)
            self.bn_esm_gcn_2 = nn.BatchNorm1d(conv_hidden2, track_running_stats=False)
            if self.use_esm_community_residual:
                self.esm_hyperconv_1 = HypergraphConv(esm_layer_input, conv_hidden1)
                self.esm_hyperconv_2 = HypergraphConv(conv_hidden1, conv_hidden2)
                self.bn_esm_hyper_1 = nn.BatchNorm1d(conv_hidden1, track_running_stats=False)
                self.bn_esm_hyper_2 = nn.BatchNorm1d(conv_hidden2, track_running_stats=False)
                self.esm_hyper_residual_logit = nn.Parameter(torch.tensor(-1.0))
        self.mlp_hidden = mlp_hidden
        self.classifier_hidden_1 = classifier_hidden_1
        self.classifier_hidden_2 = classifier_hidden_2
        if self.has_raw_mlp_branch:
            self.bn_mlp_1 = nn.BatchNorm1d(self.mlp_hidden, track_running_stats=False)
            self.bn_mlp_2 = nn.BatchNorm1d(conv_hidden2, track_running_stats=False)
        self.bn_cla_1 = nn.BatchNorm1d(self.classifier_hidden_1, track_running_stats=False)
        self.bn_cla_2 = nn.BatchNorm1d(self.classifier_hidden_2, track_running_stats=False)

        if self.has_subcellular_branch:
            self.subcell_hidden = conv_hidden1
            self.subcell_hyperconv_1 = HypergraphConv(
                sub_layer_input, self.subcell_hidden
            )
            self.subcell_hyperconv_2 = HypergraphConv(
                self.subcell_hidden, conv_hidden2
            )
            self.bn_2nd_1 = nn.BatchNorm1d(
                self.subcell_hidden, track_running_stats=False
            )
            self.bn_2nd_2 = nn.BatchNorm1d(conv_hidden2, track_running_stats=False)
        if self.has_raw_mlp_branch:
            self.mlp_for_fea = nn.Sequential(
                nn.Linear(self.raw_feature_input_dim, self.mlp_hidden),
                self.bn_mlp_1,
                nn.ReLU(),
                self.dropout_mlp,
                nn.Linear(self.mlp_hidden, conv_hidden2),
                self.bn_mlp_2,
                nn.ReLU(),
                self.dropout_mlp
            )
        if self.use_fusion_attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=conv_hidden2, num_heads=attention_heads, batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(conv_hidden2)

        self.classifier_1 = nn.Sequential(
            nn.Linear(self.num_active_branches * conv_hidden2, self.classifier_hidden_1),
            self.bn_cla_1,
            nn.ReLU(),
            self.classifier_drop,
            nn.Linear(self.classifier_hidden_1, self.classifier_hidden_2),
            self.bn_cla_2,
            nn.ReLU(),
            self.classifier_drop,
            nn.Linear(self.classifier_hidden_2, 1),
        )

        if self.has_dynamic_branch:
            root = dataset_root(species, dataset)
            dynamic_cache_key = (root, self.T, 3, None)
            if dynamic_cache_key not in _DYNAMIC_HYPERGRAPH_CACHE:
                mapped_graphs = build_dynamic_snapshots(root)
                _DYNAMIC_HYPERGRAPH_CACHE[dynamic_cache_key] = (
                    build_sliding_window_dynamic_hypergraph(
                        mapped_graphs,
                        window_size=self.T,
                        min_size=3,
                        max_size=None,
                    )
                )
            (
                self.dynamic_x,
                self.dynamic_hyperedge_index,
                self.dynamic_hyperedge_weight,
                self.dynamic_window_count,
            ) = _DYNAMIC_HYPERGRAPH_CACHE[dynamic_cache_key]

    def extract_embeddings(self, data):
        return self._extract_full_embeddings(data)

    def _extract_full_embeddings(self, data):
        device = next(self.parameters()).device
        show_shapes = not self._shape_debug_printed
        dynamic_x = self.dynamic_x.to(device=device, dtype=torch.float)
        dynamic_hyperedge_index = self.dynamic_hyperedge_index.to(device)
        dynamic_hyperedge_weight = self.dynamic_hyperedge_weight.to(
            device=device, dtype=torch.float
        )
        out_dysat = self.dynamic_hypergraph_encoder(
            dynamic_x,
            dynamic_hyperedge_index,
            dynamic_hyperedge_weight,
        )

        x = data[0].x
        if (
            not hasattr(data[0], 'esm_x')
            or not hasattr(data[0], 'esm_community_hyperedge_index')
            or not hasattr(data[0], 'ppi_edge_index')
        ):
            raise ValueError(
                "Graph data must contain esm_x and "
                "esm_community_hyperedge_index and ppi_edge_index"
            )
        esm_x = data[0].esm_x.to(device=device, dtype=torch.float)
        esm_community_hyperedge_index = (
            data[0].esm_community_hyperedge_index.to(device)
        )
        ppi_edge_index = data[0].ppi_edge_index.to(device)

        esm_gcn = self.esm_gcn_1(esm_x, ppi_edge_index)
        esm_gcn_hidden1 = esm_gcn
        esm_gcn = self.bn_esm_gcn_1(esm_gcn)
        esm_gcn = self.dropout_cov(F.relu(esm_gcn))
        esm_gcn = self.esm_gcn_2(esm_gcn, ppi_edge_index)
        esm_gcn = self.bn_esm_gcn_2(esm_gcn)
        esm_gcn = self.dropout_cov(F.relu(esm_gcn))

        esm_hyper = self.esm_hyperconv_1(
            esm_x, esm_community_hyperedge_index
        )
        esm_hyper_hidden1 = esm_hyper
        esm_hyper = self.bn_esm_hyper_1(esm_hyper)
        esm_hyper = self.dropout_cov(F.relu(esm_hyper))
        esm_hyper = self.esm_hyperconv_2(
            esm_hyper, esm_community_hyperedge_index
        )
        esm_hyper = self.bn_esm_hyper_2(esm_hyper)
        esm_hyper = self.dropout_cov(F.relu(esm_hyper))
        esm_hyper_residual_weight = torch.sigmoid(
            self.esm_hyper_residual_logit
        )
        if self.use_esm_community_residual:
            out_esm = esm_gcn + esm_hyper_residual_weight * esm_hyper
        else:
            out_esm = esm_gcn

        if not hasattr(data[0], 'homology_hyperedge_index'):
            raise ValueError("Graph data must contain homology_hyperedge_index")
        homology_hyperedge_index = data[0].homology_hyperedge_index.to(device)

        if self.use_subcellular:
            if not hasattr(data[0], 'subcell_hyperedge_index'):
                raise ValueError("Graph data must contain subcell_hyperedge_index")
            subcell_hyperedge_index = data[0].subcell_hyperedge_index.to(device)
            subcell_x = x[:, self.gene_layer_input:self.gene_layer_input + self.sub_layer_input]
            out2 = self.subcell_hyperconv_1(subcell_x, subcell_hyperedge_index)
            subcell_hidden1 = out2
            out2 = self.bn_2nd_1(out2)
            out2 = self.dropout_cov(F.relu(out2))
            out2 = self.subcell_hyperconv_2(out2, subcell_hyperedge_index)
            out2 = self.bn_2nd_2(out2)
            out2 = self.dropout_cov(F.relu(out2))

            orth_x = x[
                :,
                self.gene_layer_input + self.sub_layer_input:
                self.gene_layer_input + self.sub_layer_input + self.orth_layer_input,
            ]
            out4 = self.conv_4th_1(orth_x, homology_hyperedge_index)
            orth_hidden1 = out4
            out4 = self.bn_4th_1(out4)
            out4 = self.dropout_cov(F.relu(out4))
            out4 = self.conv_4th_2(out4, homology_hyperedge_index)
            out4 = self.bn_4th_2(out4)
            out4 = self.dropout_cov(F.relu(out4))
        else:
            orth_start = self.gene_layer_input
            orth_stop = orth_start + self.orth_layer_input
            orth_x = x[:, orth_start:orth_stop]
            out4 = self.conv_4th_1(orth_x, homology_hyperedge_index)
            orth_hidden1 = out4
            out4 = self.bn_4th_1(out4)
            out4 = self.dropout_cov(F.relu(out4))
            out4 = self.conv_4th_2(out4, homology_hyperedge_index)
            out4 = self.bn_4th_2(out4)
            out4 = self.dropout_cov(F.relu(out4))

        feature_slices = {
            "expression": x[:, :self.gene_layer_input],
        }
        if self.use_subcellular:
            subcell_start = self.gene_layer_input
            subcell_stop = subcell_start + self.sub_layer_input
            orth_start = subcell_stop
            feature_slices["subcellular"] = x[:, subcell_start:subcell_stop]
        else:
            orth_start = self.gene_layer_input
        orth_stop = orth_start + self.orth_layer_input
        feature_slices["homology"] = x[:, orth_start:orth_stop]
        raw_mlp_x = torch.cat(
            [feature_slices[name] for name in self.raw_feature_blocks], dim=1
        )
        fea_concat = self.mlp_for_fea(raw_mlp_x)
        branch_outputs = {
            "dynamic": out_dysat,
            "homology": out4,
            "raw_mlp": fea_concat,
            "esm3": out_esm,
        }
        if self.use_subcellular:
            branch_outputs["subcellular"] = out2
        branch_tokens = torch.stack(
            [branch_outputs[name] for name in self.active_branch_names], dim=1
        )
        if self.use_fusion_attention:
            attended_tokens, branch_attention = self.attention_layer(
                branch_tokens,
                branch_tokens,
                branch_tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            attended_tokens = self.fusion_norm(branch_tokens + attended_tokens)
            self.last_branch_attention = branch_attention.detach()
        else:
            attended_tokens = branch_tokens
            branch_attention = None
            self.last_branch_attention = None
        out_all_mrpin = attended_tokens.flatten(start_dim=1)
        if show_shapes:
            print("\n=== Model feature shapes (printed once) ===")
            print("Static input x:", tuple(x.shape))
            print("Dynamic hypergraph input:", tuple(dynamic_x.shape))
            print("Dynamic sliding windows:", self.dynamic_window_count)
            print("Dynamic hyperedge_index:", tuple(dynamic_hyperedge_index.shape))
            print(
                "Dynamic hyperedge_weight (min/mean/max):",
                f"{dynamic_hyperedge_weight.min().item():.4f}",
                f"{dynamic_hyperedge_weight.mean().item():.4f}",
                f"{dynamic_hyperedge_weight.max().item():.4f}",
            )
            print("Dynamic branch output:", tuple(out_dysat.shape))
            if self.use_subcellular:
                print("Subcellular input:", tuple(subcell_x.shape))
                print("Subcellular hyperedge_index:", tuple(subcell_hyperedge_index.shape))
                print("Subcellular HypergraphConv-1:", tuple(subcell_hidden1.shape))
                print("Subcellular HypergraphConv-2:", tuple(out2.shape))
            print("Homology input:", tuple(orth_x.shape))
            print("Homology hyperedge_index:", tuple(homology_hyperedge_index.shape))
            print("Homology HypergraphConv-1:", tuple(orth_hidden1.shape))
            print("Homology HypergraphConv-2:", tuple(out4.shape))
            print("Raw-feature MLP output:", tuple(fea_concat.shape))
            print("ESM3 input:", tuple(esm_x.shape))
            print(
                "ESM3 multi-resolution community hyperedge_index:",
                tuple(esm_community_hyperedge_index.shape),
            )
            print("ESM3 PPI-GCN-1 (main):", tuple(esm_gcn_hidden1.shape))
            print("ESM3 PPI-GCN-2 (main):", tuple(esm_gcn.shape))
            print("ESM3 community HypergraphConv-1 (residual):", tuple(esm_hyper_hidden1.shape))
            print("ESM3 community HypergraphConv-2 (residual):", tuple(esm_hyper.shape))
            if self.use_esm_community_residual:
                print(
                    "ESM3 hypergraph residual weight:",
                    f"{esm_hyper_residual_weight.detach().item():.4f}",
                )
            else:
                print("ESM3 hypergraph residual: disabled")
            print("ESM3 fused branch output:", tuple(out_esm.shape))
            print("Active branches:", self.active_branch_names)
            if "raw_mlp" in self.active_branch_names:
                print("Raw-feature MLP blocks:", self.raw_feature_blocks)
                print("Raw-feature MLP input:", tuple(raw_mlp_x.shape))
            print("Fusion branch tokens:", tuple(branch_tokens.shape))
            if branch_attention is None:
                print("Fusion attention: disabled")
            else:
                print("Fusion attention weights:", tuple(branch_attention.shape))
            print("Fused feature:", tuple(out_all_mrpin.shape))
            print("===========================================\n")
            self._shape_debug_printed = True
        return out_all_mrpin

    def forward(self, data):
        out = self.extract_embeddings(data)
        return self.classifier_1(out).squeeze(-1)
