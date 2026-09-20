import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from data_load import load_gat_data_coli, load_gat_data
from model import DyHTSAGCN
import pickle
import json
import argparse
from config import get_config

def set_reproducibility(run_seed, deterministic=True):
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


seed = 45
set_reproducibility(seed, deterministic=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_best_threshold(y_true, y_probs, max_thresh=0.9):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1 = f1[:-1]  # thresholds has one fewer element than precision/recall
    valid = thresholds <= max_thresh
    if not valid.any():
        print("All thresholds are too large. Using the default value 0.5.")
        return 0.5
    best_idx = np.argmax(f1[valid])
    best_thresh = thresholds[valid][best_idx]
    best_f1 = f1[valid][best_idx]
    print(f"Best threshold (F1): {best_thresh:.4f}, F1: {best_f1:.4f}")
    return best_thresh

def train_model(model, data, criterion, optimizer, num_epochs, val_mask, scheduler):
    best_val_auc = 0.0
    best_model_dict = None
    patience = 5
    patience_counter = 0
    best_val_threshold = 0.5

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        h = model.extract_embeddings(data)  # [N, hidden_dim]

        out = model.classifier_1(h).squeeze(dim=1)
        train_outputs = out[data[0].train_mask]
        train_labels = data[0].y[data[0].train_mask].float()
        loss = criterion(train_outputs, train_labels)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h_val = model.extract_embeddings(data)
            val_outputs = model.classifier_1(h_val).squeeze(dim=1)
            val_labels = data[0].y[val_mask].cpu().numpy()
            val_probs = torch.sigmoid(val_outputs[val_mask]).cpu().numpy()
            val_auc = roc_auc_score(val_labels, val_probs)

        print(f"\n Epoch {epoch + 1}")
        print(f"   Training Loss     : {loss.item():.4f}")
        print(f"   Val AUC           : {val_auc:.4f}")

        if val_auc > best_val_auc:
            print("The current model is the best and is being saved...")
            best_val_auc = val_auc
            patience_counter = 0
            best_model_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_val_threshold = find_best_threshold(val_labels, val_probs)
        else:
            print(f"No boost, EarlyStopping count: {patience_counter + 1}/{patience}")
            patience_counter += 1
        scheduler.step(val_auc)
        if patience_counter >= patience:
            print("Early Stopping!")
            break
    return best_model_dict, best_val_threshold

def evaluate_model(model, data, best_threshold=0.5):
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_outputs = out[data[0].test_mask]
        test_labels = data[0].y[data[0].test_mask].float()
        test_probs = torch.sigmoid(test_outputs).cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
        test_preds = (test_probs >= best_threshold).astype(int)

        auc = roc_auc_score(test_labels_np, test_probs)
        acc = accuracy_score(test_labels_np, test_preds)
        aupr = average_precision_score(test_labels_np, test_probs)

        tn, fp, fn, tp = confusion_matrix(test_labels_np, test_preds).ravel()
        sn = tp / (tp + fn) if tp + fn > 0 else 0
        sp = tn / (tn + fp) if tn + fp > 0 else 0
        ppv = tp / (tp + fp) if tp + fp > 0 else 0
        npv = tn / (tn + fn) if tn + fn > 0 else 0
        f1 = f1_score(test_labels_np, test_preds)
        precision = precision_score(test_labels_np, test_preds)
        recall = recall_score(test_labels_np, test_preds)
        fpr, tpr, _ = roc_curve(test_labels_np, test_probs)
        prec_curve, rec_curve, _ = precision_recall_curve(test_labels_np, test_probs)

        return auc, acc, aupr, sn, sp, ppv, npv, f1, precision, recall, test_labels_np, test_probs, fpr, tpr, rec_curve, prec_curve

def train_and_test(species, k_folds=10):
    cfg = get_config(species)
    if "UNIFIED_T" in os.environ:
        cfg["T"] = int(os.environ["UNIFIED_T"])
    if "UNIFIED_HOMOLOGY_Q" in os.environ:
        cfg["homology_q"] = float(os.environ["UNIFIED_HOMOLOGY_Q"])
    subcell_k_env = os.environ.get("UNIFIED_SUBCELL_K")
    if cfg["use_subcellular"] and subcell_k_env not in (None, "", "None"):
        cfg["subcell_k"] = float(subcell_k_env)
    for env_name, cfg_name in (
        ("UNIFIED_LR", "lr"),
        ("UNIFIED_WEIGHT_DECAY", "weight_decay"),
        ("UNIFIED_DROP_MLP", "drop_mlp"),
        ("UNIFIED_DROP_CONV", "drop_conv"),
        ("UNIFIED_DROP_CLASSIFIER", "drop_classifier"),
    ):
        if env_name in os.environ:
            cfg[cfg_name] = float(os.environ[env_name])
    if "UNIFIED_MLP_HIDDEN" in os.environ:
        cfg["mlp_hidden"] = int(os.environ["UNIFIED_MLP_HIDDEN"])
    if "UNIFIED_EPOCHS" in os.environ:
        cfg["epochs"] = int(os.environ["UNIFIED_EPOCHS"])
    dataset = cfg["dataset"]
    k_value, q_th = cfg["k_value"], cfg["q_th"]
    epochs_num = cfg["epochs"]
    lr, weight_decay = cfg["lr"], cfg["weight_decay"]
    drop_mlp, drop_conv = cfg["drop_mlp"], cfg["drop_conv"]
    drop_classifier = cfg["drop_classifier"]
    mlp_hidden = cfg["mlp_hidden"]
    classifier_hidden_1 = cfg["classifier_hidden1"]
    classifier_hidden_2 = cfg["classifier_hidden2"]
    T, esm_path = cfg["T"], cfg["esm_path"]
    if cfg["subcell_k"] is not None:
        os.environ["SUBCELL_K"] = str(cfg["subcell_k"])
    os.environ["HOMOLOGY_Q"] = str(cfg["homology_q"])
    output_dir = os.environ["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    loader = load_gat_data_coli if species == "coli" else load_gat_data
    graph_data = loader(
        species=species,
        dataset=dataset,
        k_value=k_value,
        q_th=q_th,
        esm_path=esm_path,
    )
    gene_layer_input = cfg["gene_dim"]
    sub_layer_input = cfg["subcell_dim"]
    orth_layer_input = cfg["homology_dim"]
    esm_layer_input = cfg["esm_dim"]
    conv_hidden1 = cfg["conv_hidden1"]
    conv_hidden2 = cfg["conv_hidden2"]
    labels = graph_data[0].y.cpu().numpy()
    indices = np.arange(len(labels))
    train_idx_all, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)
    train_labels_all = labels[train_idx_all]

    for g in graph_data:
        g.train_mask = torch.zeros(len(labels), dtype=torch.bool)
        g.test_mask = torch.zeros(len(labels), dtype=torch.bool)
        g.test_mask[test_idx] = True

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_probs = []
    all_labels = []
    fpr_list, tpr_list, rec_list, prec_list = [], [], [], []
    aucs = []
    accs = []
    auprs = []
    sns = []
    sps = []
    ppvs = []
    npvs = []
    f1s = []
    precisions = []
    recalls = []
    best_thresholds = []
    positive_weights = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_idx_all, train_labels_all)):
        fold_seed = seed + fold
        set_reproducibility(fold_seed, deterministic=True)
        print(f"\nFold {fold + 1}/{k_folds}: start training")
        print(f"Fold seed: {fold_seed}")
        train_ids = train_idx_all[train_idx]
        val_ids = train_idx_all[val_idx]

        for g in graph_data:
            g.train_mask[:] = False
            g.train_mask[train_ids] = True

        val_mask = torch.zeros(len(labels), dtype=torch.bool)
        val_mask[val_ids] = True

        fold_train_labels = labels[train_ids]
        positive_count = int((fold_train_labels == 1).sum())
        negative_count = int((fold_train_labels == 0).sum())
        weight_positive = (
            negative_count / positive_count * cfg["pos_weight_scale"]
        )
        positive_weights.append(weight_positive)
        print(
            f"Fold {fold + 1} pos_weight: negatives/positives "
            f"* scale = {negative_count}/{positive_count} "
            f"* {cfg['pos_weight_scale']} = {weight_positive:.6f}"
        )

        model = DyHTSAGCN(
            gene_layer_input, sub_layer_input, orth_layer_input,
            conv_hidden1, conv_hidden2, species, dataset, k_value,
            drop_mlp, drop_conv, mlp_hidden, classifier_hidden_1,
            classifier_hidden_2, drop_classifier, T,
            esm_layer_input=esm_layer_input,
            use_subcellular=cfg["use_subcellular"],
            attention_heads=cfg["attention_heads"],
        ).to(device)

        for g in graph_data:
            g.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_positive]).to(device))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        best_model_dict, best_threshold = train_model(
            model, graph_data, criterion, optimizer, epochs_num,
            val_mask.to(device), scheduler,
        )
        best_thresholds.append(best_threshold)

        if best_model_dict is not None:
            model.load_state_dict(best_model_dict)
            if model.use_esm_community_residual:
                learned_esm_residual_weight = torch.sigmoid(
                    model.esm_hyper_residual_logit.detach()
                ).item()
                print(
                    f"Fold {fold + 1} best ESM3 community residual weight: "
                    f"{learned_esm_residual_weight:.4f}"
                )
            else:
                print(f"Fold {fold + 1} ESM3 community residual: disabled")

        auc, acc, aupr, sn, sp, ppv, npv, f1, precision, recall, lbls, probs, fpr, tpr, rec, prec = evaluate_model(model, graph_data, best_threshold)
        print(f"Fold {fold + 1} - AUC: {auc:.4f}, ACC: {acc:.4f}, AUPR: {aupr:.4f}, SN: {sn:.4f}, SP: {sp:.4f}, PPV: {ppv:.4f}, NPV: {npv:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "fold": fold + 1,
            "seed": seed,
            "fold_seed": fold_seed,
            "best_threshold": float(best_threshold),
            "train_indices": np.asarray(train_ids),
            "validation_indices": np.asarray(val_ids),
            "test_indices": np.asarray(test_idx),
            "config": {
                "species": species, "q_th": q_th, "T": T,
                "subcell_k": cfg["subcell_k"],
                "homology_q": cfg["homology_q"],
                "positive_weight_scale": cfg["pos_weight_scale"],
            },
        }
        torch.save(
            checkpoint,
            os.path.join(output_dir, f"best_model_fold_{fold + 1}.pth"),
        )

        aucs.append(auc)
        accs.append(acc)
        auprs.append(aupr)
        sns.append(sn)
        sps.append(sp)
        ppvs.append(ppv)
        npvs.append(npv)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        all_probs.extend(probs)
        all_labels.extend(lbls)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        rec_list.append(rec)
        prec_list.append(prec)
    data_to_save = {
        "all_probs": all_probs,
        "all_labels": all_labels,
        "fpr_list": fpr_list,
        "tpr_list": tpr_list,
        "rec_list": rec_list,
        "prec_list": prec_list,
        "best_thresholds": best_thresholds,
        "fold_metrics": {
            "AUC": list(map(float, aucs)),
            "ACC": list(map(float, accs)),
            "AUPR": list(map(float, auprs)),
            "SN": list(map(float, sns)),
            "SP": list(map(float, sps)),
            "PPV": list(map(float, ppvs)),
            "NPV": list(map(float, npvs)),
            "F1": list(map(float, f1s)),
            "Precision": list(map(float, precisions)),
            "Recall": list(map(float, recalls)),
        },
    }
    print("\n Average indicator: ")
    def report(name, values):
        print(f"{name}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
    print(f"Species: {species}; dataset: {dataset}")
    print(f"***k_value = {k_value}, q_th = {q_th}***")
    report("AUC", aucs)
    report("ACC", accs)
    report("AUPR", auprs)
    report("SN", sns)
    report("SP", sps)
    report("PPV", ppvs)
    report("NPV", npvs)
    report("F1", f1s)
    report("Precision", precisions)
    report("Recall", recalls)
    print(f"***mlp_hidden:{mlp_hidden},classifier_hidden_1:{classifier_hidden_1}, classifier_hidden_2:{classifier_hidden_2},drop_classifier:{drop_classifier}, T={T}***")
    print(f"***learning_rate:{lr}, weight_decay:{weight_decay}, drop_mlp:{drop_mlp}, drop_conv:{drop_conv}, conv_hidden1:{conv_hidden1}, conv_hidden2:{conv_hidden2}***")
    print(
        f"***positive_weight_scale:{cfg['pos_weight_scale']},"
        f"mean/min/max:{np.mean(positive_weights):.6f}/"
        f"{np.min(positive_weights):.6f}/{np.max(positive_weights):.6f}***"
    )
    summary = {
        "species": species,
        "seed": seed,
        "fold_seeds": [seed + fold for fold in range(k_folds)],
        "deterministic": True,
        "epochs": epochs_num,
        "folds": k_folds,
        "T": T,
        "q_th": q_th,
        "subcell_k": cfg["subcell_k"],
        "homology_q": cfg["homology_q"],
        "lr": lr,
        "weight_decay": weight_decay,
        "drop_mlp": drop_mlp,
        "drop_conv": drop_conv,
        "drop_classifier": drop_classifier,
        "mlp_hidden": mlp_hidden,
        "conv_hidden1": conv_hidden1,
        "conv_hidden2": conv_hidden2,
        "classifier_hidden1": classifier_hidden_1,
        "classifier_hidden2": classifier_hidden_2,
        "attention_heads": cfg["attention_heads"],
        "positive_weight_scale": cfg["pos_weight_scale"],
        "mean_test_auc": float(np.mean(aucs)),
        "mean_test_aupr": float(np.mean(auprs)),
        "fold_test_auc": list(map(float, aucs)),
        "fold_test_aupr": list(map(float, auprs)),
        "mean_test_acc": float(np.mean(accs)),
        "mean_test_sn": float(np.mean(sns)),
        "mean_test_sp": float(np.mean(sps)),
        "mean_test_ppv": float(np.mean(ppvs)),
        "mean_test_npv": float(np.mean(npvs)),
        "mean_test_f1": float(np.mean(f1s)),
        "mean_test_precision": float(np.mean(precisions)),
        "mean_test_recall": float(np.mean(recalls)),
        "fold_test_acc": list(map(float, accs)),
        "fold_test_sn": list(map(float, sns)),
        "fold_test_sp": list(map(float, sps)),
        "fold_test_ppv": list(map(float, ppvs)),
        "fold_test_npv": list(map(float, npvs)),
        "fold_test_f1": list(map(float, f1s)),
        "fold_test_precision": list(map(float, precisions)),
        "fold_test_recall": list(map(float, recalls)),
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "evaluation_results.pkl"), "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Saved unified {species} checkpoints to:", output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True,
                        choices=("human", "yeast", "coli", "melanogaster"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Root containing <species>/PPI/BIOGRID.",
    )
    parser.add_argument("--T", type=int)
    parser.add_argument("--homology-q", type=float)
    parser.add_argument("--subcell-k", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--drop-mlp", type=float)
    parser.add_argument("--drop-conv", type=float)
    parser.add_argument("--drop-classifier", type=float)
    parser.add_argument("--mlp-hidden", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    seed = args.seed
    set_reproducibility(seed, deterministic=True)
    os.environ["OUTPUT_DIR"] = args.output_dir
    os.environ["HYVISTAEP_DATA_ROOT"] = str(
        Path(args.data_root).expanduser().resolve()
    )
    cfg = get_config(args.species)
    if args.T is not None:
        cfg["T"] = args.T
    if args.homology_q is not None:
        cfg["homology_q"] = args.homology_q
    if args.subcell_k is not None:
        cfg["subcell_k"] = args.subcell_k
    for arg_name, cfg_name in (
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
        ("drop_mlp", "drop_mlp"),
        ("drop_conv", "drop_conv"),
        ("drop_classifier", "drop_classifier"),
        ("mlp_hidden", "mlp_hidden"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg[cfg_name] = value
    os.environ["UNIFIED_T"] = str(cfg["T"])
    os.environ["UNIFIED_HOMOLOGY_Q"] = str(cfg["homology_q"])
    if cfg["use_subcellular"] and cfg["subcell_k"] is not None:
        os.environ["UNIFIED_SUBCELL_K"] = str(cfg["subcell_k"])
    else:
        os.environ.pop("UNIFIED_SUBCELL_K", None)
    os.environ["UNIFIED_LR"] = str(cfg["lr"])
    os.environ["UNIFIED_WEIGHT_DECAY"] = str(cfg["weight_decay"])
    os.environ["UNIFIED_DROP_MLP"] = str(cfg["drop_mlp"])
    os.environ["UNIFIED_DROP_CONV"] = str(cfg["drop_conv"])
    os.environ["UNIFIED_DROP_CLASSIFIER"] = str(cfg["drop_classifier"])
    os.environ["UNIFIED_MLP_HIDDEN"] = str(cfg["mlp_hidden"])
    if args.epochs is not None:
        os.environ["UNIFIED_EPOCHS"] = str(args.epochs)
    train_and_test(args.species, k_folds=args.folds)
