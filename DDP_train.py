import argparse
import os
import csv
import json
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict
from torch import nn
from torch.nn import functional as F

from model import AnnotModel
from data import ProteinDataset, build_train_dataloader, quantize_label

LABEL_HIERARCHY = {
    'pvp': [
        'tail',
        'collar',
        'head-tail_joining',
        'portal',
        'capsid',
    ],
    'tail': [
        'major_tail',
        'minor_tail',
        'tail_appendage',
        'tail_sheath',
        'baseplate',
    ],
    'capsid': [
        'major_capsid',
        'minor_capsid',
    ],
    'DNA-associated': [
        'helicase',
        'nuclease',
        'terminase',
        'integration',
        'DNA_polymerase',
        'annealing',
        'primase',
        'replication_initiation',
    ],
    'transcriptional_regulator': [
        'transcriptional_activator',
        'transcriptional_repressor',
    ],
    'lysis': [
        'endolysin',
        'spanin',
        'lysis_inhibitor',
        'holin',
    ],
    'defense_systems': [
        'toxin',
        'anti-restriction',
        'sir2',
        'crispr',
        'super_infection',
    ],
}

HIERARCHY_POLICY = "suppress_child_without_parent"


def enforce_hierarchical_child_consistency(
    y_pred: np.ndarray,
    label_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    Suppress a positive child prediction whenever its parent is negative.

    The rule is applied until stable so a positive descendant is retained only
    when every mapped ancestor in the hierarchy is also positive.
    """
    processed_pred = y_pred.copy()

    changed = True
    while changed:
        changed = False
        for parent_label, child_labels in LABEL_HIERARCHY.items():
            parent_idx = label_to_idx.get(parent_label)
            if parent_idx is None:
                continue

            for child_label in child_labels:
                child_idx = label_to_idx.get(child_label)
                if child_idx is None:
                    continue

                child_before = processed_pred[:, child_idx].copy()
                processed_pred[:, child_idx] = np.minimum(
                    processed_pred[:, child_idx],
                    processed_pred[:, parent_idx],
                )
                if not np.array_equal(child_before, processed_pred[:, child_idx]):
                    changed = True

    return processed_pred


def get_warmup_linear_decay_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """
    Creates a learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: The optimizer
        warmup_steps: Number of steps for warmup
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of peak lr (default 0.01 = 1/100)
    
    Returns:
        A lambda LR scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay: from 1 to min_lr_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_context_columns_from_df(df: pd.DataFrame, include_labels: bool = True):
    left_seqs_cols = []
    right_seqs_cols = []
    left_labels_cols = []
    right_labels_cols = []

    for i in range(3):
        l_seq = df.get(f"left_seq_{i}")
        if l_seq is None:
            l_seq = pd.Series([''] * len(df))
        left_seqs_cols.append(l_seq.fillna('').tolist())

        r_seq = df.get(f"right_seq_{i}")
        if r_seq is None:
            r_seq = pd.Series([''] * len(df))
        right_seqs_cols.append(r_seq.fillna('').tolist())

        if include_labels:
            l_lbl = df.get(f"left_label_{i}")
            if l_lbl is None:
                l_lbl = df.get(f"left_consistent_label_{i}")
            if l_lbl is None:
                l_lbl = pd.Series([[] for _ in range(len(df))])
            l_lbl = l_lbl.apply(lambda x: x if isinstance(x, list) else [])
            left_labels_cols.append(l_lbl.tolist())

            r_lbl = df.get(f"right_label_{i}")
            if r_lbl is None:
                r_lbl = df.get(f"right_consistent_label_{i}")
            if r_lbl is None:
                r_lbl = pd.Series([[] for _ in range(len(df))])
            r_lbl = r_lbl.apply(lambda x: x if isinstance(x, list) else [])
            right_labels_cols.append(r_lbl.tolist())

    left_seqs = list(map(list, zip(*left_seqs_cols)))
    right_seqs = list(map(list, zip(*right_seqs_cols)))
    if include_labels:
        left_labels = list(map(list, zip(*left_labels_cols)))
        right_labels = list(map(list, zip(*right_labels_cols)))
        return left_seqs, right_seqs, left_labels, right_labels
    return left_seqs, right_seqs


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        x: input logits (B, C)
        y: targets (B, C) - multi-label binarized vectors
        """
        # 1. Calculate Sigmoid Probabilities
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid_pos = x_sigmoid
        x_sigmoid_neg = 1 - x_sigmoid

        # 2. Asymmetric Focusing Mechanism
        # For positives: standard focal loss behavior
        if self.gamma_pos > 0:
            p_pos = x_sigmoid_pos
            loss_pos = - y * torch.pow(1 - p_pos, self.gamma_pos) * torch.log(p_pos + self.eps)
        else:
            loss_pos = - y * torch.log(x_sigmoid_pos + self.eps)

        # For negatives: Asymmetric clipping (The Key Strategy)
        # If probability is very low (easy negative), clip it to suppress gradients
        if self.clip > 0:
            x_sigmoid_neg = (x_sigmoid_neg + self.clip).clamp(max=1)
        
        # Apply focusing parameter to negatives (usually higher than positives, e.g., 4)
        if self.gamma_neg > 0:
            p_neg = x_sigmoid_neg
            loss_neg = - (1 - y) * torch.pow(1 - p_neg, self.gamma_neg) * torch.log(p_neg + self.eps)
        else:
            loss_neg = - (1 - y) * torch.log(x_sigmoid_neg + self.eps)

        # 3. Combine
        loss = loss_pos + loss_neg
        
        return loss

def build_rank_eval_loader(eval_dataset, batch_size: int, rank: int, world_size: int):
    indices = list(range(rank, len(eval_dataset), world_size))
    rank_dataset = torch.utils.data.Subset(eval_dataset, indices)
    return DataLoader(rank_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)


def gather_variable_batch_tensor(tensor: torch.Tensor, device):
    local_count = torch.tensor([tensor.size(0)], device=device, dtype=torch.long)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_counts, local_count)
    counts = [count.item() for count in gathered_counts]
    max_count = max(counts)

    if tensor.size(0) < max_count:
        pad_shape = (max_count - tensor.size(0),) + tuple(tensor.shape[1:])
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=0)

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat([rank_tensor[:count].cpu() for rank_tensor, count in zip(gathered, counts)], dim=0)


def find_best_thresholds_per_label(
    probs: np.ndarray,
    y_true: np.ndarray,
    label_to_idx: Dict[str, int],
    default_threshold: float = 0.5,
):
    """Find the exact validation-F1-optimal threshold for every label."""
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    label_thresholds = {}
    label_best_f1 = {}

    for j in range(y_true.shape[1]):
        label = idx_to_label[j]
        valid = y_true[:, j] != -100
        yj_true = y_true[valid, j]
        yj_prob = probs[valid, j]

        if yj_true.size == 0:
            label_thresholds[label] = default_threshold
            label_best_f1[label] = float("nan")
            continue

        positive_count = int((yj_true == 1).sum())
        if positive_count == 0:
            label_thresholds[label] = default_threshold
            label_best_f1[label] = 0.0
            continue

        order = np.argsort(-yj_prob, kind="stable")
        sorted_probs = yj_prob[order]
        sorted_true = (yj_true[order] == 1).astype(np.int64)
        cumulative_tp = np.cumsum(sorted_true)
        group_ends = np.flatnonzero(
            np.r_[sorted_probs[1:] != sorted_probs[:-1], True]
        )
        predicted_positive = group_ends + 1
        true_positive = cumulative_tp[group_ends]
        f1_values = (2.0 * true_positive) / (
            predicted_positive + positive_count
        )
        best_f1 = float(np.max(f1_values))
        best_indices = np.flatnonzero(
            np.isclose(f1_values, best_f1, rtol=0.0, atol=1e-12)
        )
        best_index = int(best_indices[0])
        label_thresholds[label] = float(sorted_probs[group_ends[best_index]])
        label_best_f1[label] = best_f1

    return label_thresholds, label_best_f1


def _hierarchy_prediction_requirements(label_to_idx: Dict[str, int]):
    """Return each label plus all ancestors required for its final prediction."""
    requirements = {idx: {idx} for idx in label_to_idx.values()}
    changed = True
    while changed:
        changed = False
        for parent_label, child_labels in LABEL_HIERARCHY.items():
            parent_idx = label_to_idx.get(parent_label)
            if parent_idx is None:
                continue
            for child_label in child_labels:
                child_idx = label_to_idx.get(child_label)
                if child_idx is None:
                    continue
                before = len(requirements[child_idx])
                requirements[child_idx].update(requirements[parent_idx])
                changed = changed or len(requirements[child_idx]) != before
    return requirements


def _binary_f1(y_true_column: np.ndarray, y_pred_column: np.ndarray) -> float:
    """Calculate F1 for one label while excluding unknown (-100) targets."""
    valid = y_true_column != -100
    if not np.any(valid):
        return float("nan")
    true_positive_mask = y_true_column[valid] == 1
    predicted_positive_mask = y_pred_column[valid].astype(bool)
    true_positive = int(np.sum(true_positive_mask & predicted_positive_mask))
    denominator = int(np.sum(true_positive_mask)) + int(np.sum(predicted_positive_mask))
    return 0.0 if denominator == 0 else (2.0 * true_positive) / denominator


def optimize_hierarchy_aware_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    label_to_idx: Dict[str, int],
    initial_thresholds: Dict[str, float],
    max_passes: int = 10,
    improvement_tolerance: float = 1e-12,
):
    """Coordinate-optimize exact thresholds for final hierarchy-adjusted macro-F1.

    Every accepted update strictly improves the calibration-set objective, so
    the returned thresholds cannot reduce hierarchy-adjusted macro-F1 relative
    to the independently optimized starting thresholds.
    """
    thresholds = dict(initial_thresholds)
    independent_pred = binarize_with_label_thresholds(
        probs, thresholds, label_to_idx
    ).astype(bool)
    requirements = _hierarchy_prediction_requirements(label_to_idx)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    final_pred = np.zeros_like(independent_pred, dtype=bool)
    for target_idx, required_indices in requirements.items():
        final_pred[:, target_idx] = np.all(
            independent_pred[:, list(required_indices)], axis=1
        )
    current_f1 = np.array(
        [_binary_f1(y_true[:, j], final_pred[:, j]) for j in range(y_true.shape[1])],
        dtype=np.float64,
    )
    current_sum = float(np.nansum(current_f1))
    scored_label_count = int(np.sum(~np.isnan(current_f1)))
    initial_macro_f1 = current_sum / scored_label_count if scored_label_count else 0.0

    for _ in range(max_passes):
        improved_in_pass = False
        for source_idx in range(y_true.shape[1]):
            impacted = [
                target_idx
                for target_idx, required_indices in requirements.items()
                if source_idx in required_indices
            ]
            if (
                impacted == [source_idx]
                and requirements[source_idx] == {source_idx}
            ):
                continue

            finite_indices = np.flatnonzero(np.isfinite(probs[:, source_idx]))
            if finite_indices.size == 0:
                continue
            order = finite_indices[
                np.argsort(-probs[finite_indices, source_idx], kind="stable")
            ]
            sorted_probs = probs[order, source_idx]
            group_ends = np.flatnonzero(
                np.r_[sorted_probs[1:] != sorted_probs[:-1], True]
            )
            above_max = np.nextafter(
                np.float64(sorted_probs[0]), np.float64(np.inf)
            )
            candidate_thresholds = np.r_[
                above_max, sorted_probs[group_ends].astype(np.float64)
            ]
            candidate_sums = np.full(
                candidate_thresholds.shape, current_sum, dtype=np.float64
            )

            for target_idx in impacted:
                if np.isnan(current_f1[target_idx]):
                    continue
                other_requirements = requirements[target_idx] - {source_idx}
                if other_requirements:
                    fixed_pred = np.all(
                        independent_pred[:, list(other_requirements)], axis=1
                    )
                else:
                    fixed_pred = np.ones(probs.shape[0], dtype=bool)

                valid = y_true[:, target_idx] != -100
                true_positive_mask = y_true[:, target_idx] == 1
                positive_count = int(np.sum(valid & true_positive_mask))
                eligible_in_order = valid[order] & fixed_pred[order]
                predicted_positive = np.r_[
                    0, np.cumsum(eligible_in_order)[group_ends]
                ]
                true_positive = np.r_[
                    0,
                    np.cumsum(
                        eligible_in_order & true_positive_mask[order]
                    )[group_ends],
                ]
                denominator = positive_count + predicted_positive
                candidate_f1 = np.divide(
                    2.0 * true_positive,
                    denominator,
                    out=np.zeros_like(denominator, dtype=np.float64),
                    where=denominator != 0,
                )
                candidate_sums += candidate_f1 - current_f1[target_idx]

            best_sum = float(np.max(candidate_sums))
            if best_sum <= current_sum + improvement_tolerance:
                continue
            best_indices = np.flatnonzero(
                np.isclose(
                    candidate_sums, best_sum, rtol=0.0, atol=improvement_tolerance
                )
            )
            best_index = int(best_indices[0])
            best_threshold = float(candidate_thresholds[best_index])
            thresholds[idx_to_label[source_idx]] = best_threshold
            independent_pred[:, source_idx] = probs[:, source_idx] >= best_threshold

            for target_idx in impacted:
                final_pred[:, target_idx] = np.all(
                    independent_pred[:, list(requirements[target_idx])], axis=1
                )
                current_f1[target_idx] = _binary_f1(
                    y_true[:, target_idx], final_pred[:, target_idx]
                )
            current_sum = float(np.nansum(current_f1))
            improved_in_pass = True

        if not improved_in_pass:
            break

    final_macro_f1 = current_sum / scored_label_count if scored_label_count else 0.0
    print(
        "Hierarchy-aware threshold optimization - "
        f"macro-F1: {initial_macro_f1:.6f} -> {final_macro_f1:.6f}"
    )
    label_f1 = {
        idx_to_label[j]: float(current_f1[j]) for j in range(y_true.shape[1])
    }
    return thresholds, label_f1


def binarize_with_label_thresholds(
    probs: np.ndarray,
    label_thresholds: Dict[str, float],
    label_to_idx: Dict[str, int],
) -> np.ndarray:
    """Binarize each functional label using its validation-selected threshold."""
    y_pred = np.zeros_like(probs, dtype=np.int32)
    for label, idx in label_to_idx.items():
        y_pred[:, idx] = (probs[:, idx] >= label_thresholds.get(label, 0.5)).astype(np.int32)
    return y_pred


def save_validation_thresholds(
    thresholds_dir: str,
    epoch: int,
    label_thresholds: Dict[str, float],
    label_best_f1: Dict[str, float],
) -> str:
    """Save thresholds and per-label validation F1 values for one epoch."""
    os.makedirs(thresholds_dir, exist_ok=True)
    threshold_path = os.path.join(thresholds_dir, f"thresholds_epoch{epoch}.json")
    payload = {
        "epoch": epoch,
        "threshold_step": None,
        "threshold_search": "exact_probability_breakpoints",
        "hierarchy_aware": True,
        "hierarchy_policy": HIERARCHY_POLICY,
        "labels": {
            label: {
                "threshold": float(label_thresholds[label]),
                "validation_f1": (
                    None if np.isnan(label_best_f1[label]) else float(label_best_f1[label])
                ),
            }
            for label in label_thresholds
        },
    }
    with open(threshold_path, "w", encoding="utf-8") as threshold_file:
        json.dump(payload, threshold_file, indent=2, sort_keys=True)
    return threshold_path


def evaluate(eval_data_path: str, label_to_idx: Dict[str, int], annot_model: AnnotModel, device, threshold=0.5, batch_size: int = 8, criterion=None, rank: int = 0, world_size: int = 1, optimize_label_thresholds: bool = False):
    df = pd.read_json(eval_data_path, lines=True)
    prot_seqs = df['prot_seq'].tolist()
    left_seqs, right_seqs, left_labels, right_labels = _load_context_columns_from_df(df, include_labels=True)

    labels = df['mapped_annot'].tolist()
    ground_truth = labels

    inconsistent_labels = [[] for _ in prot_seqs]
    left_inconsistent_labels = [[[], [], []] for _ in prot_seqs]
    right_inconsistent_labels = [[[], [], []] for _ in prot_seqs]
    domains = [None for _ in prot_seqs]

    eval_dataset = ProteinDataset(
        prot_seqs,
        left_seqs,
        right_seqs,
        labels,
        left_labels,
        right_labels,
        inconsistent_labels,
        left_inconsistent_labels,
        right_inconsistent_labels,
        label_to_idx,
        domains,
    )
    if world_size > 1:
        eval_loader = build_rank_eval_loader(eval_dataset, batch_size, rank, world_size)
    else:
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)
    annot_model.eval()
    all_logits = []
    all_labels = []
    eval_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    eval_loss_count = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in eval_loader:
            (
                prot_tensor,
                prot_tensor_left,
                prot_tensor_right,
                mask_tokens_ids,
                mask_tokens_ids_left,
                mask_tokens_ids_right,
                labels,
                _left_labels,
                _right_labels,
                _domains,
                _domain_weight,
            ) = batch
            prot_tensor = prot_tensor.to(device)
            mask_tokens_ids = mask_tokens_ids.to(device)
            prot_tensor_left = prot_tensor_left.to(device)
            prot_tensor_right = prot_tensor_right.to(device)
            mask_tokens_ids_left = mask_tokens_ids_left.to(device)
            mask_tokens_ids_right = mask_tokens_ids_right.to(device)
            _, protein_function_logits, _, _ = annot_model(
                prot_tensor,
                prot_tensor_left,
                prot_tensor_right,
                mask_tokens_ids,
                mask_tokens_ids_left,
                mask_tokens_ids_right,
            )
            logits = protein_function_logits
            if criterion is not None:
                labels_device = labels.to(device)
                valid_mask = labels_device != -100
                if valid_mask.sum().item() > 0:
                    targets = labels_device.clone()
                    targets[~valid_mask] = 0
                    per_label_loss = criterion(logits, targets)
                    eval_loss_sum += (per_label_loss * valid_mask.to(per_label_loss.dtype)).sum().to(eval_loss_sum.dtype)
                    eval_loss_count += valid_mask.sum().to(eval_loss_count.dtype)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if len(all_logits) > 0:
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
    else:
        num_classes = len(label_to_idx)
        logits = torch.empty((0, num_classes), dtype=torch.float32)
        labels = torch.empty((0, num_classes), dtype=torch.float32)

    if world_size > 1:
        loss_stats = torch.stack([eval_loss_sum, eval_loss_count])
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        eval_loss_sum, eval_loss_count = loss_stats[0], loss_stats[1]
        logits = gather_variable_batch_tensor(logits.to(device), device)
        labels = gather_variable_batch_tensor(labels.to(device), device)

        if rank != 0:
            return None

    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    label_thresholds = None
    label_best_f1 = None
    if optimize_label_thresholds:
        label_thresholds, _ = find_best_thresholds_per_label(
            probs, y_true, label_to_idx
        )
        label_thresholds, label_best_f1 = optimize_hierarchy_aware_thresholds(
            probs, y_true, label_to_idx, label_thresholds
        )
        y_pred = binarize_with_label_thresholds(probs, label_thresholds, label_to_idx)
    elif isinstance(threshold, dict):
        y_pred = binarize_with_label_thresholds(probs, threshold, label_to_idx)
    else:
        y_pred = (probs >= threshold).astype(np.int32)
    y_pred = enforce_hierarchical_child_consistency(y_pred, label_to_idx)

    num_classes = y_true.shape[1]
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    accs = []
    aucs = []
    aprs = []
    f1s = []
    precisions = []
    recalls = []
    mccs = []

    for j in range(num_classes):
        yj_true = y_true[:, j]
        yj_prob = probs[:, j]
        yj_pred = y_pred[:, j]

        # per-class accuracy
        acc = (yj_true == yj_pred).mean()
        accs.append(acc)

        # AUROC (skip if only one class present)
        if np.unique(yj_true).size > 1:
            try:
                auc = roc_auc_score(yj_true, yj_prob)
                aucs.append(auc)
            except Exception:
                pass

        # AUPRC (skip if no positives)
        if yj_true.sum() > 0:
            try:
                ap = average_precision_score(yj_true, yj_prob)
                aprs.append(ap)
            except Exception:
                pass

        # F1 with zero_division=0 to avoid warnings
        try:
            f1 = f1_score(yj_true, yj_pred, zero_division=0)
            f1s.append(f1)
        except Exception:
            pass

        try:
            precision = precision_score(yj_true, yj_pred, zero_division=0)
            precisions.append(precision)
        except Exception:
            pass

        try:
            recall = recall_score(yj_true, yj_pred, zero_division=0)
            recalls.append(recall)
        except Exception:
            pass

        try:
            mcc = matthews_corrcoef(yj_true, yj_pred)
            mccs.append(mcc)
        except Exception:
            pass

        print(f"Class {idx_to_label[j]} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, AUPRC: {ap:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}")

    avg_acc = float(np.mean(accs)) if len(accs) > 0 else 0.0
    avg_auroc = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
    avg_auprc = float(np.mean(aprs)) if len(aprs) > 0 else 0.0
    avg_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    avg_precision = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    avg_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    avg_mcc = float(np.mean(mccs)) if len(mccs) > 0 else 0.0
    avg_loss = float((eval_loss_sum / eval_loss_count).item()) if eval_loss_count.item() > 0 else 0.0
    mean_label_best_f1 = (
        float(np.nanmean(list(label_best_f1.values())))
        if label_best_f1 is not None
        else None
    )

    print(
        f"Eval function - loss: {avg_loss:.6f}, accuracy: {avg_acc:.4f}, "
        f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, "
        f"AUROC: {avg_auroc:.4f}, AUPRC: {avg_auprc:.4f}, MCC: {avg_mcc:.4f}"
    )
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'precision': avg_precision,
        'recall': avg_recall,
        'auroc': avg_auroc,
        'auprc': avg_auprc,
        'f1': avg_f1,
        'mcc': avg_mcc,
        'probs': probs,
        'ground_truth': ground_truth,
        'label_thresholds': label_thresholds,
        'label_best_f1': label_best_f1,
        'mean_label_best_f1': mean_label_best_f1,
    }


def evaluate_domain(eval_data_path: str, annot_model: AnnotModel, label_to_idx: Dict[str, int], device, batch_size: int = 8, rank: int = 0, world_size: int = 1):
    df = pd.read_json(eval_data_path, lines=True)
    prot_seqs = df['prot_seq'].tolist()
    left_seqs, right_seqs = _load_context_columns_from_df(df, include_labels=False)

    domains = df['domain'].tolist()
    dummy_labels = [[] for _ in prot_seqs]
    dummy_left_labels = [[[], [], []] for _ in prot_seqs]
    dummy_right_labels = [[[], [], []] for _ in prot_seqs]
    dummy_inconsistent_labels = [[] for _ in prot_seqs]
    dummy_left_inconsistent_labels = [[[], [], []] for _ in prot_seqs]
    dummy_right_inconsistent_labels = [[[], [], []] for _ in prot_seqs]

    eval_dataset = ProteinDataset(
        prot_seqs,
        left_seqs,
        right_seqs,
        dummy_labels,
        dummy_left_labels,
        dummy_right_labels,
        dummy_inconsistent_labels,
        dummy_left_inconsistent_labels,
        dummy_right_inconsistent_labels,
        label_to_idx,
        domains,
    )
    if world_size > 1:
        eval_loader = build_rank_eval_loader(eval_dataset, batch_size, rank, world_size)
    else:
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)
    annot_model.eval()
    total_tp = torch.zeros((), dtype=torch.float64, device=device)
    total_fp = torch.zeros((), dtype=torch.float64, device=device)
    total_fn = torch.zeros((), dtype=torch.float64, device=device)
    eps = 1e-12
    with torch.no_grad():
        for batch in eval_loader:
            (
                prot_tensor,
                prot_tensor_left,
                prot_tensor_right,
                mask_tokens_ids,
                mask_tokens_ids_left,
                mask_tokens_ids_right,
                _labels,
                _left_labels,
                _right_labels,
                dom,
                _domain_weight,
            ) = batch
            prot_tensor = prot_tensor.to(device)
            mask_tokens_ids = mask_tokens_ids.to(device)
            prot_tensor_left = prot_tensor_left.to(device)
            prot_tensor_right = prot_tensor_right.to(device)
            mask_tokens_ids_left = mask_tokens_ids_left.to(device)
            mask_tokens_ids_right = mask_tokens_ids_right.to(device)
            dom = dom.to(device)
            probs, _, _, _ = annot_model(
                prot_tensor,
                prot_tensor_left,
                prot_tensor_right,
                mask_tokens_ids,
                mask_tokens_ids_left,
                mask_tokens_ids_right,
            )
            pred = probs >= 0.5

            mask = mask_tokens_ids.bool()
            residue_mask = mask.clone()
            T = residue_mask.size(1)
            if T >= 1:
                residue_mask[:, 0] = False
                reversed_mask = torch.flip(mask, dims=[1])
                last_true_from_end = torch.argmax(reversed_mask.to(torch.int64), dim=1)
                last_true_idx = (T - 1) - last_true_from_end
                has_true = mask.any(dim=1)
                batch_indices = torch.arange(mask.size(0), device=mask.device)
                residue_mask[batch_indices[has_true], last_true_idx[has_true]] = False

            valid = (dom >= 0) & residue_mask
            if valid.sum().item() == 0:
                continue
            y_true = (dom == 1) & valid
            y_pred = pred & valid
            total_tp += (y_true & y_pred).sum()
            total_fp += ((~y_true) & y_pred & valid).sum()
            total_fn += (y_true & (~y_pred) & valid).sum()

    if world_size > 1:
        stats = torch.stack([total_tp, total_fp, total_fn])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_tp, total_fp, total_fn = stats[0], stats[1], stats[2]

        if rank != 0:
            return None

    total_tp = total_tp.item()
    total_fp = total_fp.item()
    total_fn = total_fn.item()
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / max(precision + recall + eps, eps)
    print(f"Eval Domain - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {'precision': precision, 'recall': recall, 'f1': f1}

def append_eval_metrics(metrics_path: str, row: dict):
    fieldnames = [
        'epoch',
        'function_loss',
        'threshold_file',
        'function_accuracy',
        'function_mean_label_best_f1',
        'function_f1',
        'function_precision',
        'function_recall',
        'function_auroc',
        'function_auprc',
        'function_mcc',
        'domain_f1',
        'domain_precision',
        'domain_recall',
    ]
    write_header = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
    with open(metrics_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in fieldnames})


def _ddp_global_mean_from_local_sum(local_loss_sum: torch.Tensor,
                                    local_count: torch.Tensor,
                                    world_size: int):
    """
    Convert a local summed loss into an exact global mean under DDP.

    DDP averages gradients across ranks. Multiplying the local sum by
    world_size / global_count makes the final averaged gradient equivalent to
    backpropagating a single loss summed over all ranks and divided by the
    global valid count.
    """
    stats = torch.stack([
        local_loss_sum.detach(),
        local_count.detach().to(dtype=local_loss_sum.dtype),
    ])
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    global_loss_sum, global_count = stats[0], stats[1]

    if global_count.item() == 0:
        return local_loss_sum * 0.0, local_loss_sum.detach() * 0.0

    loss_for_backward = local_loss_sum * (float(world_size) / global_count)
    loss_for_logging = global_loss_sum / global_count
    return loss_for_backward, loss_for_logging


def compute_function_loss_ddp(criterion: torch.nn.Module,
                              protein_function_logits: torch.Tensor,
                              labels: torch.Tensor,
                              world_size: int):
    mask = (labels != -100)
    targets = labels.clone()
    targets[~mask] = 0

    loss = criterion(protein_function_logits, targets)
    local_loss_sum = (loss * mask.to(loss.dtype)).sum()
    local_count = mask.sum().to(device=protein_function_logits.device, dtype=loss.dtype)
    return _ddp_global_mean_from_local_sum(local_loss_sum, local_count, world_size)


def compute_domain_loss_ddp(domain_probs: torch.Tensor,
                            domains: torch.Tensor,
                            mask_tokens_ids: torch.Tensor,
                            domain_weight: torch.Tensor,
                            world_size: int):
    device = domain_probs.device
    has_invalid_domains = (domains == -100).any(dim=1)
    valid_token_mask = (domains >= 0) & mask_tokens_ids
    has_any_valid_token = valid_token_mask.any(dim=1)
    keep_mask = (~has_invalid_domains) & has_any_valid_token

    if keep_mask.sum().item() == 0:
        local_loss_sum = domain_probs.sum() * 0.0
        local_count = torch.zeros((), device=device, dtype=domain_probs.dtype)
        return _ddp_global_mean_from_local_sum(local_loss_sum, local_count, world_size)

    dp = domain_probs[keep_mask]
    dom = domains[keep_mask]
    msk = mask_tokens_ids[keep_mask]
    w = domain_weight[keep_mask]

    mask = msk.bool()
    residue_mask = mask.clone()
    T = residue_mask.size(1)
    if T >= 1:
        residue_mask[:, 0] = False
        reversed_mask = torch.flip(mask, dims=[1])
        last_true_from_end = torch.argmax(reversed_mask.to(torch.int64), dim=1)
        last_true_idx = (T - 1) - last_true_from_end
        has_true = mask.any(dim=1)
        batch_indices = torch.arange(mask.size(0), device=mask.device)
        residue_mask[batch_indices[has_true], last_true_idx[has_true]] = False

    valid_mask_kept = (dom >= 0) & residue_mask
    per_token_loss = F.binary_cross_entropy(
        dp.clamp(min=1e-6, max=1 - 1e-6),
        (dom == 1).to(torch.float32),
        reduction='none',
    )
    per_token_loss = per_token_loss * valid_mask_kept.to(per_token_loss.dtype)
    valid_counts = valid_mask_kept.sum(dim=1).clamp_min(1)
    per_sample_loss = per_token_loss.sum(dim=1) / valid_counts

    local_loss_sum = (per_sample_loss * w).sum()
    local_count = keep_mask.sum().to(device=device, dtype=local_loss_sum.dtype)
    return _ddp_global_mean_from_local_sum(local_loss_sum, local_count, world_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhageFlare with distributed data parallelism.")
    parser.add_argument("--label_dict_path", default="filtered_empathi_dict_added.json")
    parser.add_argument("--function_train_data_path", required=True)
    parser.add_argument(
        "--domain_train_data_path1",
        default="domain_data/bacteria_domain_train.jsonl",
    )
    parser.add_argument(
        "--domain_train_data_path2",
        default="domain_data/virus_no_bacteriophage_domain.jsonl",
    )
    parser.add_argument("--function_validation_data_path", required=True)
    parser.add_argument(
        "--domain_validation_data_path",
        default="domain_data/bacteriophage_domain.jsonl",
    )
    parser.add_argument("--checkpoint_save_path", default="checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Initialize Distributed Data Parallel
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=80))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Set seed for reproducibility
    seed = 100 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    label_to_idx = quantize_label(args.label_dict_path)
    data_path = args.function_train_data_path
    domain_path1 = args.domain_train_data_path1
    domain_path2 = args.domain_train_data_path2
    esmc_model_name = "esmc_600m"

    target_effective_batch_size = 150
    batch_size = args.batch_size
    num_epochs = 20
    lr = 5e-4 
    grad_accum_steps = max(1, round(target_effective_batch_size / (batch_size * world_size)))
    effective_batch_size = batch_size * world_size * grad_accum_steps
    warmup_ratio = 0.1
    threshold = 0.5

    # --- Model ---
    num_classes = len(label_to_idx)
    annot_model = AnnotModel(esmc_model_name, num_classes, device)
    annot_model = annot_model.to(device)
    
    # Wrap model in DDP
    # find_unused_parameters=True is often needed if some sub-modules might not be used in forward pass
    # or if there are parameters not part of the computation graph (frozen params usually fine though).
    annot_model = DDP(annot_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    annot_model.train()
    # Ensure the frozen ESM part stays in eval mode
    annot_model.module.esmc_model.eval()

    # --- Data ---
    train_loader, _num_samples, _all_labels, train_sampler = build_train_dataloader(
        data_path,
        label_to_idx,
        domain_path1,
        domain_path2,
        batch_size=batch_size,
        shuffle=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    # --- Loss & Optimizer ---
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, annot_model.parameters()),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Scheduler
    steps_per_epoch = len(train_loader) // grad_accum_steps
    if steps_per_epoch == 0:
        raise ValueError(
            f"Not enough batches per rank ({len(train_loader)}) for "
            f"grad_accum_steps={grad_accum_steps}. Increase data size or reduce target_effective_batch_size."
        )
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_warmup_linear_decay_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01)

    checkpoints_dir = args.checkpoint_save_path
    thresholds_dir = './valid_thresholds'
    if rank == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(thresholds_dir, exist_ok=True)
        metrics_path = './validation_metric.csv'
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        dropped_batches = len(train_loader) - steps_per_epoch * grad_accum_steps
        print(
            f"Starting training on {world_size} GPUs. "
            f"per_gpu_batch={batch_size}, grad_accum_steps={grad_accum_steps}, "
            f"effective_batch_size={effective_batch_size} (target={target_effective_batch_size}), "
            f"steps_per_epoch={steps_per_epoch}, dropped_tail_batches_per_rank={dropped_batches}, "
            f"total_steps={total_steps}"
        )

    # --- Training Loop ---
    for epoch in range(num_epochs):
        # Important: set epoch for DistributedSampler
        train_sampler.set_epoch(epoch)
        
        running_loss = 0.0
        last_grad_norm = 0.0
        optimizer.zero_grad()
        max_train_batches = steps_per_epoch * grad_accum_steps
        
        for step, batch in enumerate(train_loader):
            if step >= max_train_batches:
                break

            (
                prot_tensor,
                prot_tensor_left,
                prot_tensor_right,
                mask_tokens_ids,
                mask_tokens_ids_left,
                mask_tokens_ids_right,
                labels,
                _left_labels,
                _right_labels,
                domains,
                domain_weight,
            ) = batch
            prot_tensor = prot_tensor.to(device)
            prot_tensor_left = prot_tensor_left.to(device)
            prot_tensor_right = prot_tensor_right.to(device)
            mask_tokens_ids = mask_tokens_ids.to(device)
            mask_tokens_ids_left = mask_tokens_ids_left.to(device)
            mask_tokens_ids_right = mask_tokens_ids_right.to(device)
            labels = labels.to(device)
            domains = domains.to(device)
            domain_weight = domain_weight.to(device)

            domain_probs, protein_function_logits, _, _ = annot_model(
                prot_tensor,
                prot_tensor_left,
                prot_tensor_right,
                mask_tokens_ids,
                mask_tokens_ids_left,
                mask_tokens_ids_right,
            )
            
            # Loss
            function_loss, function_loss_log = compute_function_loss_ddp(
                criterion, protein_function_logits, labels, world_size
            )
            domain_loss, domain_loss_log = compute_domain_loss_ddp(
                domain_probs, domains, mask_tokens_ids, domain_weight, world_size
            )

            # Limit the domain objective to at most one third of the function
            # objective. Use the globally reduced logging losses so every DDP
            # rank applies the same detached scale to its local backward loss.
            domain_loss_scale = torch.clamp(
                (function_loss_log.detach() / 3.0)
                / (domain_loss_log.detach() + 1e-12),
                max=1.0,
            )
            domain_loss = domain_loss * domain_loss_scale
            domain_loss_log = domain_loss_log * domain_loss_scale

            loss = function_loss + domain_loss
            loss_log = function_loss_log + domain_loss_log
            
            # Hack: Add dummy dependency on all parameters to prevent DDP hangs
            # when some ranks have unused parameters.
            loss = loss + 0.0 * sum(p.sum() for p in annot_model.parameters() if p.requires_grad)

            # Backward
            (loss / grad_accum_steps).backward()

            # Optimizer step
            if (step + 1) % grad_accum_steps == 0:
                last_grad_norm = torch.nn.utils.clip_grad_norm_(annot_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss_log.item()

            if rank == 0 and (step + 1) % 1000 == 0:
                avg_loss = running_loss / 1000
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1} Step {step+1}: loss={avg_loss:.6f}, lr={current_lr:.6f}, grad_norm={last_grad_norm:.4f}")
                running_loss = 0.0

        # --- Evaluation and Checkpointing ---
        dist.barrier()

        if rank == 0:
            print(f"Evaluating Epoch {epoch+1}...")
        eval_data_path = args.function_validation_data_path
        function_metrics = evaluate(
            eval_data_path,
            label_to_idx,
            annot_model.module,
            device,
            threshold,
            batch_size=batch_size,
            criterion=criterion,
            rank=rank,
            world_size=world_size,
            optimize_label_thresholds=True,
        )
        eval_domain_path = args.domain_validation_data_path
        domain_metrics = evaluate_domain(
            eval_domain_path,
            annot_model.module,
            label_to_idx,
            device,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
        )

        # Reset to train mode on every rank before resuming DDP training.
        annot_model.train()
        annot_model.module.esmc_model.eval()

        if rank == 0:
            threshold_path = save_validation_thresholds(
                thresholds_dir,
                epoch + 1,
                function_metrics['label_thresholds'],
                function_metrics['label_best_f1'],
            )
            print(f"Saved validation thresholds: {threshold_path}")
            append_eval_metrics(metrics_path, {
                'epoch': epoch + 1,
                'function_loss': function_metrics.get('loss'),
                'threshold_file': os.path.relpath(threshold_path, start='.'),
                'function_accuracy': function_metrics.get('accuracy'),
                'function_mean_label_best_f1': function_metrics.get('mean_label_best_f1'),
                'function_f1': function_metrics.get('f1'),
                'function_precision': function_metrics.get('precision'),
                'function_recall': function_metrics.get('recall'),
                'function_auroc': function_metrics.get('auroc'),
                'function_auprc': function_metrics.get('auprc'),
                'function_mcc': function_metrics.get('mcc'),
                'domain_f1': domain_metrics.get('f1'),
                'domain_precision': domain_metrics.get('precision'),
                'domain_recall': domain_metrics.get('recall'),
            })
            print(f"Saved evaluation metrics: {metrics_path}")

            # Save checkpoint
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch{epoch+1}_adapter_attnresidue_AsymmetricLoss_context3.pt')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': annot_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'label_to_idx': label_to_idx,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                print(f"Rank {rank} continuing...")

        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
