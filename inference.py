import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import ProteinDataset, make_empty_neighbor_context
from DDP_train import (
    _load_context_columns_from_df,
    binarize_with_label_thresholds,
    enforce_hierarchical_child_consistency,
)
from model import AnnotModel


def read_fasta(data_path: str) -> pd.DataFrame:
    records = []
    seq_id = None
    sequence_parts = []

    with open(data_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    if not sequence_parts:
                        raise ValueError(f"FASTA record {seq_id!r} has an empty sequence")
                    records.append({"seq_id": seq_id, "prot_seq": "".join(sequence_parts)})
                seq_id = line[1:].strip().split(maxsplit=1)[0] if line[1:].strip() else ""
                if not seq_id:
                    raise ValueError(f"FASTA header on line {line_number} has no sequence identifier")
                sequence_parts = []
            else:
                if seq_id is None:
                    raise ValueError(f"FASTA sequence data appears before the first header on line {line_number}")
                sequence_parts.append("".join(line.split()))

    if seq_id is not None:
        if not sequence_parts:
            raise ValueError(f"FASTA record {seq_id!r} has an empty sequence")
        records.append({"seq_id": seq_id, "prot_seq": "".join(sequence_parts)})
    return pd.DataFrame(records)


def load_target_data(data_path: str) -> tuple[pd.DataFrame, bool]:
    with open(data_path, "r", encoding="utf-8") as file:
        first_content = next((line.strip() for line in file if line.strip()), "")
    if not first_content:
        raise ValueError(f"Target dataset is empty: {data_path}")

    is_fasta = first_content.startswith(">")
    df = read_fasta(data_path) if is_fasta else pd.read_json(data_path, lines=True)
    return df, is_fasta


def build_inference_dataset(data_path: str, label_to_idx: dict) -> tuple[ProteinDataset, pd.DataFrame]:
    df, is_fasta = load_target_data(data_path)
    required_columns = {"seq_id", "prot_seq"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {data_path}: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError(f"Target dataset is empty: {data_path}")

    prot_seqs = df["prot_seq"].tolist()
    if any(not isinstance(sequence, str) or not sequence for sequence in prot_seqs):
        raise ValueError("Every prot_seq value must be a non-empty string")
    if any(len(sequence) > 2048 for sequence in prot_seqs):
        raise ValueError("Long sequences found in the dataset")

    has_context = not is_fasta and any(
        f"{side}_seq_{i}" in df.columns
        for side in ("left", "right")
        for i in range(3)
    )
    if has_context:
        left_seqs, right_seqs = _load_context_columns_from_df(df, include_labels=False)
    else:
        left_seqs, right_seqs, _, _, _, _, _ = make_empty_neighbor_context(len(df))

    labels = [[] for _ in prot_seqs]
    left_labels = [[[], [], []] for _ in prot_seqs]
    right_labels = [[[], [], []] for _ in prot_seqs]
    inconsistent_labels = [[] for _ in prot_seqs]
    left_inconsistent_labels = [[[], [], []] for _ in prot_seqs]
    right_inconsistent_labels = [[[], [], []] for _ in prot_seqs]
    domains = [None for _ in prot_seqs]

    dataset = ProteinDataset(
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
    return dataset, df


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[AnnotModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "label_to_idx" not in checkpoint:
        raise ValueError("Checkpoint must contain label_to_idx")

    label_to_idx = checkpoint["label_to_idx"]
    if not isinstance(label_to_idx, dict) or not label_to_idx:
        raise ValueError("Checkpoint label_to_idx must be a non-empty dictionary")
    if sorted(label_to_idx.values()) != list(range(len(label_to_idx))):
        raise ValueError("Checkpoint label_to_idx indices must be unique and consecutive from zero")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = AnnotModel("esmc_600m", len(label_to_idx), device).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, label_to_idx


def load_thresholds(thresholds_file: str, label_to_idx: dict) -> dict[str, float]:
    with open(thresholds_file, "r", encoding="utf-8") as file:
        thresholds = json.load(file)
    if not isinstance(thresholds, dict):
        raise ValueError("Thresholds file must contain a label-to-threshold JSON object")

    missing_labels = set(label_to_idx) - set(thresholds)
    if missing_labels:
        raise ValueError(f"Thresholds file is missing labels: {sorted(missing_labels)}")
    return {label: float(thresholds[label]) for label in label_to_idx}


def predict(
    target_data_path: str,
    checkpoint_path: str,
    thresholds_file: str,
    output_path: str,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_to_idx = load_checkpoint(checkpoint_path, device)
    label_thresholds = load_thresholds(thresholds_file, label_to_idx)
    dataset, df = build_inference_dataset(target_data_path, label_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    probability_batches = []
    with torch.inference_mode():
        for batch in loader:
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
                _domains,
                _domain_weight,
            ) = batch
            _, logits, _, _ = model(
                prot_tensor.to(device),
                prot_tensor_left.to(device),
                prot_tensor_right.to(device),
                mask_tokens_ids.to(device),
                mask_tokens_ids_left.to(device),
                mask_tokens_ids_right.to(device),
            )
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(probability_batches, axis=0)
    predictions = binarize_with_label_thresholds(probs, label_thresholds, label_to_idx)
    predictions = enforce_hierarchical_child_consistency(predictions, label_to_idx)
    labels_by_index = sorted(label_to_idx, key=label_to_idx.get)
    results = pd.DataFrame(
        {
            "index": np.arange(len(df)),
            "seq_id": df["seq_id"].tolist(),
            "predicted_labels": [
                ";".join(label for label, index in label_to_idx.items() if predictions[row, index] == 1)
                for row in range(len(df))
            ],
        }
    )
    for column, label in enumerate(labels_by_index):
        results[f"{label}_probability"] = probs[:, column]

    output_parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_parent, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protein annotation inference and save predictions as CSV.")
    parser.add_argument("--target_data_path", required=True, help="Target dataset in JSONL or FASTA format.")
    parser.add_argument("--checkpoint_path", default="checkpoint/PhageFlare.pt", help="Trained model checkpoint.")
    parser.add_argument(
        "--thresholds_file",
        default="binarization_threshold_for_each_label_high_precision.json",
        help="JSON file containing per-label binarization thresholds.",
    )
    parser.add_argument("--output_path", required=True, help="Output CSV path.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    args = parser.parse_args()
    predict(
        args.target_data_path,
        args.checkpoint_path,
        args.thresholds_file,
        args.output_path,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
