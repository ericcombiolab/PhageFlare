from esm.tokenization import EsmSequenceTokenizer
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from esm.utils import encoding
import torch
import json
import pandas as pd
from typing import List, Union


def build_domain_tensor(prot_tokens_ids: torch.Tensor, domain_label: Union[List[List[Union[int, str]]], None]) -> torch.Tensor:
    """
    Build per-token domain labels aligned to ESM tokenized sequence length.
    Domain intervals are 1-based residue positions; token index 0 is BOS.
    """
    T = prot_tokens_ids.size(0)
    domain_tensor = torch.zeros(T, dtype=torch.long)
    if not isinstance(domain_label, list):
        if pd.isna(domain_label) or domain_label is None:
            return torch.full((T,), -100, dtype=torch.long)

    residue_len = max(0, T - 2)
    for span in domain_label:
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"Domain label {span} is not a list of length 2")
        start_1b = int(span[0])
        end_1b = int(span[1])
        if residue_len == 0:
            raise ValueError("The input protein sequence length is 0, which is not allowed")

        start0 = max(0, min(residue_len - 1, start_1b - 1))
        end0 = max(0, min(residue_len - 1, end_1b - 1))
        s0, e0 = (start0, end0) if start0 <= end0 else (end0, start0)
        token_s = s0 + 1
        token_e = min(e0 + 1, T - 1)
        if token_s <= token_e:
            domain_tensor[token_s:token_e + 1] = 1

    if T >= 1:
        domain_tensor[0] = 0
    if T >= 2:
        domain_tensor[-1] = 0
    return domain_tensor


def quantize_label(data_path):
    dict_label = json.load(open(data_path))
    labels = dict_label.keys()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    return label_to_idx


def make_empty_neighbor_context(num_samples: int, num_neighbors: int = 3):
    """Return empty left/right neighbor sequences and labels for inference-only datasets."""
    empty_labels = [[] for _ in range(num_samples)]
    neighbor_empty = [[[] for _ in range(num_neighbors)] for _ in range(num_samples)]
    neighbor_seq_empty = [["" for _ in range(num_neighbors)] for _ in range(num_samples)]
    return neighbor_seq_empty, neighbor_seq_empty, neighbor_empty, neighbor_empty, empty_labels, neighbor_empty, neighbor_empty


class ProteinDataset(Dataset):
    def __init__(
        self,
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
        domain_labels,
    ):
        super().__init__()
        self.prot_seqs = prot_seqs
        self.left_seqs = left_seqs
        self.right_seqs = right_seqs
        self.labels = labels
        self.left_labels = left_labels
        self.right_labels = right_labels
        self.inconsistent_labels = inconsistent_labels
        self.left_inconsistent_labels = left_inconsistent_labels
        self.right_inconsistent_labels = right_inconsistent_labels
        self.esmc_tokenizer = EsmSequenceTokenizer()
        self.label_to_idx = label_to_idx
        self.domain_labels = domain_labels
        self.num_neighbors = 3

    def __len__(self):
        return len(self.prot_seqs)

    def __getitem__(self, idx):
        prot_seq = self.prot_seqs[idx]
        left_seq_list = self.left_seqs[idx]
        right_seq_list = self.right_seqs[idx]

        label = self.labels[idx]
        left_label_list = self.left_labels[idx]
        right_label_list = self.right_labels[idx]

        inconsistent_label = self.inconsistent_labels[idx]
        left_inconsistent_label_list = self.left_inconsistent_labels[idx]
        right_inconsistent_label_list = self.right_inconsistent_labels[idx]

        domain_label = self.domain_labels[idx]

        label_idx = self.get_label_idx(label, inconsistent_label)
        prot_tokens_ids = encoding.tokenize_sequence(prot_seq, self.esmc_tokenizer)

        left_tokens_ids_list = []
        left_label_idx_list = []
        right_tokens_ids_list = []
        right_label_idx_list = []

        for i in range(self.num_neighbors):
            l_seq = left_seq_list[i] if i < len(left_seq_list) else ""
            l_lbl = left_label_list[i] if i < len(left_label_list) else []
            l_inc = left_inconsistent_label_list[i] if i < len(left_inconsistent_label_list) else []

            if not isinstance(l_seq, str) or l_seq == "":
                l_ids = encoding.tokenize_sequence("", self.esmc_tokenizer)
            else:
                l_ids = encoding.tokenize_sequence(l_seq, self.esmc_tokenizer)
            left_tokens_ids_list.append(l_ids)
            left_label_idx_list.append(self.get_label_idx(l_lbl, l_inc))

            r_seq = right_seq_list[i] if i < len(right_seq_list) else ""
            r_lbl = right_label_list[i] if i < len(right_label_list) else []
            r_inc = right_inconsistent_label_list[i] if i < len(right_inconsistent_label_list) else []

            if not isinstance(r_seq, str) or r_seq == "":
                r_ids = encoding.tokenize_sequence("", self.esmc_tokenizer)
            else:
                r_ids = encoding.tokenize_sequence(r_seq, self.esmc_tokenizer)
            right_tokens_ids_list.append(r_ids)
            right_label_idx_list.append(self.get_label_idx(r_lbl, r_inc))

        domain_tensor = build_domain_tensor(prot_tokens_ids, domain_label)

        return (
            prot_tokens_ids,
            left_tokens_ids_list,
            right_tokens_ids_list,
            label_idx,
            left_label_idx_list,
            right_label_idx_list,
            domain_tensor,
        )

    def get_label_idx(self, label, inconsistent_label):
        if isinstance(label, list):
            label_idx = torch.zeros(len(self.label_to_idx), dtype=torch.float32)
            is_fully_invalid = False
            for l in label:
                if pd.isna(l) or l is None:
                    is_fully_invalid = True
                    break

            if is_fully_invalid:
                label_idx = torch.full((len(self.label_to_idx),), -100, dtype=torch.float32)
            else:
                for l in label:
                    if l not in self.label_to_idx:
                        raise ValueError(f"Label {l} not found in provided label_to_idx")
                    label_idx[self.label_to_idx[l]] = 1.0

                if isinstance(inconsistent_label, list) and len(inconsistent_label) > 0:
                    for l in inconsistent_label:
                        if l in self.label_to_idx:
                            label_idx[self.label_to_idx[l]] = -100.0
        else:
            if pd.isna(label) or label is None:
                label_idx = torch.full((len(self.label_to_idx),), -100, dtype=torch.float32)
            else:
                raise ValueError(f"Label {label} is not a list")
        return label_idx

    def collate_fn(self, batch):
        (
            prot_list,
            left_list_of_lists,
            right_list_of_lists,
            label_list,
            left_label_list_of_lists,
            right_label_list_of_lists,
            domain_list,
        ) = zip(*batch)

        max_len = max(t.size(0) for t in prot_list)
        pad_id = self.esmc_tokenizer.pad_token_id

        padded_prot = []
        masks = []
        for t in prot_list:
            if t.size(0) < max_len:
                pad = torch.full((max_len - t.size(0),), pad_id, dtype=t.dtype)
                t_pad = torch.cat([t, pad], dim=0)
            else:
                t_pad = t
            padded_prot.append(t_pad)
            masks.append((t_pad != pad_id))
        prot_tensor = torch.stack(padded_prot, dim=0)
        mask_tensor = torch.stack(masks, dim=0).to(torch.bool)
        labels_tensor = torch.stack(label_list, dim=0).to(torch.float32)

        all_left_tensors = [t for sublist in left_list_of_lists for t in sublist]
        max_len_left = max(t.size(0) for t in all_left_tensors) if all_left_tensors else 0

        padded_left = []
        masks_left = []
        for sublist in left_list_of_lists:
            sample_padded = []
            sample_mask = []
            for t in sublist:
                if t.size(0) < max_len_left:
                    pad = torch.full((max_len_left - t.size(0),), pad_id, dtype=t.dtype)
                    t_pad = torch.cat([t, pad], dim=0)
                else:
                    t_pad = t
                sample_padded.append(t_pad)
                sample_mask.append((t_pad != pad_id))
            padded_left.append(torch.stack(sample_padded, dim=0))
            masks_left.append(torch.stack(sample_mask, dim=0))

        prot_tensor_left = torch.stack(padded_left, dim=0) if padded_left else torch.empty(0)
        mask_tensor_left = torch.stack(masks_left, dim=0).to(torch.bool) if masks_left else torch.empty(0)
        left_labels_tensor = torch.stack(
            [torch.stack(sublist, dim=0) for sublist in left_label_list_of_lists],
            dim=0,
        ).to(torch.float32)

        all_right_tensors = [t for sublist in right_list_of_lists for t in sublist]
        max_len_right = max(t.size(0) for t in all_right_tensors) if all_right_tensors else 0

        padded_right = []
        masks_right = []
        for sublist in right_list_of_lists:
            sample_padded = []
            sample_mask = []
            for t in sublist:
                if t.size(0) < max_len_right:
                    pad = torch.full((max_len_right - t.size(0),), pad_id, dtype=t.dtype)
                    t_pad = torch.cat([t, pad], dim=0)
                else:
                    t_pad = t
                sample_padded.append(t_pad)
                sample_mask.append((t_pad != pad_id))
            padded_right.append(torch.stack(sample_padded, dim=0))
            masks_right.append(torch.stack(sample_mask, dim=0))

        prot_tensor_right = torch.stack(padded_right, dim=0) if padded_right else torch.empty(0)
        mask_tensor_right = torch.stack(masks_right, dim=0).to(torch.bool) if masks_right else torch.empty(0)
        right_labels_tensor = torch.stack(
            [torch.stack(sublist, dim=0) for sublist in right_label_list_of_lists],
            dim=0,
        ).to(torch.float32)

        padded_domain = []
        for d in domain_list:
            if d.size(0) < max_len:
                pad_dom = torch.full((max_len - d.size(0),), -1, dtype=d.dtype)
                d_pad = torch.cat([d, pad_dom], dim=0)
            else:
                d_pad = d
            padded_domain.append(d_pad)

        domain_tensor = torch.stack(padded_domain, dim=0).to(torch.long)
        # Count domain labels over amino-acid residues only.  The tokenizer
        # mask also includes BOS and EOS, so remove those positions before
        # computing the per-protein class-balance weight.
        residue_mask = mask_tensor.clone()
        T = residue_mask.size(1)
        if T >= 1:
            residue_mask[:, 0] = False
            reversed_mask = torch.flip(mask_tensor, dims=[1])
            last_true_from_end = torch.argmax(reversed_mask.to(torch.int64), dim=1)
            last_true_idx = (T - 1) - last_true_from_end
            has_true = mask_tensor.any(dim=1)
            batch_indices = torch.arange(mask_tensor.size(0), device=mask_tensor.device)
            residue_mask[batch_indices[has_true], last_true_idx[has_true]] = False

        valid_mask = (domain_tensor >= 0) & residue_mask
        pos_counts = ((domain_tensor == 1) & valid_mask).sum(dim=1).to(torch.float32)
        neg_counts = ((domain_tensor == 0) & valid_mask).sum(dim=1).to(torch.float32)
        domain_weight = torch.where(
            (pos_counts + neg_counts) > 0,
            neg_counts / torch.clamp_min(pos_counts, 1.0),
            torch.ones_like(pos_counts),
        ).to(torch.float32)

        return (
            prot_tensor,
            prot_tensor_left,
            prot_tensor_right,
            mask_tensor,
            mask_tensor_left,
            mask_tensor_right,
            labels_tensor,
            left_labels_tensor,
            right_labels_tensor,
            domain_tensor,
            domain_weight,
        )


def build_train_dataloader(
    data_path: str,
    label_to_idx,
    domain_path1: str,
    domain_path2: str,
    batch_size: int = 8,
    shuffle: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Build training DataLoader (and optional DistributedSampler) for PHROGs + domain data.
    """
    df = pd.read_json(data_path, lines=True)
    df_domain1 = pd.read_json(domain_path1, lines=True)
    df_domain2 = pd.read_json(domain_path2, lines=True)
    df_domain = pd.concat([df_domain1, df_domain2], ignore_index=True)
    df = pd.concat([df, df_domain], ignore_index=True)

    prot_seqs = df["prot_seq"].tolist()

    left_seqs_cols = []
    right_seqs_cols = []
    left_labels_cols = []
    right_labels_cols = []
    left_inc_labels_cols = []
    right_inc_labels_cols = []

    for i in range(3):
        left_seqs_cols.append(df.get(f"left_seq_{i}", pd.Series([None] * len(df))).tolist())
        right_seqs_cols.append(df.get(f"right_seq_{i}", pd.Series([None] * len(df))).tolist())
        left_labels_cols.append(df.get(f"left_consistent_label_{i}", pd.Series([None] * len(df))).tolist())
        right_labels_cols.append(df.get(f"right_consistent_label_{i}", pd.Series([None] * len(df))).tolist())
        left_inc_labels_cols.append(df.get(f"left_inconsistent_label_{i}", pd.Series([None] * len(df))).tolist())
        right_inc_labels_cols.append(df.get(f"right_inconsistent_label_{i}", pd.Series([None] * len(df))).tolist())

    left_seqs = list(map(list, zip(*left_seqs_cols)))
    right_seqs = list(map(list, zip(*right_seqs_cols)))
    left_labels = list(map(list, zip(*left_labels_cols)))
    right_labels = list(map(list, zip(*right_labels_cols)))
    left_inconsistent_labels = list(map(list, zip(*left_inc_labels_cols)))
    right_inconsistent_labels = list(map(list, zip(*right_inc_labels_cols)))

    labels = df["consistent_label"].tolist()
    inconsistent_labels = df["inconsistent_label"].tolist()
    domains = df["domain"].tolist()

    for prot_seq in prot_seqs:
        if len(prot_seq) > 2048:
            raise ValueError("Long sequences found in the dataset")

    train_dataset = ProteinDataset(
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

    if distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=train_dataset.collate_fn,
        )
    else:
        sampler = None
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=train_dataset.collate_fn,
        )

    return loader, len(train_dataset), labels, sampler
