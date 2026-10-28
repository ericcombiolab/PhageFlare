import math

import torch
import torch.nn as nn
from esm.models.esmc import ESMC


def _build_residue_mask(reference: torch.Tensor, mask_tokens_ids: torch.Tensor = None) -> torch.Tensor:
    """
    Build a residue-only mask aligned to token tensors.
    True means a real amino-acid residue; BOS, EOS, and padding are False.
    """
    B, T = reference.size(0), reference.size(1)
    if mask_tokens_ids is not None:
        mask = mask_tokens_ids.to(device=reference.device, dtype=torch.bool)
        residue_mask = mask.clone()
    else:
        residue_mask = torch.ones(B, T, dtype=torch.bool, device=reference.device)
        mask = residue_mask

    if T >= 1:
        residue_mask[:, 0] = False

    if T >= 1:
        reversed_mask = torch.flip(mask, dims=[1])
        last_true_from_end = torch.argmax(reversed_mask.to(torch.int64), dim=1)
        last_true_idx = (T - 1) - last_true_from_end
        has_true = mask.any(dim=1)
        batch_indices = torch.arange(B, device=reference.device)
        residue_mask[batch_indices[has_true], last_true_idx[has_true]] = False

    return residue_mask


def _masked_mean_pool(embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool over the sequence dimension, excluding masked positions.
    embed: (B, T, H); mask: (B, T), True = include (residue only, no BOS/EOS/pad).
    Returns: (B, H)
    """
    mask = mask.to(device=embed.device, dtype=embed.dtype).unsqueeze(-1)
    summed = (embed * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class LabelWiseAttentionPooling(nn.Module):
    """
    Label-specific multi-head attention over residue embeddings.

    Each label query attends over residue key/value embeddings using
    multi-head attention.

    Args:
        hidden_dim: number of features in each residue embedding
        num_classes: number of function labels
        num_heads: number of attention heads

    Inputs:
        residue_emb:  (B, T, H)
        residue_mask: (B, T), True = valid residue

    Output:
        label_emb: (B, C, H)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.label_queries = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.label_queries)

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        residue_emb: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, H = residue_emb.shape
        if H != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {H}")

        if residue_mask is None:
            residue_mask = torch.ones((B, T), device=residue_emb.device, dtype=torch.bool)
        else:
            residue_mask = residue_mask.to(device=residue_emb.device, dtype=torch.bool)

        # queries: (B, C, H); keys/values: (B, T, H)
        queries = self.label_queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~residue_mask
        attn_mask = None
        if residue_weights is not None:
            residue_weights = residue_weights.to(device=residue_emb.device, dtype=residue_emb.dtype)
            residue_weights = residue_weights * residue_mask.to(dtype=residue_emb.dtype)
            log_weights = residue_weights.clamp_min(1e-6).log()
            attn_mask = log_weights[:, None, :].expand(B, self.num_classes, T)
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            key_padding_mask = torch.zeros((B, T), device=residue_emb.device, dtype=residue_emb.dtype)
            key_padding_mask = key_padding_mask.masked_fill(~residue_mask, float("-inf"))

        label_emb, _ = self.mha(
            query=queries,
            key=residue_emb,
            value=residue_emb,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        label_emb = torch.nan_to_num(label_emb, nan=0.0, neginf=0.0, posinf=0.0)
        return label_emb

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class DomainBoundaryBiLSTM(nn.Module):
    def __init__(self, esmc_embed_dim: int, hidden_dim: int = None, num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or esmc_embed_dim // 2
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=esmc_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(lstm_out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, prot_embed: torch.Tensor, mask_tokens_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(prot_embed)
        if mask_tokens_ids is not None:
            lengths = mask_tokens_ids.bool().sum(dim=1).clamp_min(1).to("cpu")
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=prot_embed.size(1),
            )
        else:
            lstm_out, _ = self.lstm(x)

        probs = self.sigmoid(self.classifier(lstm_out).squeeze(-1))
        residue_mask = _build_residue_mask(prot_embed, mask_tokens_ids)
        return probs * residue_mask.to(probs.dtype)

class LabelContextCalibrator(nn.Module):
    """
    Late-fusion context module that turns neighbor predicted label probabilities into a
    low-rank context prior over labels, then applies a gated residual correction.
    """

    def __init__(
        self,
        num_classes: int,
        label_emb_dim: int = 128,
        max_k: int = 10,
        neighbor_dropout: float = 0.2,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.1,
        per_class_gate: bool = True,
        detach_neighbors: bool = True,
        detach_target_for_gate: bool = True,
        detach_target_for_context: bool = True,
        use_bilstm: bool = True,
        lstm_hidden_dim: int = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.label_emb_dim = int(label_emb_dim)
        self.max_k = int(max_k)
        self.neighbor_dropout = float(neighbor_dropout)
        self.per_class_gate = bool(per_class_gate)
        self.detach_neighbors = bool(detach_neighbors)
        self.detach_target_for_gate = bool(detach_target_for_gate)
        self.detach_target_for_context = bool(detach_target_for_context)
        self.use_bilstm = bool(use_bilstm)
        self.eps = float(eps)

        d = self.label_emb_dim
        C = self.num_classes

        self.label_emb_in = nn.Embedding(C, d)
        self.label_emb_out = nn.Embedding(C, d)
        nn.init.normal_(self.label_emb_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.label_emb_out.weight, mean=0.0, std=0.02)

        self.out_bias = nn.Parameter(torch.zeros(C))

        self.pos_emb = nn.Embedding(2 * self.max_k + 1, d)
        nn.init.zeros_(self.pos_emb.weight)

        self.pos_alpha = nn.Embedding(2 * self.max_k + 1, 1)
        nn.init.constant_(self.pos_alpha.weight, -1.0)

        self.nei_norm = nn.LayerNorm(d)

        self.nei_relevance = nn.Sequential(
            nn.Linear(3 * d + 1, d),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(d, 1),
        )
        nn.init.constant_(self.nei_relevance[-1].bias, -2.0)

        self.nei_dropout = nn.Dropout(gate_dropout)
        if self.use_bilstm:
            h = lstm_hidden_dim or (d // 2)
            self.nei_lstm = nn.LSTM(
                input_size=d,
                hidden_size=h,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
            lstm_out_dim = h * 2
            self.nei_lstm_proj = nn.Linear(lstm_out_dim, d) if lstm_out_dim != d else nn.Identity()
        else:
            self.nei_lstm = None
            self.nei_lstm_proj = None

        self.ctx_norm = nn.LayerNorm(d)
        self.target_anchor = nn.Parameter(torch.zeros(d))
        self.target_token_scale = nn.Parameter(torch.tensor(1.0))

        self.ctx_decoder = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(2 * d, d),
        )

        gate_stats_dim = 5
        gate_in_dim = 2 * C + gate_stats_dim
        gate_out_dim = C if self.per_class_gate else 1

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, gate_out_dim),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

    @staticmethod
    def _batch_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        exp_shape = list(idx.shape) + list(x.shape[2:])
        idx_exp = idx.view(idx.size(0), idx.size(1), *([1] * (x.dim() - 2))).expand(*exp_shape)
        return x.gather(1, idx_exp)

    def _mean_bernoulli_entropy(self, p: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        p32 = p.float().clamp(eps, 1.0 - eps)
        H = -(p32 * torch.log(p32) + (1.0 - p32) * torch.log(1.0 - p32))
        if p.dim() == 3:
            return H.mean(dim=-1).to(p.dtype)
        if p.dim() == 2:
            return H.mean(dim=-1, keepdim=True).to(p.dtype)
        raise ValueError("p must be (B,N,C) or (B,C)")

    def forward(
        self,
        target_logits: torch.Tensor,
        neighbor_logits: torch.Tensor = None,
        neighbor_pos: torch.Tensor = None,
        neighbor_valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if neighbor_logits is None:
            return target_logits
        if neighbor_logits.numel() == 0 or neighbor_logits.size(1) == 0:
            return target_logits

        B, N, C = neighbor_logits.shape
        if C != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {C}")

        device = target_logits.device
        dtype_out = target_logits.dtype
        d = self.label_emb_dim

        if neighbor_pos is None:
            neighbor_pos = torch.zeros((N,), device=device, dtype=torch.long)
        else:
            neighbor_pos = neighbor_pos.to(device=device, dtype=torch.long)

        if neighbor_pos.dim() == 1:
            pos = neighbor_pos[None, :].expand(B, -1)
        elif neighbor_pos.dim() == 2:
            pos = neighbor_pos
        else:
            raise ValueError("neighbor_pos must be (N,) or (B, N)")

        sort_idx = pos.argsort(dim=1)
        pos = pos.gather(1, sort_idx)

        if neighbor_valid_mask is None:
            neighbor_valid = torch.ones((B, N), device=device, dtype=torch.bool)
        else:
            neighbor_valid_mask = neighbor_valid_mask.to(device=device, dtype=torch.bool)
            if neighbor_valid_mask.dim() == 1:
                neighbor_valid = neighbor_valid_mask[None, :].expand(B, -1)
            elif neighbor_valid_mask.dim() == 2:
                neighbor_valid = neighbor_valid_mask
            else:
                raise ValueError("neighbor_valid_mask must be (N,) or (B, N)")
            if neighbor_valid.shape != (B, N):
                raise ValueError(f"neighbor_valid_mask must have shape {(B, N)}, got {tuple(neighbor_valid.shape)}")
            neighbor_valid = neighbor_valid.gather(1, sort_idx)
        has_context_neighbor = neighbor_valid.any(dim=1)

        pos_idx = (pos + self.max_k).clamp(0, 2 * self.max_k)
        pos_vec = self.pos_emb(pos_idx)
        pos_strength = torch.sigmoid(self.pos_alpha(pos_idx)).squeeze(-1)

        neighbor_logits_used = neighbor_logits.detach() if self.detach_neighbors else neighbor_logits
        neighbor_logits_used = self._batch_gather(neighbor_logits_used, sort_idx)

        target_logits_f = target_logits.to(torch.float32)
        neighbor_logits_f = neighbor_logits_used.to(torch.float32)

        if self.detach_target_for_context:
            target_logits_ctx = target_logits.detach()
        else:
            target_logits_ctx = target_logits

        target_p_ctx = torch.sigmoid(target_logits_ctx.to(torch.float32))
        E_in = self.label_emb_in.weight
        target_vec = torch.matmul(target_p_ctx, E_in)
        target_vec_norm = self.nei_norm(target_vec)

        p = torch.sigmoid(neighbor_logits_f)
        neigh_entropy = self._mean_bernoulli_entropy(p)
        m = torch.matmul(p, E_in)

        if self.training and self.neighbor_dropout > 0.0:
            keep_bool = (torch.rand((B, N), device=device) > self.neighbor_dropout) & neighbor_valid
            none_keep = has_context_neighbor & ~keep_bool.any(dim=1)
            if none_keep.any():
                random_scores = torch.rand((B, N), device=device).masked_fill(~neighbor_valid, -1.0)
                ridx = random_scores[none_keep].argmax(dim=1)
                keep_bool[none_keep, ridx] = True
            keep = keep_bool.to(torch.float32)
        else:
            keep = neighbor_valid.to(torch.float32)

        m_norm = self.nei_norm(m)
        target_exp = target_vec_norm.unsqueeze(1).expand(B, N, -1)
        pos_vec = pos_vec.to(m_norm.dtype)
        neigh_entropy_f = neigh_entropy.to(m_norm.dtype)

        rel_in = torch.cat(
            [m_norm, pos_vec, target_exp, neigh_entropy_f.unsqueeze(-1)],
            dim=-1,
        )
        rel = torch.sigmoid(self.nei_relevance(rel_in)).squeeze(-1)
        alpha = pos_strength.to(torch.float32) * rel.to(torch.float32) * keep

        valid = (alpha > 0) & neighbor_valid
        none_valid = has_context_neighbor & ~valid.any(dim=1)
        no_context = ~has_context_neighbor

        fallback_needed = none_valid | no_context
        masked_pos_strength = pos_strength.masked_fill(~neighbor_valid, -1.0)
        strongest_valid_idx = masked_pos_strength.argmax(dim=1)
        zero_idx = torch.zeros_like(strongest_valid_idx)
        fallback_idx = torch.where(no_context, zero_idx, strongest_valid_idx)
        fallback_valid = torch.zeros_like(valid).scatter(
            1, fallback_idx.unsqueeze(1), fallback_needed.unsqueeze(1)
        )
        fallback_alpha = torch.zeros_like(alpha).scatter(
            1,
            fallback_idx.unsqueeze(1),
            (fallback_needed.to(alpha.dtype) * float(self.eps)).unsqueeze(1),
        )
        valid = valid | fallback_valid
        alpha = alpha + fallback_alpha

        x_nei = self.nei_norm(m + pos_vec)
        pos0_idx = torch.full((B, 1), self.max_k, device=device, dtype=torch.long)
        pos0_vec = self.pos_emb(pos0_idx)

        x_tgt = (self.target_token_scale.to(torch.float32) * target_vec.to(torch.float32)).unsqueeze(1)
        x_tgt = x_tgt + pos0_vec.to(torch.float32) + self.target_anchor.to(torch.float32).view(1, 1, -1)
        x_tgt = self.nei_norm(x_tgt)

        pos_all = torch.cat([pos, torch.zeros((B, 1), device=device, dtype=torch.long)], dim=1)
        x_all = torch.cat([x_nei, x_tgt], dim=1)
        alpha_all = torch.cat(
            [alpha, torch.full((B, 1), float(self.eps), device=device, dtype=torch.float32)],
            dim=1,
        )
        valid_all = torch.cat([valid, torch.zeros((B, 1), device=device, dtype=torch.bool)], dim=1)
        is_target_all = torch.cat(
            [
                torch.zeros((B, N), device=device, dtype=torch.bool),
                torch.ones((B, 1), device=device, dtype=torch.bool),
            ],
            dim=1,
        )

        sort_all = pos_all.argsort(dim=1)
        pos_all = pos_all.gather(1, sort_all)
        x_all = self._batch_gather(x_all, sort_all)
        alpha_all = alpha_all.gather(1, sort_all)
        valid_all = valid_all.gather(1, sort_all)
        is_target_all = is_target_all.gather(1, sort_all)

        encoder_valid = valid_all | is_target_all
        x_all = x_all * encoder_valid.unsqueeze(-1).to(dtype=x_all.dtype)
        x_all_dropped = self.nei_dropout(x_all)
        x_all = torch.where(is_target_all.unsqueeze(-1), x_all, x_all_dropped)

        if self.use_bilstm:
            x_ctx_all, _ = self.nei_lstm(x_all)
            x_ctx_all = self.nei_lstm_proj(x_ctx_all)
        else:
            x_ctx_all = x_all

        tgt_idx = is_target_all.to(torch.int64).argmax(dim=1)
        batch_idx = torch.arange(B, device=device)
        x_tgt_enc = x_ctx_all[batch_idx, tgt_idx]

        pool_logits = (x_ctx_all.to(torch.float32) * x_tgt_enc.unsqueeze(1).to(torch.float32)).sum(dim=-1)
        pool_logits = pool_logits / math.sqrt(max(d, 1))
        pool_logits = pool_logits + torch.log(alpha_all.to(torch.float32).clamp(min=self.eps))
        pool_logits = pool_logits.masked_fill(~valid_all, float("-inf"))

        attn_w = torch.softmax(pool_logits, dim=1).to(torch.float32)
        m_ctx = torch.sum(attn_w.unsqueeze(-1) * x_ctx_all, dim=1)
        m_ctx = self.ctx_norm(m_ctx)

        hidden_ctx = self.ctx_decoder(m_ctx)
        E_out = self.label_emb_out.weight
        logits_ctx = torch.matmul(hidden_ctx, E_out.t()) + self.out_bias

        if self.detach_target_for_gate:
            t_in = target_logits.detach()
        else:
            t_in = target_logits

        target_p = torch.sigmoid(t_in.to(torch.float32))
        target_entropy = self._mean_bernoulli_entropy(target_p)

        valid_count = valid.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
        ent_for_mean = neigh_entropy.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / valid_count
        ent_for_max = neigh_entropy.masked_fill(~valid, -1e9).max(dim=1, keepdim=True).values
        ent_for_min = neigh_entropy.masked_fill(~valid, 1e9).min(dim=1, keepdim=True).values
        alpha_sum = alpha.sum(dim=1, keepdim=True).to(torch.float32)

        gate_stats = torch.cat(
            [
                target_entropy.to(torch.float32),
                ent_for_mean.to(torch.float32),
                ent_for_max.to(torch.float32),
                ent_for_min.to(torch.float32),
                alpha_sum,
            ],
            dim=-1,
        )

        gate_logits_ctx = logits_ctx.detach() if self.detach_target_for_gate else logits_ctx
        gate_in = torch.cat([t_in.to(torch.float32), gate_logits_ctx.to(torch.float32), gate_stats], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))
        if gate.size(-1) == 1:
            gate = gate.expand(-1, C)

        logits_final = target_logits_f + gate.to(torch.float32) * logits_ctx.to(torch.float32)
        if no_context.any():
            logits_final = torch.where(no_context.unsqueeze(-1), target_logits_f, logits_final)
        return logits_final.to(dtype_out)


class AdapterBlock(nn.Module):
    """
    Builds per-label classifiers for label-specific embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.layer_norm = nn.LayerNorm(input_dim)
        self.classifiers = nn.ModuleList(
            [MLPLayer(input_dim, hidden_dim, 1, dropout=dropout) for _ in range(num_classes)]
        )

    def forward(self, label_emb: torch.Tensor) -> torch.Tensor:
        """
        label_emb: (B, C, H), one embedding per functional label
        """
        if label_emb.size(1) != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} label embeddings, got {label_emb.size(1)}")

        label_emb = self.layer_norm(label_emb)
        per_label_logits = [clf(label_emb[:, i, :]) for i, clf in enumerate(self.classifiers)]
        logits = torch.cat(per_label_logits, dim=-1)
        return logits

class AnnotModel(nn.Module):
    def __init__(self, esmc_model_name: str, num_classes: int, device: torch.device):
        super().__init__()
        if isinstance(device, torch.device) and device.type == 'cuda':
            with torch.cuda.device(device):
                self.esmc_model = ESMC.from_pretrained(esmc_model_name)
        else:
            self.esmc_model = ESMC.from_pretrained(esmc_model_name)
        self.esmc_model.to(device)

        self.esmc_embed_dim = 1152
        self.num_heads = 8
        self.ff_dim = self.esmc_embed_dim * 4

        for param in self.esmc_model.parameters():
            param.requires_grad = False
        self.esmc_model.eval()

        self.hidden_weighting_layer1 = nn.Linear(36, 1, bias=False)
        self.hidden_weighting_layer2 = nn.Linear(36, 1, bias=False)
        self.domain_block = DomainBoundaryBiLSTM(self.esmc_embed_dim)

        self.label_attention_pooling = LabelWiseAttentionPooling(
            hidden_dim=self.esmc_embed_dim,
            num_classes=num_classes,
            num_heads=self.num_heads,
        )
        self.label_gate = nn.Parameter(
            torch.full((self.esmc_embed_dim,), -2.0)
        )

        hidden_dim = self.esmc_embed_dim // 2
        self.adapter_block = AdapterBlock(self.esmc_embed_dim, hidden_dim, num_classes, dropout=0.1)

        self.context_calibrator = LabelContextCalibrator(
            num_classes=num_classes,
            label_emb_dim=128,
            max_k=16,
            neighbor_dropout=0.2,
            gate_hidden_dim=128,
            gate_dropout=0.1,
            per_class_gate=True,
            detach_neighbors=True,
            detach_target_for_gate=True,
            detach_target_for_context=True,
        )
        self.use_context = True

    def set_use_context(self, use_context: bool):
        self.use_context = bool(use_context)

    def freeze_except_context(self):
        for name, p in self.named_parameters():
            if name.startswith("context_calibrator"):
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.esmc_model.eval()

    def get_residue_embed(self, outputs, mask_tokens_ids):
        hidden_states = outputs.hidden_states
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.permute(1, 2, 3, 0).contiguous()

        domain_embed = self.hidden_weighting_layer1(hidden_states).squeeze(-1)
        domain_weights = self.domain_block(domain_embed, mask_tokens_ids)

        prot_embed = self.hidden_weighting_layer2(hidden_states).squeeze(-1)
        residue_mask = _build_residue_mask(prot_embed, mask_tokens_ids)
        residue_emb = torch.nan_to_num(prot_embed, nan=0.0, posinf=0.0, neginf=0.0)
        residue_weights = torch.nan_to_num(domain_weights, nan=0.0, posinf=0.0, neginf=0.0)
        return domain_weights, residue_emb, residue_mask, residue_weights

    def _label_embedding_from_outputs(self, outputs, mask_tokens_ids):
        domain_weights, residue_emb, residue_mask, residue_weights = self.get_residue_embed(outputs, mask_tokens_ids)
        mean_embed = _masked_mean_pool(residue_emb, residue_mask)
        label_emb = self.label_attention_pooling(
            residue_emb=residue_emb,
            residue_mask=residue_mask,
            residue_weights=residue_weights,
        )
        gate = torch.sigmoid(self.label_gate).view(1, 1, -1)
        label_emb = mean_embed.unsqueeze(1) + gate * label_emb
        return domain_weights, label_emb

    def extract_embedding(self, prot_tensor, mask_tokens_ids):
        """
        Return the final label-specific target-sequence embeddings.

        Shape: (batch_size, num_classes, esmc_embed_dim).  This is the same
        ``label_emb`` tensor consumed by ``adapter_block`` during inference.
        """
        outputs = self.esmc_model(prot_tensor, mask_tokens_ids)
        _, label_emb = self._label_embedding_from_outputs(outputs, mask_tokens_ids)
        return label_emb

    def _sequence_expert_logits(self, outputs, mask_tokens_ids):
        domain_weights, label_emb = self._label_embedding_from_outputs(outputs, mask_tokens_ids)
        protein_logits = self.adapter_block(label_emb)
        return domain_weights, protein_logits

    @staticmethod
    def _valid_protein_mask_from_tokens(mask_tokens_ids: torch.Tensor) -> torch.Tensor:
        if mask_tokens_ids is None:
            return None

        mask = mask_tokens_ids.bool()
        original_shape = mask.shape[:-1]
        L = mask.size(-1)
        flat_mask = mask.reshape(-1, L)
        residue_mask = flat_mask.clone()

        if L >= 1:
            residue_mask[:, 0] = False
            reversed_mask = torch.flip(flat_mask, dims=[1])
            last_true_from_end = torch.argmax(reversed_mask.to(torch.int64), dim=1)
            last_true_idx = (L - 1) - last_true_from_end
            has_true = flat_mask.any(dim=1)
            flat_indices = torch.arange(flat_mask.size(0), device=flat_mask.device)
            residue_mask[flat_indices[has_true], last_true_idx[has_true]] = False

        return residue_mask.any(dim=1).reshape(*original_shape)

    @staticmethod
    def _masked_pool_neighbor_logits(neighbor_logits: torch.Tensor, neighbor_valid_mask: torch.Tensor):
        B, K, C = neighbor_logits.shape
        if K == 0:
            return neighbor_logits.new_zeros((B, C))
        if neighbor_valid_mask is None:
            return neighbor_logits.max(dim=1).values

        valid = neighbor_valid_mask.to(device=neighbor_logits.device, dtype=torch.bool)
        masked_logits = neighbor_logits.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = masked_logits.max(dim=1).values
        return torch.nan_to_num(pooled, neginf=0.0, posinf=0.0)

    def _protein_logits_from_tokens(self, prot_tensor: torch.Tensor, mask_tokens_ids: torch.Tensor):
        if prot_tensor.dim() == 2:
            outputs = self.esmc_model(prot_tensor, mask_tokens_ids)
            _, logits = self._sequence_expert_logits(outputs, mask_tokens_ids)
            return logits.unsqueeze(1)

        if prot_tensor.dim() == 3:
            B, K, L = prot_tensor.shape
            if K == 0:
                return prot_tensor.new_zeros((B, 0, self.adapter_block.num_classes), dtype=torch.float32)
            flat = prot_tensor.reshape(B * K, L)
            flat_mask = mask_tokens_ids.reshape(B * K, L) if mask_tokens_ids is not None else None
            logits_flat = flat.new_zeros((B * K, self.adapter_block.num_classes), dtype=torch.float32)

            if flat_mask is None:
                valid_flat = torch.ones((B * K,), device=flat.device, dtype=torch.bool)
            else:
                valid_flat = self._valid_protein_mask_from_tokens(flat_mask)

            if valid_flat.any():
                valid_tokens = flat[valid_flat]
                valid_mask = flat_mask[valid_flat] if flat_mask is not None else None
                outputs = self.esmc_model(valid_tokens, valid_mask)
                _, logits_valid = self._sequence_expert_logits(outputs, valid_mask)
                logits_flat[valid_flat] = logits_valid.to(logits_flat.dtype)
            return logits_flat.reshape(B, K, -1)

        raise ValueError("prot_tensor must be (B,L) or (B,K,L)")

    def forward(
        self,
        prot_tensor,
        prot_tensor_left,
        prot_tensor_right,
        mask_tokens_ids,
        mask_tokens_ids_left,
        mask_tokens_ids_right,
    ):
        outputs = self.esmc_model(prot_tensor, mask_tokens_ids)
        domain_weights, target_logits_seq = self._sequence_expert_logits(outputs, mask_tokens_ids)

        left_logits_all = self._protein_logits_from_tokens(prot_tensor_left, mask_tokens_ids_left)
        right_logits_all = self._protein_logits_from_tokens(prot_tensor_right, mask_tokens_ids_right)

        left_valid = self._valid_protein_mask_from_tokens(mask_tokens_ids_left)
        right_valid = self._valid_protein_mask_from_tokens(mask_tokens_ids_right)
        if left_valid is not None and left_valid.dim() == 1:
            left_valid = left_valid.unsqueeze(1)
        if right_valid is not None and right_valid.dim() == 1:
            right_valid = right_valid.unsqueeze(1)

        left_logits_pooled = self._masked_pool_neighbor_logits(left_logits_all, left_valid)
        right_logits_pooled = self._masked_pool_neighbor_logits(right_logits_all, right_valid)

        neighbor_logits = torch.cat([left_logits_all, right_logits_all], dim=1)
        device = target_logits_seq.device
        if left_valid is None:
            left_valid = torch.ones(left_logits_all.shape[:2], device=device, dtype=torch.bool)
        if right_valid is None:
            right_valid = torch.ones(right_logits_all.shape[:2], device=device, dtype=torch.bool)
        neighbor_valid_mask = torch.cat([left_valid, right_valid], dim=1)
        Nl = left_logits_all.size(1)
        Nr = right_logits_all.size(1)

        pos_left = -torch.arange(Nl, 0, -1, device=device, dtype=torch.long) if Nl > 0 else torch.zeros((0,), device=device, dtype=torch.long)
        pos_right = torch.arange(1, Nr + 1, device=device, dtype=torch.long) if Nr > 0 else torch.zeros((0,), device=device, dtype=torch.long)
        neighbor_pos = torch.cat([pos_left, pos_right], dim=0)

        if self.use_context:
            protein_function_logits = self.context_calibrator(
                target_logits=target_logits_seq,
                neighbor_logits=neighbor_logits,
                neighbor_pos=neighbor_pos,
                neighbor_valid_mask=neighbor_valid_mask,
            )
        else:
            protein_function_logits = target_logits_seq

        return domain_weights, protein_function_logits, left_logits_pooled, right_logits_pooled
