"""Feature projection, temporal experts, cross-attention, and auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


EXPERT_NAMES = (
    "text",
    "audio",
    "cross_modal",
    "speaker_low",
    "speaker_high",
)


@dataclass(frozen=True)
class SAGERAuxiliaryLoss:
    """Auxiliary representation and routing losses."""

    total: Tensor
    supervised_contrastive: Tensor
    directional_kl: Tensor
    spectral: Tensor
    router_balance: Tensor


def _zero_masked(tensor: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < tensor.ndim:
        expanded = expanded.unsqueeze(-1)
    return torch.where(expanded, tensor, torch.zeros_like(tensor))


class _TokenAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_count: int,
        hidden_dim: int,
        dropout: float,
        *,
        audio_layout: bool = False,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.audio_layout = audio_layout
        if audio_layout:
            if token_count != 27:
                raise ValueError("the registered audio layout is 3 layers x 9 statistics")
            self.layer_embedding = nn.Parameter(torch.empty(3, hidden_dim))
            self.statistic_embedding = nn.Parameter(torch.empty(9, hidden_dim))
            nn.init.normal_(self.layer_embedding, std=0.02)
            nn.init.normal_(self.statistic_embedding, std=0.02)
        else:
            self.token_embedding = nn.Parameter(torch.empty(token_count, hidden_dim))
            nn.init.normal_(self.token_embedding, std=0.02)
        self.token_score = nn.Linear(hidden_dim, 1)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, available: Tensor) -> Tensor:
        tokens = _zero_masked(tokens, available)
        hidden = self.projection(self.input_norm(tokens))
        if self.audio_layout:
            type_embedding = (
                self.layer_embedding[:, None, :] + self.statistic_embedding[None, :, :]
            ).reshape(27, -1)
        else:
            type_embedding = self.token_embedding
        hidden = F.gelu(hidden + type_embedding.view(1, 1, self.token_count, -1))
        token_weights = torch.softmax(self.token_score(hidden).squeeze(-1), dim=-1)
        pooled = torch.sum(token_weights.unsqueeze(-1) * hidden, dim=-2)
        pooled = self.output_norm(self.dropout(pooled))
        return _zero_masked(pooled, available)


class _SparseTemporalMoE(nn.Module):
    """Top-k mixture of convolutions with genuinely different context widths."""

    def __init__(
        self,
        hidden_dim: int,
        kernels: tuple[int, ...],
        top_k: int,
        dropout: float,
        quality_dim: int = 0,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.quality_dim = quality_dim
        self.experts = nn.ModuleList(
            nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2)
            for kernel in kernels
        )
        self.expert_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in kernels)
        if quality_dim:
            self.quality_norm = nn.LayerNorm(quality_dim)
        self.router = nn.Linear(hidden_dim + quality_dim, len(kernels))
        self.dialogue_gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        active_mask: Tensor,
        utterance_mask: Tensor,
        quality: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden = _zero_masked(hidden, active_mask)
        convolution_input = hidden.transpose(1, 2)
        expert_states = []
        for convolution, norm in zip(self.experts, self.expert_norms):
            state = convolution(convolution_input).transpose(1, 2)
            state = norm(hidden + self.dropout(F.gelu(state)))
            expert_states.append(_zero_masked(state, active_mask))
        stacked = torch.stack(expert_states, dim=-2)

        router_parts = [hidden]
        if self.quality_dim:
            if quality is None:
                raise ValueError("quality features are required by this temporal router")
            clean_quality = _zero_masked(quality, active_mask)
            clean_quality = _zero_masked(self.quality_norm(clean_quality), active_mask)
            router_parts.append(clean_quality)
        router_logits = self.router(torch.cat(router_parts, dim=-1))
        top_values, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        router_weights = torch.zeros_like(router_logits).scatter(
            -1, top_indices, top_weights
        )
        router_weights = _zero_masked(router_weights, active_mask)
        mixture = torch.sum(stacked * router_weights.unsqueeze(-1), dim=-2)

        lengths = utterance_mask.sum(dim=1).to(torch.long).cpu()
        packed = pack_padded_sequence(
            mixture,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _ = self.dialogue_gru(packed)
        dialogue_context, _ = pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=hidden.shape[1],
        )
        output = self.output_norm(mixture + self.dropout(dialogue_context))
        return _zero_masked(output, active_mask), router_weights


def _safe_sequence_attention(
    attention: nn.MultiheadAttention,
    query: Tensor,
    source: Tensor,
    source_mask: Tensor,
) -> Tensor:
    """Run attention only for dialogues with at least one unmasked key."""

    has_source = source_mask.any(dim=1)
    if not bool(has_source.any().item()):
        return torch.zeros_like(query)
    indices = has_source.nonzero(as_tuple=False).squeeze(-1)
    attended, _ = attention(
        query.index_select(0, indices),
        source.index_select(0, indices),
        source.index_select(0, indices),
        key_padding_mask=~source_mask.index_select(0, indices),
        need_weights=False,
    )
    return torch.zeros_like(query).index_copy(0, indices, attended)


class _TextAudioCrossAttention(nn.Module):
    """One text-query/audio-key-value attention layer."""
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: Tensor, audio: Tensor, utterance_mask: Tensor, audio_mask: Tensor):
        update = _safe_sequence_attention(self.attention, text, audio, audio_mask)
        return _zero_masked(self.norm(text + self.dropout(update)), utterance_mask)


def masked_focal_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    gamma: float = 2.0,
) -> Tensor:
    """Numerically stable focal loss over explicitly selected rows."""

    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("focal logits, targets, and mask shapes do not align")
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    if selected_targets.numel() == 0:
        return logits.sum() * 0.0
    log_probability = F.log_softmax(selected_logits, dim=-1)
    selected_log_probability = log_probability.gather(
        1, selected_targets.unsqueeze(1)
    ).squeeze(1)
    probability = selected_log_probability.exp()
    loss = -(1.0 - probability).pow(gamma) * selected_log_probability
    return loss.mean()


def padding_safe_supervised_contrastive_loss(
    first_view: Tensor,
    second_view: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """Two-view supervised contrastive loss with padding removed up front."""

    if first_view.shape != second_view.shape:
        raise ValueError("contrastive views must have identical shapes")
    if first_view.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("contrastive views, targets, and mask shapes do not align")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    active = mask & (targets >= 0)
    if not bool(active.any().item()):
        return (first_view.sum() + second_view.sum()) * 0.0
    first = F.normalize(first_view[active], dim=-1)
    second = F.normalize(second_view[active], dim=-1)
    labels = targets[active]
    representations = torch.cat((first, second), dim=0)
    labels = torch.cat((labels, labels), dim=0)
    similarity = representations @ representations.transpose(0, 1)
    similarity = similarity / temperature
    count = similarity.shape[0]
    diagonal = torch.eye(count, dtype=torch.bool, device=similarity.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    denominator_logits = similarity.masked_fill(diagonal, -torch.inf)
    log_probability = similarity - torch.logsumexp(
        denominator_logits, dim=-1, keepdim=True
    )
    positive_count = positive.sum(dim=-1)
    usable = positive_count > 0
    per_anchor = -(
        torch.where(positive, log_probability, torch.zeros_like(log_probability)).sum(
            dim=-1
        )
        / positive_count.clamp_min(1)
    )
    if not bool(usable.any().item()):
        return representations.sum() * 0.0
    return per_anchor[usable].mean()


def directional_kl_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    mask: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """KL(teacher || student), with a stopped teacher direction."""

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if teacher_logits.shape[:-1] != mask.shape:
        raise ValueError("directional KL mask shape does not align")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not bool(mask.any().item()):
        return (teacher_logits.sum() + student_logits.sum()) * 0.0
    teacher_probability = torch.softmax(
        teacher_logits[mask].detach() / temperature, dim=-1
    )
    student_log_probability = torch.log_softmax(
        student_logits[mask] / temperature, dim=-1
    )
    return (
        F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="none",
        )
        .sum(dim=-1)
        .mean()
        * temperature**2
    )


def _spectral_separation_loss(output: Any) -> Tensor:
    mask = output.availability_for("speaker_low")
    if not bool(mask.any().item()):
        return (
            output.speaker_low_representation.sum()
            + output.speaker_high_representation.sum()
        ) * 0.0
    low = F.normalize(output.speaker_low_representation[mask], dim=-1)
    high = F.normalize(output.speaker_high_representation[mask], dim=-1)
    return torch.sum(low * high, dim=-1).pow(2).mean()


def _router_balance_loss(output: Any) -> Tensor:
    losses = []
    for weights, mask in (
        (output.text_router_weights, output.utterance_mask),
        (output.audio_router_weights, output.availability_for("audio")),
    ):
        if bool(mask.any().item()):
            load = weights[mask].mean(dim=0)
            target = torch.full_like(load, 1.0 / load.numel())
            losses.append((load - target).pow(2).mean())
    if not losses:
        return output.logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_auxiliary_objective(
    output: Any,
    targets: Tensor,
    *,
    supcon_weight: float = 1.0,
    directional_kl_weight: float = 0.001,
    spectral_weight: float = 0.10,
    router_balance_weight: float = 0.02,
    temperature: float = 0.05,
) -> SAGERAuxiliaryLoss:
    """Compute all supervised terms outside the model's forward path."""

    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long")
    if targets.shape != output.utterance_mask.shape:
        raise ValueError("targets must have shape [batch, turns]")
    if targets.device != output.logits.device:
        raise ValueError("targets and SAGER output must be on the same device")
    valid_targets = targets[output.utterance_mask]
    num_classes = output.logits.shape[-1]
    if num_classes < 2:
        raise ValueError("SAGER output must contain at least two classes")
    if output.anchor_logits.shape[-1] != num_classes or output.expert_logits.shape[-1] != num_classes:
        raise ValueError("SAGER output class dimensions do not align")
    if bool(((valid_targets < 0) | (valid_targets >= num_classes)).any().item()):
        raise ValueError(
            f"valid targets must be in [0, {num_classes - 1}]"
        )

    contrastive_losses = []
    for representation, mask in (
        (output.audio_representation, output.availability_for("audio")),
        (
            output.cross_modal_representation,
            output.availability_for("cross_modal"),
        ),
    ):
        if bool(mask.any().item()):
            contrastive_losses.append(
                padding_safe_supervised_contrastive_loss(
                    output.text_representation,
                    representation,
                    targets,
                    mask,
                    temperature=temperature,
                )
            )
    supervised_contrastive = (
        torch.stack(contrastive_losses).mean()
        if contrastive_losses
        else output.logits.sum() * 0.0
    )

    cross_logits = output.logits_for("cross_modal")
    kl_losses = []
    for name in ("audio",):
        mask = output.availability_for(name) & output.availability_for("cross_modal")
        if bool(mask.any().item()):
            kl_losses.append(
                directional_kl_loss(
                    cross_logits,
                    output.logits_for(name),
                    mask,
                )
            )
    directional_kl = (
        torch.stack(kl_losses).mean()
        if kl_losses
        else output.logits.sum() * 0.0
    )
    spectral = _spectral_separation_loss(output)
    router_balance = _router_balance_loss(output)
    total = (
        supcon_weight * supervised_contrastive
        + directional_kl_weight * directional_kl
        + spectral_weight * spectral
        + router_balance_weight * router_balance
    )
    return SAGERAuxiliaryLoss(
        total=total,
        supervised_contrastive=supervised_contrastive,
        directional_kl=directional_kl,
        spectral=spectral,
        router_balance=router_balance,
    )


