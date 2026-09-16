"""SAGER backbone: counterfactual anchor-override logit-mixture."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .components import (
    EXPERT_NAMES,
    _TextAudioCrossAttention,
    _SparseTemporalMoE,
    _TokenAdapter,
    _zero_masked,
    compute_auxiliary_objective,
    masked_focal_loss,
)


RELATION_NAMES = (
    "same_speaker_history",
    "cross_speaker_reply",
    "local_adjacent",
    "long_distance_context",
)


@dataclass(frozen=True)
class SAGERBackboneConfig:
    num_classes: int
    hidden_dim: int = 192
    dropout: float = 0.10
    temporal_kernels: tuple[int, ...] = (1, 3, 5)
    top_k_experts: int = 2
    cross_modal_layers: int = 2
    attention_heads: int = 4
    relation_count: int = 4
    graph_band_count: int = 2
    graph_long_distance_minimum: int = 2
    graph_distance_decay: float = 0.25
    stable_graph_layers: int = 2
    stable_graph_alpha: float = 0.10
    stable_graph_lambda: float = 0.50
    utility_gate_scale: float = 1.50
    correction_gate_initial_bias: float = -2.0
    state_space_hidden_dim: int = 192
    change_boundary_hidden_dim: int = 96
    change_reset_initial_bias: float = -1.0
    path_local_scales: tuple[int, ...] = (1, 2, 4)
    tree_parent_top_k: int = 3
    tree_distance_decay: float = 0.08
    boundary_prior_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if len(self.temporal_kernels) < 3 or len(set(self.temporal_kernels)) != len(self.temporal_kernels):
            raise ValueError("three distinct temporal kernels are required")
        if any(k <= 0 or k % 2 == 0 for k in self.temporal_kernels):
            raise ValueError("temporal kernels must be positive odd integers")
        if not 1 <= self.top_k_experts <= len(self.temporal_kernels):
            raise ValueError("top_k_experts is invalid")
        if self.cross_modal_layers != 2:
            raise ValueError("The paper uses two cross-attention layers")
        if self.relation_count != len(RELATION_NAMES) or self.graph_band_count != 2:
            raise ValueError("Typed Graph is four relations x low/high")
        if self.graph_long_distance_minimum < 2 or self.graph_distance_decay <= 0:
            raise ValueError("invalid long-distance graph constants")
        if self.stable_graph_layers < 1 or not 0.0 < self.stable_graph_alpha < 1.0:
            raise ValueError("invalid stable graph depth/alpha")
        if self.stable_graph_lambda <= 0.0:
            raise ValueError("stable graph lambda must be positive")
        if self.utility_gate_scale < 0:
            raise ValueError("utility_gate_scale must be nonnegative")
        if not math.isfinite(self.correction_gate_initial_bias):
            raise ValueError("correction gate initial bias must be finite")
        if self.state_space_hidden_dim != self.hidden_dim:
            raise ValueError("Memory width equals the SAGER hidden width")
        if self.change_boundary_hidden_dim != self.hidden_dim // 2:
            raise ValueError("Memory boundary head width is half the SAGER hidden width")
        if not math.isfinite(self.change_reset_initial_bias):
            raise ValueError("change reset initial bias must be finite")
        if self.path_local_scales != (1, 2, 4):
            raise ValueError("Local path scales must be one/two/four hops")
        if self.tree_parent_top_k != 3 or self.tree_distance_decay <= 0:
            raise ValueError("invalid context-path constants")
        if self.boundary_prior_scale < 0 or not math.isfinite(self.boundary_prior_scale):
            raise ValueError("invalid boundary prior scale")


@dataclass(frozen=True)
class SAGERInputs:
    text_cls: Tensor
    text_anchor_logits: Tensor
    audio_tokens: Tensor
    audio_quality: Tensor
    audio_available: Tensor
    utterance_mask: Tensor
    speaker_index: Tensor
    temporal_position: Tensor


@dataclass(frozen=True)
class SAGERBackboneOutput:
    logits: Tensor
    anchor_logits: Tensor
    expert_logits: Tensor
    expert_available: Tensor
    correction_strength: Tensor
    correction_evidence_logits: Tensor
    override_logit: Tensor
    text_representation: Tensor
    audio_representation: Tensor
    cross_modal_representation: Tensor
    speaker_low_representation: Tensor
    speaker_high_representation: Tensor
    predicted_audio_utility: Tensor
    change_boundary_logits: Tensor
    change_state_available: Tensor
    speaker_index: Tensor
    text_router_weights: Tensor
    audio_router_weights: Tensor
    utterance_mask: Tensor
    base_evidence_logits: Tensor
    relational_view_logits: Tensor
    relational_consensus_logits: Tensor
    relational_view_disagreement: Tensor
    relational_verifier_logit: Tensor
    relational_verifier_probability: Tensor
    expert_names: tuple[str, ...] = EXPERT_NAMES

    def logits_for(self, expert_name: str) -> Tensor:
        try:
            index = self.expert_names.index(expert_name)
        except ValueError as exc:
            raise KeyError(expert_name) from exc
        return self.expert_logits[..., index, :]

    def availability_for(self, expert_name: str) -> Tensor:
        try:
            index = self.expert_names.index(expert_name)
        except ValueError as exc:
            raise KeyError(expert_name) from exc
        return self.expert_available[..., index]


@dataclass(frozen=True)
class SAGERBackboneLoss:
    total: Tensor
    final_focal: Tensor
    expert_focal: Tensor
    supervised_contrastive: Tensor
    directional_kl: Tensor
    spectral: Tensor
    router_balance: Tensor
    counterfactual_utility: Tensor
    change_point: Tensor
    counterfactual_override: Tensor
    relational_proposal_focal: Tensor
    relational_view_consistency: Tensor
    relational_verifier: Tensor


def _require_shape(tensor: Tensor, expected: Sequence[int], name: str) -> None:
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(f"{name} must be {tuple(expected)}, got {tuple(tensor.shape)}")


def _validate_inputs(inputs: SAGERInputs, *, num_classes: int) -> tuple[int, int]:
    if inputs.text_cls.ndim != 3:
        raise ValueError("text_cls must have shape [batch, turns, 1024]")
    batch, turns = inputs.text_cls.shape[:2]
    shapes = {
        "text_cls": (batch, turns, 1024),
        "text_anchor_logits": (batch, turns, num_classes),
        "audio_tokens": (batch, turns, 27, 768),
        "audio_quality": (batch, turns, 6),
        "audio_available": (batch, turns),
        "utterance_mask": (batch, turns),
        "speaker_index": (batch, turns),
        "temporal_position": (batch, turns),
    }
    for field in fields(inputs):
        value = getattr(inputs, field.name)
        if not isinstance(value, Tensor):
            raise TypeError(f"{field.name} must be Tensor")
        _require_shape(value, shapes[field.name], field.name)
    for name in ("text_cls", "text_anchor_logits", "audio_tokens", "audio_quality"):
        value = getattr(inputs, name)
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains nonfinite values")
    for name in ("audio_available", "utterance_mask"):
        if getattr(inputs, name).dtype != torch.bool:
            raise TypeError(f"{name} must be bool")
    for name in ("speaker_index", "temporal_position"):
        if getattr(inputs, name).dtype != torch.long:
            raise TypeError(f"{name} must be long")
    if len({getattr(inputs, f.name).device for f in fields(inputs)}) != 1:
        raise ValueError("all inputs must share one device")
    if turns == 0 or not bool(inputs.utterance_mask.any(dim=1).all().item()):
        raise ValueError("each dialogue must contain a valid utterance")
    seen_padding = (~inputs.utterance_mask).to(torch.int64).cumsum(dim=1) > 0
    if bool((inputs.utterance_mask & seen_padding).any().item()):
        raise ValueError("dialogue masks must be prefix packed")
    if bool((inputs.audio_available & ~inputs.utterance_mask).any().item()):
        raise ValueError("audio mask outside dialogue")
    if bool((inputs.speaker_index[inputs.utterance_mask] < 0).any().item()) or bool((inputs.speaker_index[~inputs.utterance_mask] != -1).any().item()):
        raise ValueError("speaker indices violate padded dialogue contract")
    if bool((inputs.temporal_position[inputs.utterance_mask] < 0).any().item()):
        raise ValueError("negative temporal position")
    return batch, turns


class _ChangePointSpeakerDialogueStateSpace(nn.Module):
    """Bidirectional local and same-speaker memories with learned reset boundaries."""

    def __init__(self, cfg: SAGERBackboneConfig) -> None:
        super().__init__()
        hidden = cfg.hidden_dim
        boundary_input = 3 * hidden
        self.input_norm = nn.LayerNorm(hidden)
        self.local_cells = nn.ModuleDict({direction: nn.GRUCell(hidden, hidden) for direction in ("past", "future")})
        self.speaker_cells = nn.ModuleDict({direction: nn.GRUCell(hidden, hidden) for direction in ("past", "future")})
        self.local_boundary = nn.ModuleDict(
            {
                direction: nn.Sequential(
                    nn.LayerNorm(boundary_input),
                    nn.Linear(boundary_input, cfg.change_boundary_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.change_boundary_hidden_dim, 1),
                )
                for direction in ("past", "future")
            }
        )
        self.speaker_boundary = nn.ModuleDict(
            {
                direction: nn.Sequential(
                    nn.LayerNorm(boundary_input),
                    nn.Linear(boundary_input, cfg.change_boundary_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.change_boundary_hidden_dim, 1),
                )
                for direction in ("past", "future")
            }
        )
        for head in tuple(self.local_boundary.values()) + tuple(self.speaker_boundary.values()):
            nn.init.constant_(head[-1].bias, cfg.change_reset_initial_bias)
        fusion_width = 5 * hidden
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(fusion_width),
            nn.Linear(fusion_width, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 5),
        )
        self.output_projection = nn.Linear(hidden, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _boundary_features(current: Tensor, memory: Tensor) -> Tensor:
        return torch.cat((current, memory, torch.abs(current - memory)), dim=-1)

    def _directional_states(
        self,
        hidden: Tensor,
        mask: Tensor,
        speaker_index: Tensor,
        *,
        direction: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, turns, width = hidden.shape
        valid_speakers = speaker_index.masked_fill(~mask, 0)
        speaker_count = max(1, int(valid_speakers.max().detach().cpu().item()) + 1)
        local_memory = hidden.new_zeros((batch, width))
        speaker_memory = hidden.new_zeros((batch, speaker_count, width))
        local_states: list[Tensor | None] = [None] * turns
        speaker_states: list[Tensor | None] = [None] * turns
        local_logits: list[Tensor | None] = [None] * turns
        speaker_logits: list[Tensor | None] = [None] * turns
        order = range(turns) if direction == "past" else range(turns - 1, -1, -1)
        for turn in order:
            active = mask[:, turn]
            current = hidden[:, turn]
            safe_speaker = valid_speakers[:, turn]
            gather = safe_speaker.view(batch, 1, 1).expand(-1, 1, width)
            previous_speaker = speaker_memory.gather(1, gather).squeeze(1)
            local_logit = self.local_boundary[direction](self._boundary_features(current, local_memory)).squeeze(-1)
            speaker_logit = self.speaker_boundary[direction](self._boundary_features(current, previous_speaker)).squeeze(-1)
            reset_local = 1.0 - torch.sigmoid(local_logit)
            reset_speaker = 1.0 - torch.sigmoid(speaker_logit)
            next_local = self.local_cells[direction](current, reset_local.unsqueeze(-1) * local_memory)
            next_speaker = self.speaker_cells[direction](current, reset_speaker.unsqueeze(-1) * previous_speaker)
            next_local = torch.where(active.unsqueeze(-1), next_local, local_memory)
            next_speaker = torch.where(active.unsqueeze(-1), next_speaker, previous_speaker)
            selector = F.one_hot(safe_speaker, num_classes=speaker_count).to(hidden.dtype)
            selector = selector * active.to(hidden.dtype).unsqueeze(-1)
            speaker_memory = speaker_memory * (1.0 - selector.unsqueeze(-1)) + next_speaker.unsqueeze(1) * selector.unsqueeze(-1)
            local_memory = next_local
            local_states[turn] = torch.where(active.unsqueeze(-1), next_local, torch.zeros_like(next_local))
            speaker_states[turn] = torch.where(active.unsqueeze(-1), next_speaker, torch.zeros_like(next_speaker))
            local_logits[turn] = torch.where(active, local_logit, torch.zeros_like(local_logit))
            speaker_logits[turn] = torch.where(active, speaker_logit, torch.zeros_like(speaker_logit))
        return (
            torch.stack([value for value in local_states if value is not None], dim=1),
            torch.stack([value for value in speaker_states if value is not None], dim=1),
            torch.stack([value for value in local_logits if value is not None], dim=1),
            torch.stack([value for value in speaker_logits if value is not None], dim=1),
        )

    def forward(self, hidden: Tensor, mask: Tensor, speaker_index: Tensor) -> tuple[Tensor, Tensor]:
        clean = _zero_masked(self.input_norm(hidden), mask)
        forward_local, forward_speaker, forward_local_logit, forward_speaker_logit = self._directional_states(
            clean, mask, speaker_index, direction="past"
        )
        backward_local, backward_speaker, backward_local_logit, backward_speaker_logit = self._directional_states(
            clean, mask, speaker_index, direction="future"
        )
        branches = torch.stack((hidden, forward_local, backward_local, forward_speaker, backward_speaker), dim=-2)
        fusion_features = torch.cat((hidden, forward_local, backward_local, forward_speaker, backward_speaker), dim=-1)
        fusion_weights = _zero_masked(torch.softmax(self.fusion_gate(fusion_features), dim=-1), mask)
        context = torch.sum(fusion_weights.unsqueeze(-1) * branches, dim=-2)
        output = _zero_masked(self.output_norm(hidden + self.dropout(self.output_projection(context))), mask)
        boundary_logits = torch.stack(
            (forward_local_logit, backward_local_logit, forward_speaker_logit, backward_speaker_logit), dim=-1
        )
        boundary_logits = _zero_masked(boundary_logits, mask)
        return output, boundary_logits


class _RelationTypedSpectralGraph(nn.Module):
    def __init__(self, cfg: SAGERBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        quality_dim = 6 + 1
        self.quality_norm = nn.LayerNorm(quality_dim)
        self.relation_router = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + quality_dim),
            nn.Linear(cfg.hidden_dim + quality_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.relation_count * cfg.graph_band_count),
        )
        self.band_gate = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim + quality_dim),
            nn.Linear(cfg.hidden_dim + quality_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 2),
        )
        self.low_norm = nn.LayerNorm(cfg.hidden_dim)
        self.high_norm = nn.LayerNorm(cfg.hidden_dim)

    def _adjacency(self, mask: Tensor, speaker_index: Tensor, dtype: torch.dtype) -> Tensor:
        batch, turns = mask.shape
        adjacency = torch.zeros((batch, 4, turns, turns), device=mask.device, dtype=dtype)
        mask_cpu = mask.detach().cpu(); speaker_cpu = speaker_index.detach().cpu()
        for b in range(batch):
            length = int(mask_cpu[b].sum().item())
            for i in range(length):
                adjacency[b, :, i, i] = 1.0
                for j in range(i):
                    distance = i - j
                    same = int(speaker_cpu[b, i].item()) == int(speaker_cpu[b, j].item())
                    if same:
                        weight = math.exp(-self.cfg.graph_distance_decay * distance)
                        adjacency[b, 0, i, j] = weight; adjacency[b, 0, j, i] = weight
                    if distance == 1 and not same:
                        adjacency[b, 1, i, j] = 1.0; adjacency[b, 1, j, i] = 1.0
                    if distance == 1:
                        adjacency[b, 2, i, j] = 1.0; adjacency[b, 2, j, i] = 1.0
                    if distance >= self.cfg.graph_long_distance_minimum:
                        weight = math.exp(-self.cfg.graph_distance_decay * (distance - self.cfg.graph_long_distance_minimum))
                        adjacency[b, 3, i, j] = weight; adjacency[b, 3, j, i] = weight
        degree = adjacency.sum(dim=-1).clamp_min(1e-12)
        inv = degree.rsqrt()
        return adjacency * inv.unsqueeze(-1) * inv.unsqueeze(-2)

    def forward(
        self,
        hidden: Tensor,
        utterance_mask: Tensor,
        speaker_index: Tensor,
        audio_quality: Tensor,
        audio_available: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        adjacency = self._adjacency(utterance_mask, speaker_index, hidden.dtype)
        low_rel = torch.einsum("brij,bjh->brih", adjacency, hidden)
        high_rel = hidden.unsqueeze(1) - low_rel
        quality = torch.cat((audio_quality, audio_available.to(hidden.dtype).unsqueeze(-1)), dim=-1)
        quality = self.quality_norm(quality)
        router_input = torch.cat((hidden, quality), dim=-1)
        router_logits = self.relation_router(router_input).reshape(*hidden.shape[:2], 4, 2)
        relation_weights = torch.softmax(router_logits, dim=-2)
        relation_weights = _zero_masked(relation_weights, utterance_mask)
        low = torch.sum(low_rel.permute(0, 2, 1, 3) * relation_weights[..., 0].unsqueeze(-1), dim=-2)
        high = torch.sum(high_rel.permute(0, 2, 1, 3) * relation_weights[..., 1].unsqueeze(-1), dim=-2)
        band = torch.sigmoid(self.band_gate(router_input))
        band = _zero_masked(band, utterance_mask)
        low = _zero_masked(self.low_norm(low) * band[..., :1], utterance_mask)
        high = _zero_masked(self.high_norm(high) * band[..., 1:], utterance_mask)
        return low, high, adjacency


class _StableInitialResidualPropagation(nn.Module):
    """GCNII-style propagation retaining initial features.

    Initial-feature injection and layer-dependent identity mapping retain the
    initial features. Output propagation has fixed unit strength in the paper model.
    """

    def __init__(self, cfg: SAGERBackboneConfig) -> None:
        super().__init__()
        self.alpha = float(cfg.stable_graph_alpha)
        self.lamda = float(cfg.stable_graph_lambda)
        self.layers = nn.ModuleList(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            for _ in range(cfg.stable_graph_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.stable_graph_layers)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        hidden: Tensor,
        adjacency: Tensor,
        mask: Tensor,
    ) -> Tensor:
        h0 = hidden
        value = hidden
        for index, (linear, norm) in enumerate(zip(self.layers, self.norms), start=1):
            propagated = torch.matmul(adjacency, value)
            support = (1.0 - self.alpha) * propagated + self.alpha * h0
            theta = math.log(self.lamda / index + 1.0)
            value = (1.0 - theta) * support + theta * linear(support)
            value = _zero_masked(norm(F.gelu(value)), mask)
        update = value - h0
        result = h0 + self.dropout(update)
        return _zero_masked(result, mask)


class _BoundaryConditionedMultiScaleRelationPathMoE(nn.Module):
    """SAGER-native local-chain, speaker-memory and sparse relation-tree paths."""

    def __init__(self, cfg: SAGERBackboneConfig) -> None:
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.input_projection = nn.Sequential(
            nn.LayerNorm(4 * h), nn.Linear(4 * h, h), nn.GELU(), nn.Dropout(cfg.dropout)
        )
        local_branches = 1 + 2 * len(cfg.path_local_scales)
        self.local_gate = nn.Sequential(
            nn.LayerNorm(local_branches * h), nn.Linear(local_branches * h, h), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(h, local_branches),
        )
        self.local_norm = nn.LayerNorm(h)
        self.speaker_query = nn.Linear(h, h, bias=False)
        self.speaker_key = nn.Linear(h, h, bias=False)
        self.speaker_value = nn.Linear(h, h, bias=False)
        self.speaker_norm = nn.LayerNorm(h)
        self.tree_query = nn.Linear(h, h, bias=False)
        self.tree_key = nn.Linear(h, h, bias=False)
        self.tree_value = nn.Linear(h, h, bias=False)
        self.relation_parent_bias = nn.Parameter(torch.zeros(cfg.relation_count))
        self.tree_norm = nn.LayerNorm(h)
        self.stable_expert = nn.Sequential(
            nn.LayerNorm(3 * h), nn.Linear(3 * h, 2 * h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * h, h)
        )
        self.boundary_expert = nn.Sequential(
            nn.LayerNorm(3 * h), nn.Linear(3 * h, 2 * h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * h, h)
        )
        self.path_router = nn.Sequential(
            nn.LayerNorm(h + 3), nn.Linear(h + 3, h // 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h // 2, 2)
        )
        self.output_norm = nn.LayerNorm(h)

    @staticmethod
    def _shift(hidden: Tensor, mask: Tensor, offset: int) -> tuple[Tensor, Tensor]:
        shifted = torch.zeros_like(hidden)
        available = torch.zeros_like(mask)
        if offset > 0:
            shifted[:, offset:] = hidden[:, :-offset]
            available[:, offset:] = mask[:, :-offset]
        else:
            width = -offset
            shifted[:, :-width] = hidden[:, width:]
            available[:, :-width] = mask[:, width:]
        return shifted, available

    def _local_chain(self, hidden: Tensor, mask: Tensor) -> Tensor:
        branches = [hidden]
        branch_masks = [mask]
        for scale in self.cfg.path_local_scales:
            for offset in (scale, -scale):
                shifted, available = self._shift(hidden, mask, offset)
                branches.append(shifted)
                branch_masks.append(available & mask)
        stacked = torch.stack(branches, dim=-2)
        available = torch.stack(branch_masks, dim=-1)
        logits = self.local_gate(torch.cat(branches, dim=-1))
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        local = torch.sum(stacked * weights.unsqueeze(-1), dim=-2)
        return _zero_masked(self.local_norm(local), mask)

    def _speaker_memory(self, hidden: Tensor, mask: Tensor, speaker_index: Tensor) -> Tensor:
        batch, turns, width = hidden.shape
        q = self.speaker_query(hidden)
        k = self.speaker_key(hidden)
        v = self.speaker_value(hidden)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(width)
        target = torch.arange(turns, device=hidden.device).view(1, turns, 1)
        source = torch.arange(turns, device=hidden.device).view(1, 1, turns)
        same = speaker_index.unsqueeze(-1).eq(speaker_index.unsqueeze(-2))
        candidate = mask.unsqueeze(-1) & mask.unsqueeze(-2) & same & source.le(target)
        diagonal = torch.eye(turns, device=hidden.device, dtype=torch.bool).unsqueeze(0)
        candidate = torch.where(candidate.any(dim=-1, keepdim=True), candidate, diagonal)
        distance = (target - source).clamp_min(0).to(hidden.dtype)
        scores = scores - self.cfg.tree_distance_decay * distance
        scores = scores.masked_fill(~candidate, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        memory = torch.matmul(attention, v)
        return _zero_masked(self.speaker_norm(memory), mask)

    def _relation_tree(self, hidden: Tensor, mask: Tensor, relation_adjacency: Tensor) -> Tensor:
        batch, turns, width = hidden.shape
        q = self.tree_query(hidden)
        k = self.tree_key(hidden)
        v = self.tree_value(hidden)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(width)
        relation_strength = torch.einsum("brij,r->bij", relation_adjacency, self.relation_parent_bias)
        target = torch.arange(turns, device=hidden.device).view(1, turns, 1)
        source = torch.arange(turns, device=hidden.device).view(1, 1, turns)
        distance = (target - source).clamp_min(0).to(hidden.dtype)
        scores = scores + relation_strength - self.cfg.tree_distance_decay * distance
        past = mask.unsqueeze(-1) & mask.unsqueeze(-2) & source.lt(target)
        diagonal = torch.eye(turns, device=hidden.device, dtype=torch.bool).unsqueeze(0)
        candidates = torch.where(past.any(dim=-1, keepdim=True), past, diagonal)
        masked = scores.masked_fill(~candidates, torch.finfo(scores.dtype).min)
        top_k = min(self.cfg.tree_parent_top_k, turns)
        top_indices = masked.topk(top_k, dim=-1).indices
        sparse = torch.zeros_like(candidates).scatter(-1, top_indices, True) & candidates
        attention = torch.softmax(masked.masked_fill(~sparse, torch.finfo(scores.dtype).min), dim=-1)
        tree = torch.matmul(attention, v)
        return _zero_masked(self.tree_norm(tree), mask)

    def forward(
        self,
        text: Tensor,
        cross: Tensor,
        low: Tensor,
        high: Tensor,
        mask: Tensor,
        speaker_index: Tensor,
        relation_adjacency: Tensor,
        change_probability: Tensor,
        audio_available: Tensor,
        audio_quality_summary: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = _zero_masked(self.input_projection(torch.cat((text, cross, low, high), dim=-1)), mask)
        local = self._local_chain(hidden, mask)
        speaker = self._speaker_memory(hidden, mask, speaker_index)
        tree = self._relation_tree(hidden, mask, relation_adjacency)
        stable = self.stable_expert(torch.cat((hidden, speaker, tree), dim=-1))
        boundary = self.boundary_expert(torch.cat((hidden, local, tree), dim=-1))
        router_features = torch.cat(
            (
                hidden,
                change_probability.unsqueeze(-1),
                audio_available.to(hidden.dtype).unsqueeze(-1),
                audio_quality_summary.unsqueeze(-1),
            ),
            dim=-1,
        )
        router_logits = self.path_router(router_features)
        router_logits[..., 0] = router_logits[..., 0] + self.cfg.boundary_prior_scale * (1.0 - change_probability)
        router_logits[..., 1] = router_logits[..., 1] + self.cfg.boundary_prior_scale * change_probability
        gate = _zero_masked(torch.softmax(router_logits, dim=-1), mask)
        mixed = gate[..., :1] * stable + gate[..., 1:] * boundary
        path = hidden + mixed
        path = _zero_masked(self.output_norm(path), mask)
        return path, local, speaker, tree


from .module_strength import strength, blend

class SAGERBackboneModel(nn.Module):
    def __init__(self, config: SAGERBackboneConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.text_adapter = _TokenAdapter(1024, 1, h, config.dropout)
        self.audio_adapter = _TokenAdapter(768, 27, h, config.dropout, audio_layout=True)
        self.pre_state = nn.ModuleDict(
            {
                "text": _ChangePointSpeakerDialogueStateSpace(config),
                "audio": _ChangePointSpeakerDialogueStateSpace(config),
            }
        )
        self.text_temporal = _SparseTemporalMoE(h, config.temporal_kernels, config.top_k_experts, config.dropout)
        self.audio_temporal = _SparseTemporalMoE(h, config.temporal_kernels, config.top_k_experts, config.dropout, quality_dim=6)
        self.cross_modal_layers = nn.ModuleList(_TextAudioCrossAttention(h, config.attention_heads, config.dropout) for _ in range(config.cross_modal_layers))
        self.post_state = _ChangePointSpeakerDialogueStateSpace(config)
        self.typed_graph = _RelationTypedSpectralGraph(config)
        self.stable_graph = _StableInitialResidualPropagation(config)
        self.relation_path_moe = _BoundaryConditionedMultiScaleRelationPathMoE(config)
        self.relational_proposal_head = nn.Linear(h, config.num_classes)
        self.relational_view_router = nn.Sequential(
            nn.LayerNorm(h + 8), nn.Linear(h + 8, h // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(h // 2, 3),
        )
        self.relational_verifier = nn.Sequential(
            nn.LayerNorm(h + 12), nn.Linear(h + 12, h // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(h // 2, 1),
        )
        nn.init.zeros_(self.relational_verifier[-1].weight)
        nn.init.constant_(self.relational_verifier[-1].bias, -2.0)
        self.expert_heads = nn.ModuleDict({name: nn.Linear(h, config.num_classes) for name in EXPERT_NAMES})
        self.audio_quality_norm = nn.LayerNorm(6)
        utility_input_dim = 2 * h + 6 + 1
        self.utility_router = nn.Sequential(nn.LayerNorm(utility_input_dim), nn.Linear(utility_input_dim, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, 1))
        gate_input_dim = h + 17
        self.quality_gate = nn.Sequential(nn.LayerNorm(gate_input_dim), nn.Linear(gate_input_dim, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, len(EXPERT_NAMES) - 1))
        correction_input_dim = h + 7
        self.correction_gate = nn.Sequential(
            nn.LayerNorm(correction_input_dim),
            nn.Linear(correction_input_dim, h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h // 2, 1),
        )
        nn.init.zeros_(self.correction_gate[-1].weight)
        nn.init.constant_(self.correction_gate[-1].bias, config.correction_gate_initial_bias)


    def forward(self, inputs: SAGERInputs) -> SAGERBackboneOutput:
        cfg = self.config
        _validate_inputs(inputs, num_classes=cfg.num_classes)
        um, am = inputs.utterance_mask, inputs.audio_available
        cross_mask = um & am
        text0 = self.text_adapter(inputs.text_cls.unsqueeze(-2), um)
        audio0 = self.audio_adapter(inputs.audio_tokens, am)
        text0, text_change = self.pre_state["text"](text0, um, inputs.speaker_index)
        audio0, audio_change = self.pre_state["audio"](audio0, am, inputs.speaker_index)
        text, text_router = self.text_temporal(text0, um, um)
        audio, audio_router = self.audio_temporal(audio0, am, um, inputs.audio_quality)
        cross = text
        for layer in self.cross_modal_layers:
            cross = layer(cross, audio, um, am)
        cross = blend(self, "cross_attention", text, cross)
        cross, cross_change = self.post_state(cross, cross_mask, inputs.speaker_index)
        cross = _zero_masked(cross, cross_mask)
        aq = _zero_masked(self.audio_quality_norm(_zero_masked(inputs.audio_quality, am)), am)
        base_adjacency = self.typed_graph._adjacency(um, inputs.speaker_index, cross.dtype).mean(dim=1)
        cross = self.stable_graph(cross, base_adjacency, cross_mask)
        low, high, relation_adjacency = self.typed_graph(cross, um, inputs.speaker_index, aq, am)
        low = _zero_masked(low, cross_mask)
        high = _zero_masked(high, cross_mask)
        audio_quality_summary = _zero_masked(torch.sigmoid(aq.mean(dim=-1)), am)
        change_signal = _zero_masked(torch.sigmoid(cross_change).mean(dim=-1), cross_mask)
        path, local_path, speaker_path, tree_path = self.relation_path_moe(
            text, cross, low, high, um, inputs.speaker_index, relation_adjacency,
            change_signal, am, audio_quality_summary,
        )
        representations = {"text": text, "audio": audio, "cross_modal": cross,
                           "speaker_low": low, "speaker_high": high}
        availability = {"text": um, "audio": am, "cross_modal": cross_mask,
                        "speaker_low": cross_mask, "speaker_high": cross_mask}
        expert_logits = torch.stack([
            _zero_masked(self.expert_heads[name](representations[name]), availability[name])
            for name in EXPERT_NAMES
        ], dim=-2)
        expert_available = torch.stack([availability[name] for name in EXPERT_NAMES], dim=-1)
        correction_available = expert_available[..., 1:]
        anchor = _zero_masked(inputs.text_anchor_logits.detach(), um)
        probability = torch.softmax(anchor, dim=-1)
        entropy = _zero_masked(-torch.sum(probability * torch.log(probability.clamp_min(1e-12)), dim=-1), um)
        utility_input = torch.cat((text, audio, aq, am.to(text.dtype).unsqueeze(-1)), dim=-1)
        audio_utility = _zero_masked(torch.tanh(self.utility_router(utility_input)), um)
        low_energy = _zero_masked(low.norm(dim=-1) / math.sqrt(cfg.hidden_dim), cross_mask)
        high_energy = _zero_masked(high.norm(dim=-1) / math.sqrt(cfg.hidden_dim), cross_mask)
        confidence = torch.softmax(expert_logits, dim=-1).amax(dim=-1)
        confidence = torch.where(expert_available, confidence, torch.zeros_like(confidence))
        correction_confidence = confidence[..., 1:].amax(dim=-1)
        denominator = _zero_masked(inputs.temporal_position, um).amax(dim=1, keepdim=True).clamp_min(1)
        relative = _zero_masked(inputs.temporal_position, um).to(text.dtype) / denominator.to(text.dtype)
        gate_features = torch.cat((text, aq, am.to(text.dtype).unsqueeze(-1), entropy.unsqueeze(-1),
                                   low_energy.unsqueeze(-1), high_energy.unsqueeze(-1), confidence,
                                   relative.unsqueeze(-1), audio_utility), dim=-1)
        gate_logits = self.quality_gate(gate_features)
        utility_bias = torch.cat((audio_utility, 0.5 * audio_utility,
                                  0.25 * audio_utility, 0.25 * audio_utility), dim=-1)
        gate_logits = _zero_masked(gate_logits + cfg.utility_gate_scale * utility_bias, um)
        masked = gate_logits.masked_fill(~correction_available, torch.finfo(gate_logits.dtype).min)
        gate_weights = torch.softmax(masked, dim=-1)
        gate_weights = torch.where(correction_available, gate_weights, torch.zeros_like(gate_weights))
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(gate_weights.dtype).eps)
        gate_weights = _zero_masked(gate_weights, um)
        base_evidence = _zero_masked(torch.sum(gate_weights.unsqueeze(-1) * expert_logits[..., 1:, :], dim=-2), um)
        relational_views = torch.stack((local_path, speaker_path, tree_path), dim=-2)
        relational_view_logits = _zero_masked(self.relational_proposal_head(relational_views), um)
        view_probabilities = torch.softmax(relational_view_logits, dim=-1)
        view_entropies = -torch.sum(
            view_probabilities * torch.log(view_probabilities.clamp_min(1e-12)), dim=-1
        ) / math.log(cfg.num_classes)
        mean_view_probability = view_probabilities.mean(dim=-2)
        view_disagreement = torch.sum(
            view_probabilities * (torch.log(view_probabilities.clamp_min(1e-12))
                                  - torch.log(mean_view_probability.unsqueeze(-2).clamp_min(1e-12))),
            dim=-1,
        ).mean(dim=-1)
        view_router_features = torch.cat((
            path, view_entropies, view_disagreement.unsqueeze(-1), change_signal.unsqueeze(-1),
            audio_utility, am.to(text.dtype).unsqueeze(-1), audio_quality_summary.unsqueeze(-1),
        ), dim=-1)
        relational_view_weights = _zero_masked(torch.softmax(self.relational_view_router(view_router_features), dim=-1), um)
        relational_consensus = _zero_masked(
            torch.sum(relational_view_weights.unsqueeze(-1) * relational_view_logits, dim=-2), um
        )
        sorted_anchor_probability = probability.sort(dim=-1, descending=True).values
        anchor_margin = _zero_masked(sorted_anchor_probability[..., 0] - sorted_anchor_probability[..., 1], um)
        base_confidence = torch.softmax(base_evidence, dim=-1).amax(dim=-1)
        consensus_confidence = torch.softmax(relational_consensus, dim=-1).amax(dim=-1)
        verifier_features = torch.cat((
            path, view_entropies, view_disagreement.unsqueeze(-1), change_signal.unsqueeze(-1),
            (entropy / math.log(cfg.num_classes)).unsqueeze(-1), anchor_margin.unsqueeze(-1),
            base_confidence.unsqueeze(-1), consensus_confidence.unsqueeze(-1), audio_utility,
            am.to(text.dtype).unsqueeze(-1), audio_quality_summary.unsqueeze(-1),
        ), dim=-1)
        relational_verifier_logit = _zero_masked(self.relational_verifier(verifier_features).squeeze(-1), um)
        relational_verifier_probability = _zero_masked(torch.sigmoid(relational_verifier_logit), um)
        relational_verifier_probability = relational_verifier_probability * strength(self, "relation_path")
        evidence = _zero_masked(
            base_evidence + relational_verifier_probability.unsqueeze(-1) * (relational_consensus - base_evidence), um
        )
        raw = _zero_masked(evidence - anchor, um)
        correction_features = torch.cat((
            text, audio_utility, (entropy / math.log(cfg.num_classes)).unsqueeze(-1),
            anchor_margin.unsqueeze(-1), correction_confidence.unsqueeze(-1),
            am.to(text.dtype).unsqueeze(-1), audio_quality_summary.unsqueeze(-1), change_signal.unsqueeze(-1),
        ), dim=-1)
        override_logit = _zero_masked(self.correction_gate(correction_features).squeeze(-1), um)
        correction_strength = torch.where(
            correction_available.any(dim=-1), torch.sigmoid(override_logit), torch.zeros_like(override_logit)
        )
        correction_strength = _zero_masked(correction_strength, um)
        correction = _zero_masked(raw * correction_strength.unsqueeze(-1), um)
        logits = _zero_masked(anchor + correction, um)
        return SAGERBackboneOutput(
            logits=logits, anchor_logits=anchor, expert_logits=expert_logits, expert_available=expert_available,
            correction_strength=correction_strength, correction_evidence_logits=evidence,
            override_logit=override_logit, text_representation=text, audio_representation=audio,
            cross_modal_representation=cross, speaker_low_representation=low, speaker_high_representation=high,
            predicted_audio_utility=audio_utility.squeeze(-1),
            change_boundary_logits=torch.stack((text_change, audio_change, cross_change), dim=-2),
            change_state_available=torch.stack((um, am, cross_mask), dim=-1),
            speaker_index=inputs.speaker_index, text_router_weights=text_router, audio_router_weights=audio_router,
            utterance_mask=um, base_evidence_logits=base_evidence, relational_view_logits=relational_view_logits,
            relational_consensus_logits=relational_consensus, relational_view_disagreement=view_disagreement,
            relational_verifier_logit=relational_verifier_logit,
            relational_verifier_probability=relational_verifier_probability,
        )


def compute_sager_backbone_objective(
    output: SAGERBackboneOutput,
    targets: Tensor,
    utility_targets: Tensor,
    utility_mask: Tensor,
    *,
    gamma: float,
    final_weight: float,
    expert_weight: float,
    supcon_weight: float,
    temperature: float,
    directional_kl_weight: float,
    spectral_weight: float,
    router_balance_weight: float,
    utility_weight: float,
    utility_huber_delta: float,
    change_point_weight: float,
    override_weight: float,
    relational_proposal_weight: float = 0.0,
    relational_consistency_weight: float = 0.0,
    relational_verifier_weight: float = 0.0,
) -> SAGERBackboneLoss:
    if utility_targets.shape != targets.shape or utility_mask.shape != targets.shape:
        raise ValueError("utility target/mask shapes differ")
    if utility_targets.device != output.logits.device or utility_mask.device != output.logits.device:
        raise ValueError("utility tensors and output must share a device")
    final_focal = masked_focal_loss(output.logits, targets, output.utterance_mask, gamma=gamma)
    expert_losses = []
    for index in range(len(output.expert_names)):
        mask = output.expert_available[..., index]
        if bool(mask.any().item()):
            expert_losses.append(masked_focal_loss(output.expert_logits[..., index, :], targets, mask, gamma=gamma))
    expert_focal = torch.stack(expert_losses).mean() if expert_losses else output.expert_logits.sum() * 0.0
    auxiliary = compute_auxiliary_objective(
        output, targets,
        supcon_weight=supcon_weight, temperature=temperature,
        directional_kl_weight=directional_kl_weight, spectral_weight=spectral_weight,
        router_balance_weight=router_balance_weight,
    )
    active_utility = utility_mask & output.utterance_mask & output.availability_for("audio")
    if bool(active_utility.any().item()) and utility_weight > 0:
        utility = F.huber_loss(output.predicted_audio_utility[active_utility], utility_targets[active_utility], reduction="mean", delta=utility_huber_delta)
    else:
        utility = output.predicted_audio_utility.sum() * 0.0
    logits = output.change_boundary_logits
    if logits.shape != (*targets.shape, 3, 4) or output.change_state_available.shape != (*targets.shape, 3):
        raise ValueError("change-point tensor shapes differ")
    batch, turns = targets.shape
    boundary_targets = torch.zeros_like(logits)
    boundary_mask = torch.zeros_like(logits, dtype=torch.bool)
    positions = torch.arange(turns, device=targets.device)
    target_positions = positions.view(1, turns, 1)
    source_positions = positions.view(1, 1, turns)
    same_speaker = output.speaker_index.unsqueeze(-1).eq(output.speaker_index.unsqueeze(-2))
    for stream in range(3):
        available = output.change_state_available[..., stream] & output.utterance_mask
        adjacent = available[:, 1:] & available[:, :-1]
        adjacent_change = targets[:, 1:].ne(targets[:, :-1]).to(logits.dtype)
        boundary_targets[:, 1:, stream, 0] = adjacent_change
        boundary_mask[:, 1:, stream, 0] = adjacent
        boundary_targets[:, :-1, stream, 1] = adjacent_change
        boundary_mask[:, :-1, stream, 1] = adjacent
        pair_available = available.unsqueeze(-1) & available.unsqueeze(-2) & same_speaker
        previous_candidates = pair_available & source_positions.lt(target_positions)
        previous_indices = torch.where(previous_candidates, source_positions, torch.full_like(source_positions, -1)).amax(dim=-1)
        has_previous = previous_indices.ge(0) & available
        previous_labels = targets.gather(1, previous_indices.clamp_min(0))
        boundary_targets[..., stream, 2] = targets.ne(previous_labels).to(logits.dtype)
        boundary_mask[..., stream, 2] = has_previous
        next_candidates = pair_available & source_positions.gt(target_positions)
        next_indices = torch.where(next_candidates, source_positions, torch.full_like(source_positions, turns)).amin(dim=-1)
        has_next = next_indices.lt(turns) & available
        next_labels = targets.gather(1, next_indices.clamp_max(turns - 1))
        boundary_targets[..., stream, 3] = targets.ne(next_labels).to(logits.dtype)
        boundary_mask[..., stream, 3] = has_next
    if bool(boundary_mask.any().item()) and change_point_weight > 0:
        change_point = F.binary_cross_entropy_with_logits(logits[boundary_mask], boundary_targets[boundary_mask])
    else:
        change_point = logits.sum() * 0.0
    override_available = output.utterance_mask & output.expert_available[..., 1:].any(dim=-1)
    if bool(override_available.any().item()) and override_weight > 0:
        safe_targets = targets.clamp(min=0, max=output.logits.shape[-1] - 1)
        anchor_nll = -torch.log_softmax(output.anchor_logits.detach(), dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        evidence_nll = -torch.log_softmax(output.correction_evidence_logits.detach(), dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        override_targets = evidence_nll.lt(anchor_nll).to(output.override_logit.dtype)
        active_targets = override_targets[override_available]
        positives = active_targets.sum().clamp_min(1.0)
        negatives = (1.0 - active_targets).sum().clamp_min(1.0)
        count = active_targets.numel()
        weights = torch.where(
            active_targets.gt(0.5),
            torch.full_like(active_targets, 0.5 * count) / positives,
            torch.full_like(active_targets, 0.5 * count) / negatives,
        )
        counterfactual_override = F.binary_cross_entropy_with_logits(
            output.override_logit[override_available], active_targets, weight=weights
        )
    else:
        counterfactual_override = output.override_logit.sum() * 0.0
    proposal_losses = [
        masked_focal_loss(output.relational_view_logits[..., index, :], targets, output.utterance_mask, gamma=gamma)
        for index in range(output.relational_view_logits.shape[-2])
    ]
    relational_proposal = torch.stack(proposal_losses).mean()
    relational_consistency = output.relational_view_disagreement[output.utterance_mask].mean()
    safe_targets = targets.clamp(min=0, max=output.logits.shape[-1] - 1)
    base_nll = -torch.log_softmax(output.base_evidence_logits.detach(), dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    consensus_nll = -torch.log_softmax(output.relational_consensus_logits.detach(), dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    verifier_target = consensus_nll.lt(base_nll).to(output.relational_verifier_logit.dtype)
    verifier_active = output.utterance_mask
    active_verifier_target = verifier_target[verifier_active]
    positives = active_verifier_target.sum().clamp_min(1.0)
    negatives = (1.0 - active_verifier_target).sum().clamp_min(1.0)
    verifier_count = active_verifier_target.numel()
    verifier_weights = torch.where(
        active_verifier_target.gt(0.5),
        torch.full_like(active_verifier_target, 0.5 * verifier_count) / positives,
        torch.full_like(active_verifier_target, 0.5 * verifier_count) / negatives,
    )
    relational_verifier = F.binary_cross_entropy_with_logits(
        output.relational_verifier_logit[verifier_active], active_verifier_target, weight=verifier_weights
    )
    total = (
        final_weight * final_focal
        + expert_weight * expert_focal
        + auxiliary.total
        + utility_weight * utility
        + change_point_weight * change_point
        + override_weight * counterfactual_override
        + relational_proposal_weight * relational_proposal
        + relational_consistency_weight * relational_consistency
        + relational_verifier_weight * relational_verifier
    )
    return SAGERBackboneLoss(
        total,
        final_focal,
        expert_focal,
        auxiliary.supervised_contrastive,
        auxiliary.directional_kl,
        auxiliary.spectral,
        auxiliary.router_balance,
        utility,
        change_point,
        counterfactual_override,
        relational_proposal,
        relational_consistency,
        relational_verifier,
    )


__all__ = [
    "RELATION_NAMES", "SAGERBackboneConfig", "SAGERInputs", "SAGERBackboneLoss",
    "SAGERBackboneModel", "SAGERBackboneOutput", "compute_sager_backbone_objective",
]
