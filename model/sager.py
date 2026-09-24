"""SAGER implementation aligned to the ICASSP 2027 manuscript equations."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping
from torch import Tensor
from .backbone import SAGERBackboneConfig, SAGERBackboneModel, SAGERInputs, compute_sager_backbone_objective
from .module_strength import ModuleStrengths


@dataclass(frozen=True)
class SAGERConfig(SAGERBackboneConfig):
    num_classes: int = 7
    implementation_id: str = "sager_paper_v1"

    def __post_init__(self):
        super().__post_init__()
        if self.implementation_id != "sager_paper_v1":
            raise ValueError("This implementation requires sager_paper_v1")


class SAGERModel(SAGERBackboneModel):
    def __init__(self, config: SAGERConfig):
        super().__init__(config)
        self.module_strengths = ModuleStrengths()

    def forward(self, inputs: SAGERInputs):
        # Upstream feature extractors and the text anchor remain frozen.
        values = replace(
            inputs,
            text_cls=inputs.text_cls.detach(),
            text_anchor_logits=inputs.text_anchor_logits.detach(),
            audio_tokens=inputs.audio_tokens.detach(),
            audio_quality=inputs.audio_quality.detach(),
        )
        return super().forward(values)


def compute_sager_objective(output, targets: Tensor, utility_targets: Tensor,
                            utility_mask: Tensor,
                            cfg: Mapping[str, Any]) -> dict[str, Tensor]:
    losses = compute_sager_backbone_objective(
        output, targets, utility_targets, utility_mask,
        gamma=float(cfg["focal_gamma"]), final_weight=float(cfg["final_weight"]),
        expert_weight=float(cfg["expert_weight"]),
        supcon_weight=float(cfg["supervised_contrastive_weight"]),
        temperature=float(cfg["supervised_contrastive_temperature"]),
        directional_kl_weight=float(cfg["directional_kl_weight"]),
        spectral_weight=float(cfg["spectral_weight"]),
        router_balance_weight=float(cfg["router_balance_weight"]),
        utility_weight=float(cfg["counterfactual_utility_weight"]),
        utility_huber_delta=float(cfg["utility_huber_delta"]),
        change_point_weight=float(cfg["change_point_weight"]),
        override_weight=float(cfg["counterfactual_override_weight"]),
        relational_proposal_weight=float(cfg["relational_proposal_weight"]),
        relational_consistency_weight=float(cfg["relational_consistency_weight"]),
        relational_verifier_weight=float(cfg["relational_verifier_weight"]),
    )
    # Do not dataclasses.asdict() tensors: it deep-copies non-leaf autograd tensors.
    return {name: getattr(losses, name) for name in losses.__dataclass_fields__}


__all__ = ["SAGERConfig", "SAGERModel", "compute_sager_objective"]
