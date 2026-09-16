"""The paper's two global strengths, projected onto [0, 1]."""
from __future__ import annotations
import torch
from torch import nn

SITES = ("cross_attention", "relation_path")

class ModuleStrengths(nn.Module):
    """s_ca = cross_attention; s_ver = relation_path."""
    def __init__(self):
        super().__init__()
        for name in SITES:
            self.register_parameter(name, nn.Parameter(torch.ones(())))

    def value(self, name):
        if name not in SITES:
            raise KeyError(f"The paper has no global strength named {name}")
        return getattr(self, name)

    def project(self):
        with torch.no_grad():
            for value in self.parameters():
                if not bool(torch.isfinite(value)):
                    raise RuntimeError("nonfinite module strength")
                value.clamp_(0.0, 1.0)

    def values(self):
        return {name: float(getattr(self, name).detach().cpu()) for name in SITES}


def strength(model, name):
    return model.module_strengths.value(name)


def blend(model, name, source, processed):
    return source + strength(model, name) * (processed - source)


def project(model):
    model.module_strengths.project()
