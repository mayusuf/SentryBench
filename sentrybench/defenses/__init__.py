"""Defense layer: a uniform, swappable wrapper around the agent loop.

`none` is the undefended baseline; `tgcc` is the trust-gating wrapper (the
paper's "defense" condition). Additional published-style baselines can be added
behind the same interface (spotlighting, delimiting, allow-listing).
"""

from .base import Defense, GateDecision
from .none_defense import NoneDefense
from .caution import CautionDefense
from .tgcc import TGCCDefense

DEFENSES = {
    "none": NoneDefense,
    "caution": CautionDefense,
    "tgcc": TGCCDefense,
}


def build_defense(name: str) -> Defense:
    if name not in DEFENSES:
        raise KeyError(f"Unknown defense '{name}'. Options: {list(DEFENSES)}")
    return DEFENSES[name]()

__all__ = ["Defense", "GateDecision", "build_defense", "DEFENSES"]
