"""No-op (undefended) baseline."""

from __future__ import annotations

from .base import Defense


class NoneDefense(Defense):
    name = "none"
