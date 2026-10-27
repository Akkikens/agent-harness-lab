"""Capability-based access control.

A tool never asks "is this agent allowed to use me?"; it declares the exact
capabilities a call needs, and the registry checks them against the set the
session was granted. Sets can only be narrowed (attenuated), never widened.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator


class CapabilityDenied(Exception):
    """A tool call needed a capability the session does not hold."""


@dataclass(frozen=True)
class Capability:
    kind: str  # fs.read | fs.write | exec | net
    scope: str = "*"

    def __str__(self) -> str:
        return f"{self.kind}:{self.scope}"

    @classmethod
    def parse(cls, text: str) -> "Capability":
        kind, _, scope = text.partition(":")
        return cls(kind, scope or "*")

    def covers(self, need: "Capability") -> bool:
        if need.kind != self.kind:
            return False
        if self.scope == "*":
            return True
        if self.kind.startswith("fs."):
            # realpath resolves "..", and symlinks that point outside the root.
            root = os.path.realpath(self.scope)
            target = os.path.realpath(need.scope)
            return os.path.commonpath([root, target]) == root
        return self.scope == need.scope


class CapabilitySet:
    def __init__(self, caps: Iterable[Capability] = ()):
        self._caps = frozenset(caps)

    def __iter__(self) -> Iterator[Capability]:
        return iter(sorted(self._caps, key=str))

    def allows(self, need: Capability) -> bool:
        return any(c.covers(need) for c in self._caps)

    def check(self, needs: Iterable[Capability]) -> None:
        for need in needs:
            if not self.allows(need):
                raise CapabilityDenied(f"missing capability {need}")

    def attenuate(self, *narrower: Capability) -> "CapabilitySet":
        """Derive a subset. Asking for anything this set does not already
        allow is an error, so authority can only shrink down a call chain."""
        for cap in narrower:
            if not self.allows(cap):
                raise CapabilityDenied(f"cannot widen authority to {cap}")
        return CapabilitySet(narrower)
