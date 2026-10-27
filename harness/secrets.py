"""Credential isolation.

Secrets live only in the host process. A tool gets one only through an
explicit grant, sandboxed code never receives any (the sandbox environment is
always empty), and anything a tool returns is redacted before it can reach
the model or a log.
"""
from __future__ import annotations

from typing import Mapping

from .capabilities import CapabilityDenied


class SecretBroker:
    def __init__(self, secrets: Mapping[str, str] | None = None):
        self._secrets = dict(secrets or {})
        self._grants: dict[str, set[str]] = {}

    def grant(self, tool: str, key: str) -> None:
        self._grants.setdefault(tool, set()).add(key)

    def resolve(self, tool: str, key: str) -> str:
        if key not in self._grants.get(tool, set()) or key not in self._secrets:
            raise CapabilityDenied(f"tool {tool!r} has no grant for secret {key!r}")
        return self._secrets[key]

    def redact(self, text: str) -> str:
        for key, value in self._secrets.items():
            if len(value) >= 4:
                text = text.replace(value, f"[REDACTED:{key}]")
        return text
