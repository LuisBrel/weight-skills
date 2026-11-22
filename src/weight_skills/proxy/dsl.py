"""Model alias DSL: ``qwen-35b+aws-cso@1.0+security@1.2``.

The alias is what a user types into Cursor / LM Studio / any OpenAI-compatible
client. It carries the base model plus the ordered list of weight skills to
mount, so no client-side configuration is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ALIAS = re.compile(
    r"^(?P<base>[A-Za-z0-9_.\-/]+)"
    r"(?P<skills>(?:\+[A-Za-z0-9_.\-]+(?:@[A-Za-z0-9_.\-]+)?)*)$"
)
_SKILL = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)(?:@(?P<version>[A-Za-z0-9_.\-]+))?$")


@dataclass(frozen=True)
class SkillRef:
    name: str
    version: str | None = None

    def __str__(self) -> str:
        return f"{self.name}@{self.version}" if self.version else self.name


@dataclass(frozen=True)
class AliasSpec:
    """Parsed model alias."""

    base: str
    skills: tuple[SkillRef, ...] = field(default_factory=tuple)
    raw: str = ""

    @property
    def is_plain(self) -> bool:
        return not self.skills

    def as_dict(self) -> dict[str, object]:
        return {
            "base": self.base,
            "skills": [{"name": s.name, "version": s.version} for s in self.skills],
            "raw": self.raw,
        }


def parse_alias(alias: str) -> AliasSpec:
    """Parse a model alias, raising ``ValueError`` on malformed input."""
    text = (alias or "").strip()
    if not text:
        raise ValueError("empty model alias")
    match = _ALIAS.match(text)
    if not match:
        raise ValueError(
            f"invalid model alias {alias!r}; expected 'base+skill[@version]+skill2[@version]'"
        )
    base = match.group("base")
    skills: list[SkillRef] = []
    for chunk in filter(None, match.group("skills").split("+")):
        skill = _SKILL.match(chunk)
        if not skill:
            raise ValueError(f"invalid skill segment {chunk!r} in alias {alias!r}")
        skills.append(SkillRef(skill.group("name"), skill.group("version")))
    return AliasSpec(base=base, skills=tuple(skills), raw=text)


def build_alias(base: str, skills: list[SkillRef]) -> str:
    """Inverse of :func:`parse_alias` (useful for round-trip tests)."""
    return "+".join([base, *[str(s) for s in skills]])
