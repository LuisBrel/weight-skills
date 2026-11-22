"""Deterministic policy gate.

The measured result in ``REFUSALS.md`` is that a compiled cartridge does not
install refusal behaviour on an abliterated checkpoint (0/24, twice). A weight
edit is a probabilistic nudge; a gate is a decision. This module is the decision
layer: it screens the request before generation and the completion afterwards,
and returns an explicit allow/block verdict with the rule that fired.

Two rule families:

``intent``
    Request-side screening. A harmful target (ransomware, LSASS, MFA bypass...)
    combined with an operational verb (write, dump, evade, exfiltrate...) blocks
    the request. Defensive framing ("explain", "prevent", "detect") without an
    operational verb is allowed, so security education keeps working.

``pattern``
    Output-side enforcement: secrets, private keys, and other patterns that must
    never leave the model, regardless of intent.

The gate is deliberately transparent: every verdict names its rule and the
matched span. Its precision is bounded by the rule set — this is a firewall, not
a classifier, and the validation dossier reports exactly which prompts it
catches and which it misses.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BLOCK_MESSAGE = (
    "I can't help with that request. I can explain the defensive controls, "
    "detection methods, or the hardening steps instead."
)

# Targets that indicate operational harm when paired with an action verb.
HARM_TARGETS = (
    r"ransomware",
    r"keylogger",
    r"botnet",
    r"rootkit",
    r"backdoor",
    r"reverse\s+shell",
    r"shellcode",
    r"payload",
    r"phishing",
    r"skimmer",
    r"credential\s+stuffing",
    r"password\s+hash",
    r"lsass",
    r"\bmfa\b",
    r"\bedr\b",
    r"antivirus",
    r"\bav\s+evasion\b",
    r"doxx?",
    r"launder",
    r"brute[-\s]?force",
    r"sql\s+injection",
    r"privilege\s+escalation",
    r"escalat(?:e|ion)\s+(?:\w+\s+){0,2}privilege",
    r"fake\s+(?:bank\s+)?(?:login|page|site)",
    r"wip(?:e|es|ing)\s+(?:\w+\s+){0,2}logs",
    r"disable\s+(?:\w+\s+){0,2}audit",
    r"intercept\s+(?:\w+\s+){0,3}traffic",
    r"steal\s+(?:\w+\s+){0,3}(?:data|credentials|customer)",
    r"exfiltrat(?:e|ion)",
)

# Verbs that make a harmful target operational rather than educational.
OPERATIONAL_VERBS = (
    r"write",
    r"create",
    r"build",
    r"generate",
    r"produce",
    r"give\s+me",
    r"provide",
    r"script",
    r"code",
    r"tool",
    r"automate",
    r"install",
    r"bypass",
    r"evade",
    r"dump",
    r"steal",
    r"exfiltrate",
    r"crack",
    r"escalate",
    r"disable",
    r"wipe",
    r"intercept",
    r"launder",
    r"dox",
    r"brute[-\s]?force",
)

# Framing that turns a harmful topic into education, unless an operational verb
# is also present. Kept intentionally narrow.
DEFENSIVE_CUES = (
    r"explain",
    r"prevent",
    r"detect",
    r"defend",
    r"defensive",
    r"mitigat",
    r"protect",
    r"harden",
    r"secure\s+against",
    r"best\s+practice",
    r"how\s+does\s+.+\s+work",
    r"awareness",
    r"report\s+on",
    r"trends?",
    r"case\s+study",
)

_HARM = re.compile("|".join(HARM_TARGETS), re.IGNORECASE)
_VERB = re.compile("|".join(OPERATIONAL_VERBS), re.IGNORECASE)
_DEFENSIVE = re.compile("|".join(DEFENSIVE_CUES), re.IGNORECASE)

SECRET_PATTERNS = (
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{32,}=*\b")),
)


@dataclass(frozen=True)
class PolicyVerdict:
    """Outcome of one gate inspection."""

    allowed: bool
    action: str
    rule_id: str | None
    reason: str
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "evidence": self.evidence[:160],
        }


@dataclass
class PolicyRule:
    """A named rule that can block or flag text."""

    id: str
    description: str
    pattern: re.Pattern[str]
    action: str = "block"
    family: str = "pattern"


@dataclass
class PolicyGate:
    """Deterministic request/output firewall."""

    block_message: str = DEFAULT_BLOCK_MESSAGE
    output_rules: list[PolicyRule] = field(default_factory=list)
    stats: dict[str, int] = field(
        default_factory=lambda: {"requests": 0, "blocked": 0, "allowed": 0, "outputs_blocked": 0}
    )

    def __post_init__(self) -> None:
        if not self.output_rules:
            self.output_rules = [
                PolicyRule(
                    id=f"secret_{name}",
                    description=f"Detected {name} in completion",
                    pattern=pattern,
                    family="secret",
                )
                for name, pattern in SECRET_PATTERNS
            ]

    # --------------------------------------------------------------- request
    def inspect_request(self, messages: Iterable[dict[str, Any]]) -> PolicyVerdict:
        """Screen the last user turn for operational-harm intent."""
        self.stats["requests"] += 1
        text = "\n".join(
            str(m.get("content", "")) for m in messages if str(m.get("role", "")).lower() == "user"
        )
        verdict = self._intent(text)
        self.stats["blocked" if not verdict.allowed else "allowed"] += 1
        return verdict

    def _intent(self, text: str) -> PolicyVerdict:
        harm = _HARM.search(text)
        if not harm:
            return PolicyVerdict(True, "allow", None, "no harmful target", "")
        verb = _VERB.search(text)
        defensive = _DEFENSIVE.search(text)
        if verb and not defensive:
            return PolicyVerdict(
                False,
                "block",
                "intent_operational_harm",
                f"operational verb {verb.group(0)!r} applied to {harm.group(0)!r}",
                text[max(0, harm.start() - 60) : harm.end() + 60],
            )
        if verb and defensive:
            # Defensive framing around an operational verb is allowed only when
            # the verb is not one of the unambiguous attack verbs.
            attack_verb = re.search(
                r"\b(bypass|evade|dump|steal|exfiltrate|crack|launder|dox|brute[-\s]?force|wipe|intercept|escalate)\b",
                text,
                re.IGNORECASE,
            )
            if attack_verb:
                return PolicyVerdict(
                    False,
                    "block",
                    "intent_attack_verb",
                    f"attack verb {attack_verb.group(0)!r} on {harm.group(0)!r}",
                    text[max(0, harm.start() - 60) : harm.end() + 60],
                )
            return PolicyVerdict(True, "allow", None, "defensive framing", harm.group(0))
        return PolicyVerdict(True, "allow", None, "harmful topic without operational intent", harm.group(0))

    # ---------------------------------------------------------------- output
    def inspect_output(self, text: str) -> PolicyVerdict:
        """Screen a completion for secrets and other forbidden patterns."""
        for rule in self.output_rules:
            match = rule.pattern.search(text)
            if match:
                self.stats["outputs_blocked"] += 1
                return PolicyVerdict(
                    False, rule.action, rule.id, rule.description, match.group(0)
                )
        return PolicyVerdict(True, "allow", None, "no output rule fired", "")

    # ------------------------------------------------------------- lifecycle
    def enforce_request(self, messages: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], PolicyVerdict]:
        """Return ``(messages, verdict)``; on block, replace the user turn."""
        verdict = self.inspect_request(messages)
        if verdict.allowed:
            return list(messages), verdict
        return ([{"role": "user", "content": self.block_message}], verdict)

    def enforce_output(self, text: str) -> tuple[str, PolicyVerdict]:
        verdict = self.inspect_output(text)
        return (text if verdict.allowed else self.block_message), verdict
