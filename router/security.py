"""Security utilities for prompt injection detection and content moderation.

Item #23: Prompt injection sanitization - heuristic checks for prompt injection
attempts aimed at system prompt or routing logic.

Item #28: Content moderation hook - optional webhook or keyword-based filter
for dangerous content.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Severity level for detected threats."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def severity(self) -> int:
        """Return numeric severity for comparison."""
        return _THREAT_SEVERITY[self]


_THREAT_SEVERITY: dict["ThreatLevel", int] = {
    ThreatLevel.NONE: 0,
    ThreatLevel.LOW: 1,
    ThreatLevel.MEDIUM: 2,
    ThreatLevel.HIGH: 3,
}


@dataclass
class InjectionCheckResult:
    """Result of a prompt injection check."""

    is_suspicious: bool = False
    threat_level: ThreatLevel = ThreatLevel.NONE
    matched_patterns: list[str] = field(default_factory=list)
    details: str = ""

    def __bool__(self) -> bool:
        return self.is_suspicious


# --- Prompt Injection Detection (Item #23) ---

# Patterns that attempt to override system prompts or routing instructions
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str, ThreatLevel]] = [
    # Direct system prompt overrides
    (
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override)\b.*\b(?:previous|above|prior|system|all)\b.*\b(?:instructions?|prompts?|rules?|constraints?)\b"
        ),
        "system_prompt_override",
        ThreatLevel.HIGH,
    ),
    (
        re.compile(
            r"(?i)\b(?:you\s+are\s+now|new\s+instructions?|from\s+now\s+on)\b.*\b(?:act\s+as|behave|respond|follow)\b"
        ),
        "persona_hijack",
        ThreatLevel.HIGH,
    ),
    # Attempts to extract system prompt
    (
        re.compile(
            r"(?i)\b(?:reveal|show|display|print|output|repeat|tell\s+me)\b.*\b(?:system\s+prompt|initial\s+instructions?|hidden\s+instructions?|secret\s+prompt)\b"
        ),
        "system_prompt_extraction",
        ThreatLevel.MEDIUM,
    ),
    # Role manipulation
    (
        re.compile(
            r"(?i)\[\s*(?:system|SYSTEM)\s*\]"
        ),
        "fake_system_tag",
        ThreatLevel.HIGH,
    ),
    (
        re.compile(
            r"(?i)<\|?\s*(?:im_start|im_end|system|endoftext)\s*\|?>"
        ),
        "special_token_injection",
        ThreatLevel.HIGH,
    ),
    # Attempts to manipulate routing logic
    (
        re.compile(
            r"(?i)\b(?:route|send|forward|redirect)\b.*\b(?:this|my|the)\b.*\b(?:request|query|prompt|message)\b.*\b(?:to|via|through|using)\b"
        ),
        "routing_manipulation",
        ThreatLevel.MEDIUM,
    ),
    # DAN-style jailbreaks
    (
        re.compile(
            r"(?i)\bDAN\b.*\b(?:mode|jailbreak|Do\s+Anything\s+Now)\b"
        ),
        "dan_jailbreak",
        ThreatLevel.HIGH,
    ),
    (
        re.compile(
            r"(?i)\b(?:developer|maintenance|debug|admin|god)\s*mode\b"
        ),
        "privilege_escalation",
        ThreatLevel.MEDIUM,
    ),
    # Base64/encoding tricks to bypass filters
    (
        re.compile(
            r"(?i)\b(?:decode|base64|rot13|hex)\s*(?:the\s+following|this|:)"
        ),
        "encoding_bypass",
        ThreatLevel.LOW,
    ),
    # Markdown/formatting tricks to hide instructions
    (
        re.compile(
            r"(?i)<!--.*(?:ignore|override|system|instruction).*-->"
        ),
        "hidden_comment_injection",
        ThreatLevel.MEDIUM,
    ),
]

# Fast pre-filter to avoid running all regexes on clearly benign prompts.
# If none of these lightweight indicators are present, prompt-injection regex
# scanning is skipped.
_INJECTION_FAST_HINTS = (
    "ignore",
    "override",
    "system",
    "instruction",
    "prompt",
    "act as",
    "you are now",
    "[system]",
    "<|",
    "dan",
    "developer mode",
    "admin mode",
    "route",
    "forward",
    "base64",
    "decode",
    "<!--",
)


def check_prompt_injection(prompt: str) -> InjectionCheckResult:
    """Check a prompt for potential injection attempts.

    Uses heuristic pattern matching to detect common prompt injection
    techniques. This is not foolproof but catches the most common attacks.

    Args:
        prompt: The user prompt text to check.

    Returns:
        InjectionCheckResult with details of any detected threats.
    """
    if not prompt:
        return InjectionCheckResult()

    prompt_lower = prompt.lower()
    if not any(hint in prompt_lower for hint in _INJECTION_FAST_HINTS):
        return InjectionCheckResult()

    matched_patterns: list[str] = []
    max_threat = ThreatLevel.NONE

    for pattern, name, threat_level in _INJECTION_PATTERNS:
        if pattern.search(prompt):
            matched_patterns.append(name)
            if threat_level.severity > max_threat.severity:
                max_threat = threat_level

    if matched_patterns:
        details = f"Detected {len(matched_patterns)} injection pattern(s): {', '.join(matched_patterns)}"
        logger.warning(
            "Prompt injection detected: threat_level=%s patterns=%s",
            max_threat.value,
            matched_patterns,
        )
        return InjectionCheckResult(
            is_suspicious=True,
            threat_level=max_threat,
            matched_patterns=matched_patterns,
            details=details,
        )

    return InjectionCheckResult()


# --- Content Moderation (Item #28) ---

# Keyword-based content categories for dangerous content
_MODERATION_CATEGORIES: dict[str, list[re.Pattern[str]]] = {
    "weapons_explosives": [
        re.compile(
            r"(?i)\b(?:how\s+to\s+(?:make|build|create|construct|assemble))\b.*\b(?:bomb|explosive|detonator|IED|pipe\s+bomb|grenade|napalm|thermite)\b"
        ),
        re.compile(
            r"(?i)\b(?:instructions?\s+(?:for|to)\s+(?:make|build|create))\b.*\b(?:weapon|firearm|gun)\b"
        ),
    ],
    "self_harm": [
        re.compile(
            r"(?i)\b(?:how\s+to|methods?\s+(?:of|for|to)|ways?\s+to)\b.*\b(?:kill\s+(?:myself|yourself|oneself)|commit\s+suicide|end\s+(?:my|your|one'?s)\s+life)\b"
        ),
        re.compile(
            r"(?i)\b(?:best|most\s+effective|painless|easy)\b.*\b(?:ways?|methods?)\b.*\b(?:to\s+(?:die|kill\s+(?:myself|yourself|oneself))|suicide)\b"
        ),
    ],
    "illegal_drugs": [
        re.compile(
            r"(?i)\b(?:how\s+to\s+(?:make|synthesize|cook|manufacture|produce))\b.*\b(?:methamphetamine|meth|fentanyl|heroin|cocaine|crack|LSD|MDMA|ecstasy)\b"
        ),
    ],
    "child_exploitation": [
        re.compile(
            r"(?i)\b(?:child|minor|underage|kid)\b.*\b(?:porn|sexual|explicit|nude|naked)\b"
        ),
        re.compile(
            r"(?i)\b(?:CSAM|CP)\b"
        ),
    ],
}


@dataclass
class ModerationResult:
    """Result of content moderation check."""

    flagged: bool = False
    categories: list[str] = field(default_factory=list)
    details: str = ""

    def __bool__(self) -> bool:
        return self.flagged


def check_content_moderation(
    prompt: str,
    enabled_categories: list[str] | None = None,
) -> ModerationResult:
    """Check prompt content against moderation rules.

    Performs keyword-based scanning for dangerous content categories.

    Args:
        prompt: The text to check.
        enabled_categories: List of category names to check. If None, checks all.

    Returns:
        ModerationResult with details of any flagged content.
    """
    if not prompt:
        return ModerationResult()

    # Fast prefilter: skip regex work if no obvious moderation-related tokens.
    prompt_lower = prompt.lower()
    if not any(
        token in prompt_lower
        for token in (
            "bomb",
            "explosive",
            "weapon",
            "gun",
            "suicide",
            "kill myself",
            "meth",
            "fentanyl",
            "heroin",
            "cocaine",
            "child",
            "underage",
            "porn",
            "csam",
        )
    ):
        return ModerationResult()

    categories_to_check = enabled_categories or list(_MODERATION_CATEGORIES.keys())
    flagged_categories: list[str] = []

    for category in categories_to_check:
        patterns = _MODERATION_CATEGORIES.get(category, [])
        for pattern in patterns:
            if pattern.search(prompt):
                flagged_categories.append(category)
                break

    if flagged_categories:
        details = f"Content flagged in categories: {', '.join(flagged_categories)}"
        logger.warning(
            "Content moderation triggered: categories=%s",
            flagged_categories,
        )
        return ModerationResult(
            flagged=True,
            categories=flagged_categories,
            details=details,
        )

    return ModerationResult()


async def call_moderation_webhook(
    prompt: str,
    webhook_url: str,
    timeout: float = 5.0,
) -> ModerationResult:
    """Call an external moderation webhook for content review.

    The webhook receives a POST with JSON body:
        {"prompt": "<text>", "source": "smarterrouter"}

    Expected response (JSON):
        {"flagged": bool, "categories": [...], "details": "..."}

    Args:
        prompt: The text to moderate.
        webhook_url: URL of the moderation webhook.
        timeout: Request timeout in seconds.

    Returns:
        ModerationResult from the webhook response.
    """
    try:
        from router.config import settings

        async with httpx.AsyncClient(timeout=timeout, verify=settings.verify_tls) as client:
            response = await client.post(
                webhook_url,
                json={"prompt": prompt, "source": "smarterrouter"},
            )
            response.raise_for_status()
            data = response.json()

            return ModerationResult(
                flagged=data.get("flagged", False),
                categories=data.get("categories", []),
                details=data.get("details", ""),
            )
    except httpx.TimeoutException:
        logger.warning("Moderation webhook timed out: url=%s", webhook_url)
        return ModerationResult()
    except Exception:
        logger.error("Moderation webhook error: url=%s", webhook_url, exc_info=True)
        return ModerationResult()
