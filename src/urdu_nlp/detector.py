"""Script identification for Urdu script, Roman Urdu, and code-switched text.

References: TRD §2.1 & PRD FR-1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ScriptDetectionResult:
    """Detailed script classification outcome with character distribution evidence."""

    script: str  # 'urdu_script' | 'roman_urdu' | 'mixed'
    urdu_ratio: float
    latin_ratio: float
    urdu_chars: int
    latin_chars: int
    total_evaluated_chars: int


# Unicode ranges for Perso-Arabic scripts (Urdu, Arabic, extended Nastaliq)
# Arabic (0600-06FF), Arabic Supplement (0750-077F), Arabic Presentation Forms A & B
URDU_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_CHAR_PATTERN = re.compile(r"[a-zA-Z]")
MASKED_TOKEN_PATTERN = re.compile(r"<(URL|EMAIL|USER|HASHTAG)>")


def identify_script(text: str) -> ScriptDetectionResult:
    """Identifies whether an input string is written in Urdu script, Roman Urdu, or mixed.

    Rule definitions (TRD §2.1):
    - urdu_ratio > 0.50 -> 'urdu_script'
    - latin_ratio > 0.90 -> 'roman_urdu'
    - Otherwise -> 'mixed'
    """
    if not text or not text.strip():
        return ScriptDetectionResult(
            script="mixed",
            urdu_ratio=0.0,
            latin_ratio=0.0,
            urdu_chars=0,
            latin_chars=0,
            total_evaluated_chars=0,
        )

    # Strip masked placeholder tokens before computing script ratios
    clean_text = MASKED_TOKEN_PATTERN.sub("", text)

    urdu_matches = URDU_CHAR_PATTERN.findall(clean_text)
    latin_matches = LATIN_CHAR_PATTERN.findall(clean_text)

    n_urdu = len(urdu_matches)
    n_latin = len(latin_matches)
    total_alpha = n_urdu + n_latin

    if total_alpha == 0:
        return ScriptDetectionResult(
            script="mixed",
            urdu_ratio=0.0,
            latin_ratio=0.0,
            urdu_chars=0,
            latin_chars=0,
            total_evaluated_chars=0,
        )

    urdu_ratio = round(n_urdu / total_alpha, 4)
    latin_ratio = round(n_latin / total_alpha, 4)

    if urdu_ratio > 0.50:
        script = "urdu_script"
    elif latin_ratio > 0.90:
        script = "roman_urdu"
    else:
        script = "mixed"

    return ScriptDetectionResult(
        script=script,
        urdu_ratio=urdu_ratio,
        latin_ratio=latin_ratio,
        urdu_chars=n_urdu,
        latin_chars=n_latin,
        total_evaluated_chars=total_alpha,
    )

