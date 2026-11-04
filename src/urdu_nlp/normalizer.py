"""Multi-stage text normalizer for Urdu script and Roman Urdu.

References: TRD §2.2 & PRD FR-2, FR-3, FR-4.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.urdu_nlp.detector import identify_script


@dataclass(frozen=True)
class ChangeRecord:
    """Audit log entry capturing a specific text transformation."""

    stage: str
    original_token: str
    new_token: str
    rule: str


@dataclass
class NormalizedResult:
    """Output structure containing cleaned text and an audit trace of applied changes."""

    text: str
    script: str
    changes: list[ChangeRecord] = field(default_factory=list)
    original: str = ""


# Regex patterns for masking entities
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
USER_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")

# Arabic to Urdu character normalization map
ARABIC_TO_URDU_CHAR_MAP = {
    "\u0643": "\u06a9",  # ك -> ک (Kaf)
    "\u064a": "\u06cc",  # ي -> ی (Yeh)
    "\u0649": "\u06cc",  # ى -> ی (Alif Maksura -> Yeh)
    "\u0629": "\u06c1",  # ة -> ہ (Teh Marbuta -> Gol Heh)
    "\u06c0": "\u06c1",  # ۀ -> ہ
    "\u0640": "",  # Tatweel (Kashida)
}

# Diacritics regex (combining marks)
DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")

# Character elongation pattern: 3 or more repeated letters -> 2
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)


class UrduNormalizer:
    """Configurable normalization pipeline supporting Urdu script, Roman Urdu, and code-switched text."""

    def __init__(
        self,
        strip_diacritics: bool = True,
        max_repeated_chars: int = 2,
        lexicon_path: Path | str | None = None,
    ):
        self.strip_diacritics = strip_diacritics
        self.max_repeated_chars = max_repeated_chars
        self.lexicon_map: dict[str, str] = {}
        self.english_guard_set: set[str] = set()
        self._load_lexicon(lexicon_path)

    def _load_lexicon(self, lexicon_path: Path | str | None) -> None:
        """Loads canonical Roman Urdu mapping dictionary."""
        if lexicon_path is None:
            lexicon_path = Path(__file__).parent / "lexicon" / "roman_urdu_v1.yaml"
        else:
            lexicon_path = Path(lexicon_path)

        if lexicon_path.exists():
            with open(lexicon_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            mappings = data.get("mappings", {})
            for canonical, variants in mappings.items():
                self.lexicon_map[canonical.lower()] = canonical.lower()
                for v in variants:
                    self.lexicon_map[v.lower()] = canonical.lower()

            guards = data.get("english_guard_tokens", [])
            self.english_guard_set = {g.lower() for g in guards}

    def normalize(
        self,
        text: str,
        script_hint: str | None = None,
    ) -> NormalizedResult:
        """Applies ordered stages (TRD §2.2):

        1. Mask URLs, emails, mentions, hashtags
        2. Unicode NFC normalization
        3. Urdu script normalization (character unification, tatweel removal, diacritics)
        4. Roman Urdu cleanup (lowercasing, elongation compression)
        5. Canonical Roman Urdu lexicon mapping
        6. Produce structured change report
        """
        changes: list[ChangeRecord] = []
        original_text = text

        if not text:
            return NormalizedResult(text="", script="mixed", changes=[], original=original_text)

        # Stage 1: Masking
        s1_text = text
        for pat, tag in [
            (URL_PATTERN, "<URL>"),
            (EMAIL_PATTERN, "<EMAIL>"),
            (USER_PATTERN, "<USER>"),
            (HASHTAG_PATTERN, "<HASHTAG>"),
        ]:
            matches = pat.findall(s1_text)
            for m in matches:
                changes.append(
                    ChangeRecord(
                        stage="masking", original_token=m, new_token=tag, rule="mask_entity"
                    )
                )
            s1_text = pat.sub(tag, s1_text)

        # Stage 2: Unicode NFC Normalization
        s2_text = unicodedata.normalize("NFC", s1_text)
        if s2_text != s1_text:
            changes.append(
                ChangeRecord(
                    stage="unicode",
                    original_token="[non-NFC string]",
                    new_token="[NFC string]",
                    rule="unicode_nfc",
                )
            )

        # Detect script after masking and NFC
        detected_script = script_hint or identify_script(s2_text).script

        # Stage 3: Urdu Script Cleanup
        s3_text = s2_text
        if detected_script in ("urdu_script", "mixed"):
            cleaned_chars = []
            for ch in s3_text:
                if ch in ARABIC_TO_URDU_CHAR_MAP:
                    mapped = ARABIC_TO_URDU_CHAR_MAP[ch]
                    if mapped != ch:
                        changes.append(
                            ChangeRecord(
                                stage="urdu_script",
                                original_token=ch,
                                new_token=mapped or "[removed]",
                                rule="arabic_to_urdu_char",
                            )
                        )
                    cleaned_chars.append(mapped)
                else:
                    cleaned_chars.append(ch)
            s3_text = "".join(cleaned_chars)

            if self.strip_diacritics:
                stripped = DIACRITICS_PATTERN.sub("", s3_text)
                if stripped != s3_text:
                    changes.append(
                        ChangeRecord(
                            stage="urdu_script",
                            original_token="[diacritics]",
                            new_token="[stripped]",
                            rule="strip_diacritics",
                        )
                    )
                s3_text = stripped

            # Normalize Urdu punctuation spacing (۔ ، ؛ ؟)
            s3_text = re.sub(r"\s*([۔،؛؟])\s*", r" \1 ", s3_text)

        # Stage 4 & 5: Roman Urdu Cleanup and Canonical Lexicon Mapping
        s4_text = s3_text
        if detected_script in ("roman_urdu", "mixed"):
            tokens = s4_text.split()
            normalized_tokens = []

            for tok in tokens:
                # Do not modify placeholder tags
                if tok in ("<URL>", "<EMAIL>", "<USER>", "<HASHTAG>"):
                    normalized_tokens.append(tok)
                    continue

                lowered = tok.lower()

                # Stage 4: Elongation compression
                # Replace 3+ repeated characters with max_repeated_chars (default 2)
                compressed = ELONGATION_PATTERN.sub(r"\1\1", lowered)
                if compressed != lowered:
                    changes.append(
                        ChangeRecord(
                            stage="roman_cleanup",
                            original_token=tok,
                            new_token=compressed,
                            rule="elongation_compression",
                        )
                    )

                # Stage 5: Lexicon mapping
                # Clean punctuation boundary for lookup
                core_tok = re.sub(r"^[^\w]+|[^\w]+$", "", compressed)
                if core_tok in self.lexicon_map:
                    # English guard check (e.g. do not map English 'main' if part of English phrase)
                    if core_tok in self.english_guard_set and detected_script == "mixed":
                        # Preserve token as-is when in ambiguous mixed mode
                        final_tok = compressed
                    else:
                        canonical = self.lexicon_map[core_tok]
                        final_tok = compressed.replace(core_tok, canonical)
                        if final_tok != compressed:
                            changes.append(
                                ChangeRecord(
                                    stage="lexicon_mapping",
                                    original_token=compressed,
                                    new_token=final_tok,
                                    rule="canonical_spelling",
                                )
                            )
                else:
                    final_tok = compressed

                normalized_tokens.append(final_tok)

            s4_text = " ".join(normalized_tokens)

        # Final whitespace collapse
        final_text = re.sub(r"\s+", " ", s4_text).strip()

        return NormalizedResult(
            text=final_text,
            script=detected_script,
            changes=changes,
            original=original_text,
        )


# Global default normalizer instance
default_normalizer = UrduNormalizer()


def normalize(text: str, script_hint: str | None = None) -> NormalizedResult:
    """Convenience functional wrapper for text normalization."""
    return default_normalizer.normalize(text, script_hint=script_hint)
