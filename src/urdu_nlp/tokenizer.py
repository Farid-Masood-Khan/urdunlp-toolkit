"""Punctuation-aware tokenizer and sentence segmenter for Urdu and Roman Urdu.

References: TRD §2.3 & PRD FR-5.
"""

from __future__ import annotations

import re

# Masked placeholder pattern to treat as indivisible atomic tokens
PLACEHOLDER_REGEX = re.compile(r"<(URL|EMAIL|USER|HASHTAG)>")

# Punctuation to detach from word boundaries
# Urdu: ۔ ، ؛ ؟ ' " ; Roman: . , ! ? ' " ; : ( ) [ ]
PUNCT_SPLIT_REGEX = re.compile(r"([۔،؛؟\.,!\?:;\"'\(\)\[\]\{\}]+)")

# Sentence terminators: Urdu khatam (۔), Urdu question mark (؟), Latin period, exclamation, question mark
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[۔\.!\?؟])\s+")


def tokenize(text: str) -> list[str]:
    """Tokenizes text into words and detached punctuation, keeping placeholders intact.

    Examples:
        "ye acha hai!" -> ["ye", "acha", "hai", "!"]
        "یہ بہت عمدہ ہے۔" -> ["یہ", "بہت", "عمدہ", "ہے", "۔"]
        "Check this <URL> please." -> ["Check", "this", "<URL>", "please", "."]
    """
    if not text or not text.strip():
        return []

    tokens: list[str] = []
    # Split by whitespace first
    raw_words = text.split()

    for word in raw_words:
        # If the word is an exact placeholder, keep it atomic
        if PLACEHOLDER_REGEX.fullmatch(word):
            tokens.append(word)
            continue

        # If the word contains a placeholder alongside attached punctuation (e.g. "<URL>.")
        parts = PLACEHOLDER_REGEX.split(word)
        if len(parts) > 1:
            for i, part in enumerate(parts):
                if not part:
                    continue
                if i % 2 == 1:
                    # Captured placeholder name: e.g. 'URL' -> '<URL>'
                    tokens.append(f"<{part}>")
                else:
                    # Detach punctuation from surrounding text
                    sub_pieces = [p for p in PUNCT_SPLIT_REGEX.split(part) if p.strip()]
                    tokens.extend(sub_pieces)
            continue

        # Standard word: detach leading/trailing punctuation
        sub_pieces = [p for p in PUNCT_SPLIT_REGEX.split(word) if p.strip()]
        tokens.extend(sub_pieces)

    return tokens


def split_sentences(text: str) -> list[str]:
    """Segments paragraph into sentences using Urdu and Roman terminal delimiters."""
    if not text or not text.strip():
        return []

    # Split by terminal punctuation followed by space
    raw_sentences = SENTENCE_SPLIT_REGEX.split(text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences
