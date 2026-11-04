"""UrduNLP Toolkit library exports."""

from src.urdu_nlp.detector import identify_script
from src.urdu_nlp.normalizer import UrduNormalizer, normalize
from src.urdu_nlp.tokenizer import split_sentences, tokenize

__all__ = [
    "identify_script",
    "normalize",
    "UrduNormalizer",
    "tokenize",
    "split_sentences",
]

