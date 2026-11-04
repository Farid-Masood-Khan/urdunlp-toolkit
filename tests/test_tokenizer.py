"""Unit tests for tokenization and sentence segmentation."""

from src.urdu_nlp.tokenizer import split_sentences, tokenize


def test_tokenize_separates_punctuation():
    """Punctuation marks must be detached from adjacent word tokens."""
    text = "ye product acha hai, lekin delivery late thi!"
    tokens = tokenize(text)
    assert "," in tokens
    assert "!" in tokens
    assert "hai," not in tokens
    assert "thi!" not in tokens


def test_tokenize_urdu_script_punctuation():
    """Urdu khatam (۔) and comma (،) must be detached."""
    text = "یہ موبائل اچھا ہے، لیکن مہنگا ہے۔"
    tokens = tokenize(text)
    assert "،" in tokens
    assert "۔" in tokens
    assert "ہے،" not in tokens
    assert "ہے۔" not in tokens


def test_tokenize_preserves_atomic_placeholders():
    """Masked entities like <URL> must never be split on internal characters."""
    text = "check <URL> and contact <USER> please."
    tokens = tokenize(text)
    assert "<URL>" in tokens
    assert "<USER>" in tokens
    assert "." in tokens


def test_split_sentences_urdu_and_roman():
    """Sentences must be accurately split on sentence-final punctuation."""
    text = "Pehla jumla acha hai. Doosra jumla thora lamba hai! Teesri baat bhi sun lein۔"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Pehla jumla acha hai."
    assert sentences[1] == "Doosra jumla thora lamba hai!"
    assert sentences[2] == "Teesri baat bhi sun lein۔"

