"""Unit tests for the multi-stage Urdu & Roman Urdu normalization pipeline."""

from src.urdu_nlp.normalizer import UrduNormalizer, normalize


def test_arabic_to_urdu_character_unification():
    """Arabic letter forms (Kaf, Yeh, Teh Marbuta, Tatweel) must be mapped to Urdu forms."""
    normalizer = UrduNormalizer(strip_diacritics=True)
    # Arabic: ك (U+0643), ي (U+064A), ة (U+0629), ـ (U+0640 tatweel)
    arabic_raw = "كتابـي جميـلة"
    result = normalizer.normalize(arabic_raw)

    # Must be unified to Urdu: ک, ی, ہ, with tatweel stripped
    assert "ك" not in result.text
    assert "ي" not in result.text
    assert "ة" not in result.text
    assert "ـ" not in result.text
    assert "کتابی" in result.text
    assert any(c.rule == "arabic_to_urdu_char" for c in result.changes)


def test_diacritic_stripping():
    """Zabar, Zer, Pesh, Tashdeed must be removed when strip_diacritics is True."""
    normalizer = UrduNormalizer(strip_diacritics=True)
    # بَہُتْ عُمْدَہ (with zabar, sukun, pesh)
    diacritics_raw = "بَہُتْ عُمْدَہ"
    result = normalizer.normalize(diacritics_raw)
    assert result.text == "بہت عمدہ"
    assert any(c.rule == "strip_diacritics" for c in result.changes)


def test_roman_urdu_elongation_compression():
    """Runs of 3+ identical letters must collapse to max_repeated_chars (default 2)."""
    normalizer = UrduNormalizer(max_repeated_chars=2)
    raw = "bohattt achhaaa zabardastttt"
    result = normalizer.normalize(raw)

    # 'bohattt' -> 'bohatt' -> mapped to 'bohat' in lexicon
    # 'achhaaa' -> 'achhaa' -> mapped to 'acha'
    assert "ttt" not in result.text
    assert "aaa" not in result.text
    assert any(c.rule == "elongation_compression" for c in result.changes)


def test_roman_urdu_lexicon_canonical_mapping():
    """Frequent spelling variants must map to canonical lexicon forms."""
    raw = "bht achha mobile hai nhi khareedna"
    result = normalize(raw)

    # bht -> bohat, achha -> acha, nhi -> nahi
    assert "bohat" in result.text
    assert "acha" in result.text
    assert "nahi" in result.text
    assert any(c.rule == "canonical_spelling" for c in result.changes)


def test_entity_masking_preservation():
    """URLs and email addresses must be replaced with atomic tokens."""
    raw = "Check our catalog at https://daraz.pk/deals or email support@shop.pk thanks"
    result = normalize(raw)

    assert "<URL>" in result.text
    assert "<EMAIL>" in result.text
    assert "https://" not in result.text
    assert any(c.rule == "mask_entity" for c in result.changes)

