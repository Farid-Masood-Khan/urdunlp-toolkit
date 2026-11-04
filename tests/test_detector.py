"""Unit tests for Urdu and Roman Urdu script identification."""

from src.urdu_nlp.detector import identify_script


def test_detector_identifies_nastaliq_urdu():
    """Inputs with >50% Perso-Arabic characters must be labeled as urdu_script."""
    res = identify_script("یہ موبائل بہت بہترین ہے بیٹری ٹائمنگ شاندار ہے۔")
    assert res.script == "urdu_script"
    assert res.urdu_ratio > 0.50
    assert res.latin_ratio == 0.0


def test_detector_identifies_roman_urdu():
    """Inputs with >90% Latin alphabet characters must be labeled as roman_urdu."""
    res = identify_script("ye mobile bohat acha hai camera result bht kamaal hai")
    assert res.script == "roman_urdu"
    assert res.latin_ratio > 0.90
    assert res.urdu_ratio == 0.0


def test_detector_identifies_mixed_script():
    """Inputs with substantial mixtures of Arabic and Latin characters must be labeled as mixed."""
    res = identify_script("ye product بہت ہی خراب hai waste of money")
    assert res.script == "mixed"
    assert 0.0 < res.urdu_ratio < 0.50


def test_detector_excludes_masked_placeholders_from_script_ratio():
    """Masked entities like <URL> or <USER> should not distort script detection."""
    res = identify_script("یہ کتاب پڑھیں <URL> شکریہ")
    assert res.script == "urdu_script"
    # Only Arabic characters should be counted; the English 'URL' tag is ignored
    assert res.latin_chars == 0

