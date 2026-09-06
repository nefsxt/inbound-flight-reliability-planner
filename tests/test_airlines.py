import math

from src.airlines import extract_airline_code, airline_label


def test_extract_airline_code_basic():
    assert extract_airline_code("AEE1421") == "AEE"
    assert extract_airline_code("AEE1421 ") == "AEE"
    assert extract_airline_code("thy1zc") == "THY"


def test_extract_airline_code_missing_or_blank():
    assert extract_airline_code(None) is None
    assert extract_airline_code("") is None
    assert extract_airline_code("   ") is None
    assert extract_airline_code(float("nan")) is None


def test_extract_airline_code_no_letters():
    assert extract_airline_code("12345") is None


def test_airline_label_known_and_unknown():
    assert airline_label("AEE") == "Aegean Airlines"
    assert "Unmapped carrier" in airline_label("ZZZ")


def test_airline_label_none_and_nan():
    assert airline_label(None) == "Unknown carrier"
    assert airline_label(float("nan")) == "Unknown carrier"
