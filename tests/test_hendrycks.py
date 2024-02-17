import pytest

from src import hendrycks


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("\\boxed{test}", "test"),
        ("\\boxed{123}", "123"),
        ("not a boxed string", None),
        ("blah blah \\boxed{1234} blah blah", None),
        ("\\boxed{incomplete", None),
    ],
)
def test_remove_boxed(input_str, expected_output):
    assert hendrycks._remove_boxed(input_str) == expected_output


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("Some text \\boxed{test}", "\\boxed{test}"),
        ("\\boxed{first} and \\boxed{second}", "\\boxed{second}"),
        ("No boxed string here", None),
        ("Incomplete \\boxed{string", None),
    ],
)
def test_last_boxed_only_string(input_str, expected_output):
    assert hendrycks._last_boxed_only_string(input_str) == expected_output


@pytest.mark.parametrize(
    "str1, str2, expected_output",
    [
        ("test", "test", True),
        ("\\frac{1}{2}", "1/2", True),
        ("\\frac{1}{2}", "\\frac{2}{4}", False),
        (None, None, False),
        (None, "test", False),
    ],
)
def test_is_equiv(str1, str2, expected_output):
    assert hendrycks.is_equiv(str1, str2, verbose=True) == expected_output


@pytest.mark.parametrize(
    "str1, str2, expected",
    [
        # Test cases for parse_prediction
        ("\\boxed{1}", "blah blah \\boxed{1} blah blah", True),
        ("\\boxed{2}", "\\boxed{1}", False),
        ("\\boxed{\\frac{1}{2}}", "\\boxed{1/2}", True),
        ("\\boxed{1}", "blah blah", False),
        (None, "\\boxed{12}", False),
        (None, "blah blah", False),
    ],
)
def test_is_equiv_and_parse_prediction(str1, str2, expected):
    parsed_str1 = hendrycks.parse_prediction(str1)
    parsed_str2 = hendrycks.parse_prediction(str2)
    result = hendrycks.is_equiv(parsed_str1, parsed_str2)

    assert result == expected


@pytest.mark.expensive
def test_load_hendrycks():
    t1 = hendrycks.load_hendrycks(split="all")
    assert isinstance(t1, list)
    assert isinstance(t1[0], hendrycks.MathSample)
    assert len(t1) == 12500

    t2 = hendrycks.load_hendrycks(split="test")
    assert len(t2) == 5000

    t3 = hendrycks.load_hendrycks(split="test", subject="intermediate_algebra")
    assert len(t3) == 903

    t4 = hendrycks.load_hendrycks(split="test", subject="intermediate_algebra", level=5)
    assert len(t4) == 280
