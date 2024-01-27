import pytest
from src.hendrycks import remove_boxed, last_boxed_only_string, is_equiv


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("\\boxed{test}", "test"),
        ("\\boxed{123}", "123"),
        ("not a boxed string", None),
        ("blah blah \\boxed{1234} blah blah", "1234"),
        ("blah blah \\boxed{incomplete", None),
    ],
)
def test_remove_boxed(input_str, expected_output):
    assert remove_boxed(input_str) == expected_output


# Test cases for last_boxed_only_string
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
    assert last_boxed_only_string(input_str) == expected_output


# Test cases for is_equiv
@pytest.mark.parametrize(
    "str1, str2, expected_output",
    [("test", "test", True), ("\\frac{1}{2}", "1/2", True), ("\\frac{1}{2}", "\\frac{2}{4}", False), (None, None, True), (None, "test", False)],
)
def test_is_equiv(str1, str2, expected_output):
    assert is_equiv(str1, str2) == expected_output


# Add more test cases for other helper functions as needed
