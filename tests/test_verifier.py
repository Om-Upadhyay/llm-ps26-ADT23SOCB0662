import pytest
from math_verifier import verify_answer

@pytest.mark.parametrize(("expression", "answer"), [
    ("18 + 7 - 5", "20 apples"),
    ("48 / 6", "8 pencils"),
    ("80 * (1 - 15 / 100)", "$68"),
    ("60 * 2.5", "150 km"),
    ("(26 - 5) / 3", "7"),
])
def test_verifies_correct_arithmetic(expression, answer):
    assert verify_answer(expression, answer).status == "verified"

def test_detects_mismatch():
    assert verify_answer("6 * 7", "41").status == "failed"

@pytest.mark.parametrize("expression", ["", "NONE", "__import__('os').system('whoami')", "2 / 0", "x + 2"])
def test_unsupported_or_invalid_expression_is_not_verified(expression):
    assert verify_answer(expression, "2").status == "unavailable"

def test_ambiguous_final_answer_is_unavailable():
    assert verify_answer("2 + 2", "4 or 5").status == "unavailable"
