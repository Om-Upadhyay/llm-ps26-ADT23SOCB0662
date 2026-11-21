"""Deterministic arithmetic fixtures for the five sample problems."""
from math_verifier import verify_answer

def test_five_sample_problem_calculations():
    cases = [
        ("18 + 7 - 5", "20 apples"),
        ("48 / 6", "8 pencils"),
        ("80 * (1 - 15 / 100)", "$68"),
        ("60 * 2.5", "150 km"),
        ("(26 - 5) / 3", "7"),
    ]
    assert all(verify_answer(expression, answer).status == "verified" for expression, answer in cases)

