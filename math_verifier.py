"""Conservative arithmetic verification using SymPy, without Python eval."""
from dataclasses import dataclass
import re
from sympy import SympifyError, sympify
from sympy.core.expr import Expr

@dataclass(frozen=True)
class VerificationResult:
    status: str  # verified, failed, or unavailable
    details: str

_ALLOWED = re.compile(r"^[0-9+\-*/().\s]+$")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")

def _answer_number(answer: str):
    matches = _NUMBER.findall(answer.replace(",", ""))
    if len(matches) != 1:
        return None
    try:
        return sympify(matches[0], evaluate=True)
    except (SympifyError, TypeError, ValueError):
        return None

def verify_answer(expression: str, answer: str) -> VerificationResult:
    if not expression or expression.strip().upper() == "NONE":
        return VerificationResult("unavailable", "No safe arithmetic expression was provided.")
    expression = expression.strip()
    if len(expression) > 200 or not _ALLOWED.fullmatch(expression):
        return VerificationResult("unavailable", "Expression contains unsupported syntax; it was not evaluated.")
    try:
        value = sympify(expression, evaluate=True)
        if not isinstance(value, Expr) or value.free_symbols or value.is_real is not True or value.is_finite is not True:
            return VerificationResult("unavailable", "Expression did not produce a finite real number.")
        proposed = _answer_number(answer)
        if proposed is None:
            return VerificationResult("unavailable", "Could not safely extract one numeric value from the final answer.")
        if value.equals(proposed):
            return VerificationResult("verified", f"{expression} = {value}; matches the final answer.")
        return VerificationResult("failed", f"{expression} = {value}, but the final answer gives {proposed}.")
    except (SympifyError, ZeroDivisionError, TypeError, ValueError, SyntaxError):
        return VerificationResult("unavailable", "Expression is invalid or unsupported; it was not evaluated.")
