"""Focused unit tests for Ollama response and failure handling."""
import requests
import pytest
import solver

class FakeResponse:
    def __init__(self, body=None, status=200):
        self.body = body
        self.status_code = status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)
    def json(self):
        return self.body

VALID_TEXT = """Understanding: Find the cost of twelve pens.
Given values: 5 pens cost Rs. 100.
Need to find: Cost of 12 pens.
Equation/formula: unit cost = total cost / number of pens.
Steps:
1. 100 / 5 = 20 rupees per pen.
2. 20 * 12 = 240 rupees.
Verification expression: 100 / 5 * 12
Final answer: Rs. 240"""

RAW_QWEN_TEXT = r"""Understanding: Find the cost of twelve pens.
Given values: Five pens cost Rs. 100.
Need to find: Cost of twelve pens.
Equation/formula: Cost is proportional to quantity.
Steps:
1. 100 / 5 = 20 Rs per pen.
2. 20 * 12 = 240 Rs.
Verification expression:
The arithmetic expression is:
\[
\frac{100}{5} \times 12 = 240
\]
Final answer:
\[
\boxed{240 \text{ Rs}}
\]"""

def test_sends_configured_model_to_local_generate_api(monkeypatch):
    seen = {}
    def fake_post(url, json, timeout):
        seen.update(url=url, payload=json, timeout=timeout)
        return FakeResponse({"response": VALID_TEXT, "done": True})
    monkeypatch.setattr(solver.requests, "post", fake_post)
    result = solver.solve_problem("A shopkeeper sells 5 pens for Rs. 100. What is the cost of 12 pens?")
    assert result["final_answer"] == "Rs. 240"
    assert result["verification_expression"] == "100 / 5 * 12"
    assert seen["url"].endswith("/api/generate")
    assert seen["payload"]["model"] == solver.OLLAMA_MODEL
    assert seen["payload"]["stream"] is False

def test_parses_qwen_boxed_final_answer_and_latex_expression():
    result = solver._format_solution(RAW_QWEN_TEXT)
    assert result["final_answer"] == "240 Rs"
    assert result["verification_expression"] == "20 * 12"
    assert result["solution"].splitlines()[-1] == "Final answer: 240 Rs"

def test_rejects_empty_and_too_long_input(monkeypatch):
    with pytest.raises(solver.SolverError, match="Enter a mathematical"):
        solver.solve_problem("  ")
    monkeypatch.setattr(solver, "MAX_QUESTION_CHARS", 3)
    with pytest.raises(solver.SolverError, match="under 3 characters"):
        solver.solve_problem("long")

def test_reports_server_unavailable(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(solver.requests, "post", fail)
    with pytest.raises(solver.SolverError, match="Cannot connect to Ollama"):
        solver.solve_problem("What is 2 + 2?")

def test_reports_timeout(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("slow")
    monkeypatch.setattr(solver.requests, "post", fail)
    with pytest.raises(solver.SolverError, match="first request can take longer"):
        solver.solve_problem("What is 2 + 2?")

def test_reports_missing_model(monkeypatch):
    def missing(*args, **kwargs):
        response = FakeResponse({"error": "model not found"}, status=404)
        raise requests.HTTPError("404", response=response)
    monkeypatch.setattr(solver.requests, "post", missing)
    with pytest.raises(solver.SolverError, match="could not find model"):
        solver.solve_problem("What is 2 + 2?")

def test_reports_malformed_api_body(monkeypatch):
    monkeypatch.setattr(solver.requests, "post", lambda *args, **kwargs: FakeResponse({"unexpected": "shape"}))
    with pytest.raises(solver.SolverError, match="empty or unrecognized"):
        solver.solve_problem("What is 2 + 2?")

def test_normalizes_clear_unlabelled_numeric_answer(monkeypatch):
    monkeypatch.setattr(solver.requests, "post", lambda *args, **kwargs: FakeResponse({"response": "The answer is 4"}))
    result = solver.solve_problem("What is 2 + 2?")
    assert result["final_answer"] == "4"
    assert result["solution"].splitlines()[-1] == "Final answer: 4"

def test_handles_invalid_json(monkeypatch):
    class InvalidJson(FakeResponse):
        def json(self):
            raise ValueError("not json")
    monkeypatch.setattr(solver.requests, "post", lambda *args, **kwargs: InvalidJson())
    with pytest.raises(solver.SolverError, match="invalid JSON"):
        solver.solve_problem("What is 2 + 2?")

def test_falls_back_to_calculation_in_steps_when_verification_section_has_no_expression():
    text = """Understanding: Find the total.
Given values: 5 pens cost Rs. 100.
Steps:
1. The unit cost is 100 / 5 = 20 Rs.
2. Then 20 \\times 12 = 240 Rs.
Verification expression: The arithmetic is shown in the steps above.
Final answer: Rs. 240"""
    result = solver._format_solution(text)
    assert result["verification_expression"] == "20 * 12"
    assert result["final_answer"] == "Rs. 240"

def test_selects_equation_that_matches_answer_when_model_lists_intermediate_results(monkeypatch):
    text = """Understanding: Calculate a sale price.
Given values: Original price is 80; discount is 15 percent.
Steps:
1. 80 * 0.15 = 12 dollars discount.
2. 80 - 12 = 68 dollars sale price.
Verification expression: 80 * 0.15
Final answer: $68"""
    monkeypatch.setattr(solver.requests, "post", lambda *args, **kwargs: FakeResponse({"response": text}))
    result = solver.solve_problem("An $80 item is discounted by 15%. What is the sale price?")
    assert result["verification_expression"] == "80 - 12"

def test_normalizes_a_boxed_answer_without_a_heading(monkeypatch):
    text = r"Reasoning complete. The cost is \boxed{240 \text{ Rs}}."
    monkeypatch.setattr(solver.requests, "post", lambda *args, **kwargs: FakeResponse({"response": text}))
    result = solver.solve_problem("What is the cost?")
    assert result["final_answer"] == "240 Rs"
    assert result["solution"].splitlines()[-1] == "Final answer: 240 Rs"

def test_rejects_ambiguous_answer_text(monkeypatch):
    monkeypatch.setattr(solver.requests, "post", lambda *args, **kwargs: FakeResponse({"response": "Final answer: 4 or 5"}))
    with pytest.raises(solver.SolverError, match="ambiguous"):
        solver.solve_problem("An unclear problem")

def test_requests_a_local_verification_expression_when_initial_answer_has_none(monkeypatch):
    calls = []
    initial = """Understanding: Add two numbers.
Given values: 2 and 2.
Steps: The total is four.
Final answer: 4"""
    responses = [
        FakeResponse({"response": initial}),
        FakeResponse({"response": "Verification expression: 2 + 2"}),
    ]
    def fake_post(url, json, timeout):
        calls.append(json)
        return responses.pop(0)
    monkeypatch.setattr(solver.requests, "post", fake_post)
    result = solver.solve_problem("What is 2 + 2?")
    assert len(calls) == 2
    assert calls[1]["model"] == solver.OLLAMA_MODEL
    assert result["verification_expression"] == "2 + 2"

def test_retries_cuda_runner_failure_on_cpu(monkeypatch):
    calls = []
    def fake_post(url, json, timeout):
        calls.append(json)
        if len(calls) == 1:
            raise requests.HTTPError(
                "HTTP 500", response=FakeResponse(
                    {"error": "llama-server process has terminated: CUDA error: shared object initialization failed"},
                    status=500,
                )
            )
        return FakeResponse({"response": VALID_TEXT})
    monkeypatch.setattr(solver.requests, "post", fake_post)
    result = solver.solve_problem("A shopkeeper sells 5 pens for Rs. 100. What is the cost of 12 pens?")
    assert len(calls) == 2
    assert calls[0]["options"].get("num_gpu") is None
    assert calls[1]["options"]["num_gpu"] == 0
    assert result["execution_mode"] == "cpu_fallback"
    assert result["final_answer"] == "Rs. 240"

def test_includes_ollama_error_detail_for_other_server_errors(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.HTTPError("HTTP 500", response=FakeResponse({"error": "runner crashed"}, status=500))
    monkeypatch.setattr(solver.requests, "post", fail)
    with pytest.raises(solver.SolverError, match="runner crashed"):
        solver.solve_problem("What is 2 + 2?")
