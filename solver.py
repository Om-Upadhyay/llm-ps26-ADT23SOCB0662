"""Ollama HTTP client and response validation."""
import re
import requests

from config import MAX_QUESTION_CHARS, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS
from prompts import SYSTEM_PROMPT, make_user_prompt


class SolverError(Exception):
    """A user-displayable model or connection error."""


_ALLOWED_ARITHMETIC = re.compile(r"^[0-9+\-*/().\s]+$")


def _extract_response_text(body: dict) -> str:
    if not isinstance(body, dict):
        raise SolverError("Ollama returned an invalid response. Check the local Ollama service and try again.")
    api_error = body.get("error")
    if api_error:
        message = str(api_error)
        if "not found" in message.lower() and "model" in message.lower():
            raise SolverError(f"Model '{OLLAMA_MODEL}' is not available. Confirm it with `ollama list` or run `ollama pull {OLLAMA_MODEL}`.")
        raise SolverError(f"Ollama reported an error: {message}")
    text = body.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise SolverError("Ollama returned an empty or unrecognized response. Try again or check the model logs.")


def _clean_math_text(value: str) -> str:
    value = re.sub(r"\\text(?:rm)?\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\boxed{", "")
    for token in (r"\[", r"\]", r"\(", r"\)", r"\left", r"\right"):
        value = value.replace(token, "")
    value = value.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    value = value.replace("×", "*").replace("÷", "/")
    value = value.replace("\\", "").replace("{", "").replace("}", "")
    return " ".join(value.strip().split())


def _answer_from_section(text: str) -> str:
    boxed = re.search(r"\\boxed\{(.+)", text, flags=re.DOTALL)
    if boxed:
        candidate = boxed.group(1)
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip() and line.strip() not in {r"\[", r"\]"}]
        candidate = lines[-1] if lines else ""
        numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?", candidate)
        if len(numbers) != 1 and not re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", candidate):
            raise SolverError("The model did not provide one clear final value. Please try again.")
    candidate = _clean_math_text(candidate).strip(" $:;,.\t")
    while candidate.lower().startswith("final answer:"):
        candidate = candidate.split(":", 1)[1].strip()
    if not candidate:
        raise SolverError("The model returned an empty final answer. Please try again.")
    return candidate


def _has_operation(expression: str) -> bool:
    return any(operator in expression for operator in ("+", "*", "/")) or "-" in expression[1:]


def _expressions_from_equation_line(raw_line: str) -> list[str]:
    line = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", raw_line)
    line = _clean_math_text(line)
    expressions = []
    parts = line.split("=")
    for part_index in range(len(parts) - 2, -1, -1):
        lhs, rhs = parts[part_index], parts[part_index + 1]
        if not re.search(r"\d", rhs):
            continue
        number = r"(?:\d+(?:\.\d+)?|\.\d+)"
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){number}(?:\s*[+\-*/]\s*{number})*")
        for match in reversed(list(pattern.finditer(lhs))):
            candidate = " ".join(match.group(0).split())
            if _ALLOWED_ARITHMETIC.fullmatch(candidate) and _has_operation(candidate):
                expressions.append(candidate)
    return expressions


def _verification_section_candidates(text: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"\s*Verification expression\s*:\s*(.*)", line, flags=re.IGNORECASE)
        if not match:
            continue
        section = ([match.group(1).strip()] if match.group(1).strip() else []) + lines[index + 1:]
        candidates = []
        for raw_line in section:
            if re.match(r"\s*(Final answer|Understanding|Given values|Need to find|Equation/formula|Steps)\s*:", raw_line, flags=re.IGNORECASE):
                break
            expression = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", raw_line)
            expression = _clean_math_text(expression).strip(" $:;,.\t")
            if "=" in expression:
                candidates.extend(_expressions_from_equation_line(raw_line))
            elif _ALLOWED_ARITHMETIC.fullmatch(expression) and _has_operation(expression):
                candidates.append(" ".join(expression.split()))
        return list(dict.fromkeys(candidates))
    return []


def _verification_candidates(text: str) -> list[str]:
    section_candidates = _verification_section_candidates(text)
    equation_candidates = []
    for raw_line in reversed(text.splitlines()):
        equation_candidates.extend(_expressions_from_equation_line(raw_line))
    return list(dict.fromkeys(section_candidates + equation_candidates))


def _verification_expression(text: str) -> str:
    candidates = _verification_candidates(text)
    return candidates[0] if candidates else ""


def _normalize_final_answer(candidate: str) -> str:
    candidate = _clean_math_text(candidate).strip(" $:;,.\t")
    if re.search(r"\bor\b", candidate, flags=re.IGNORECASE):
        raise SolverError("The model's final answer is ambiguous. Please try the problem again or clarify it.")
    number_pattern = r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
    numbers = list(re.finditer(number_pattern, candidate))
    if not numbers:
        return candidate
    match = numbers[-1]
    number = match.group(0).replace(",", "")
    before = candidate[:match.start()].rstrip()
    after = candidate[match.end():].strip(" .,:;!?")
    prefix = ""
    currency = re.search(r"(Rs\.?|₹|\$)\s*$", before, flags=re.IGNORECASE)
    if currency:
        prefix = currency.group(1) + " "
    elif before.endswith("$"):
        prefix = "$"
    unit_match = re.match(r"([A-Za-z%₹$]+(?:/[A-Za-z]+)?)", after)
    unit = unit_match.group(1) if unit_match else ""
    return f"{prefix}{number}{(' ' + unit) if unit else ''}".strip()

def _format_solution(text: str) -> dict[str, str]:
    nonblank_lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_index = next((i for i in range(len(nonblank_lines) - 1, -1, -1) if nonblank_lines[i].lower().startswith("final answer:")), None)
    if final_index is not None:
        final_line_content = nonblank_lines[final_index].split(":", 1)[1].strip()
        final_answer = (_clean_math_text(final_line_content).strip(" $:;,.\t")
                        if final_line_content else
                        _answer_from_section("\n".join(nonblank_lines[final_index + 1:])))
    else:
        boxed = re.search(r"\\boxed\{", text)
        if boxed:
            final_answer = _answer_from_section(text[boxed.start():])
        else:
            final_answer = _answer_from_section("\n".join(nonblank_lines[-1:]))
    final_answer = _normalize_final_answer(final_answer)
    while final_answer.lower().startswith("final answer:"):
        final_answer = final_answer.split(":", 1)[1].strip()
    if not final_answer:
        raise SolverError("The model returned an empty final answer. Please try again.")

    solution_lines = text.rstrip().splitlines()
    last_index = next((i for i in range(len(solution_lines) - 1, -1, -1) if solution_lines[i].strip()), None)
    if last_index is not None and solution_lines[last_index].strip().lower().startswith("final answer:"):
        solution_lines[last_index] = f"Final answer: {final_answer}"
    else:
        solution_lines.extend(["", f"Final answer: {final_answer}"])
    return {"solution": "\n".join(solution_lines).rstrip(), "final_answer": final_answer,
            "verification_expression": _verification_expression(text)}


def _request_verification_expression(problem: str, final_answer: str) -> str:
    """Ask the same local model for a strict arithmetic expression if needed."""
    payload = {
        "model": OLLAMA_MODEL,
        "system": "You check arithmetic for a math tutor. Return exactly one line: Verification expression: <numeric arithmetic expression>. Use only numbers, decimal points, parentheses, +, -, *, and /. The expression must use the problem data and evaluate to the proposed numeric answer. Do not include code or explanation. If no safe arithmetic expression applies, return exactly: Verification expression: NONE",
        "prompt": f"Problem: {problem}\nProposed final answer: {final_answer}",
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 96},
    }
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
        text = _extract_response_text(body)
    except (requests.exceptions.RequestException, ValueError, SolverError):
        return ""
    for line in text.splitlines():
        match = re.match(r"\s*Verification expression\s*:\s*(.*)", line, flags=re.IGNORECASE)
        if not match:
            continue
        expression = match.group(1).strip().strip("`$")
        if expression.upper() == "NONE":
            return ""
        expression = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expression)
        expression = _clean_math_text(expression).strip(" $:;,.\t")
        if "=" in expression:
            expression = expression.split("=", 1)[0].strip()
        if _ALLOWED_ARITHMETIC.fullmatch(expression) and _has_operation(expression):
            return " ".join(expression.split())
    return ""

def _http_error_details(exc: requests.exceptions.HTTPError) -> str:
    response = exc.response
    if response is None:
        return "No response details were provided."
    try:
        body = response.json()
    except ValueError:
        return (response.text or "No response details were provided.")[:500]
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    return (response.text or "No response details were provided.")[:500]


def solve_problem(problem: str) -> dict[str, str]:
    if not isinstance(problem, str) or not problem.strip():
        raise SolverError("Enter a mathematical word problem before solving.")
    if len(problem) > MAX_QUESTION_CHARS:
        raise SolverError(f"Please keep the problem under {MAX_QUESTION_CHARS} characters.")
    if not OLLAMA_BASE_URL:
        raise SolverError("OLLAMA_BASE_URL is empty. Set it to your local Ollama address, usually http://localhost:11434.")
    if not OLLAMA_MODEL:
        raise SolverError("OLLAMA_MODEL is empty. Set it to the model name shown by `ollama list`.")

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "system": SYSTEM_PROMPT,
               "prompt": make_user_prompt(problem), "stream": False,
               "keep_alive": "5m", "options": {"temperature": 0.2, "num_predict": 768}}
    execution_mode = "auto"
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise SolverError(
            f"The local model did not respond within {OLLAMA_TIMEOUT_SECONDS:g} seconds. "
            "The first request can take longer while the model loads. Increase OLLAMA_TIMEOUT_SECONDS or try again."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise SolverError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Confirm Ollama is installed, start the Ollama app (or run `ollama serve`), then try again."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        detail = _http_error_details(exc)
        lowered = detail.lower()
        if status == 404:
            raise SolverError(
                f"Ollama could not find model '{OLLAMA_MODEL}'. Confirm the exact name with `ollama list` or run `ollama pull {OLLAMA_MODEL}`."
            ) from exc
        if status == 500 and "cuda" in lowered and (
            "shared object initialization failed" in lowered or "process has terminated" in lowered
        ):
            cpu_payload = {**payload, "options": {**payload["options"], "num_gpu": 0}}
            try:
                response = requests.post(url, json=cpu_payload, timeout=OLLAMA_TIMEOUT_SECONDS)
                response.raise_for_status()
                execution_mode = "cpu_fallback"
            except requests.exceptions.Timeout as cpu_exc:
                raise SolverError(
                    "Ollama's GPU runner failed to initialize, and the CPU fallback timed out. Increase OLLAMA_TIMEOUT_SECONDS or try again."
                ) from cpu_exc
            except requests.exceptions.HTTPError as cpu_exc:
                cpu_detail = _http_error_details(cpu_exc)
                raise SolverError(
                    f"Ollama's GPU runner failed to initialize, and CPU fallback returned HTTP {cpu_exc.response.status_code}: {cpu_detail}"
                ) from cpu_exc
            except requests.exceptions.RequestException as cpu_exc:
                raise SolverError("Ollama's GPU runner failed to initialize, and the CPU fallback request also failed.") from cpu_exc
        else:
            raise SolverError(f"Ollama returned HTTP {status}: {detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise SolverError("The local Ollama request failed. Check the server address and try again.") from exc

    try:
        body = response.json()
    except ValueError as exc:
        raise SolverError("Ollama returned invalid JSON. Check the local service and try again.") from exc
    result = _format_solution(_extract_response_text(body))
    from math_verifier import verify_answer
    candidates = _verification_candidates(result["solution"])
    for candidate in candidates:
        if verify_answer(candidate, result["final_answer"]).status == "verified":
            result["verification_expression"] = candidate
            break
    else:
        follow_up = _request_verification_expression(problem, result["final_answer"])
        explicit = _verification_section_candidates(result["solution"])
        result["verification_expression"] = follow_up or (explicit[0] if explicit else "")
    result["execution_mode"] = execution_mode
    return result


