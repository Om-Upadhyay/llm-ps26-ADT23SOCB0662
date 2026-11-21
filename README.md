# Math Word Problem Solver

**Student:** Om Upadhyay | **Roll:** 26 | **Problem Statement:** 26  
**Assignment:** 100 GenAI (LLM) Mini Project Problem Statements

## Problem statement

Students struggle with word problems. Build a tool that solves a word problem step by step and explains each step.

## Objective and features

This Streamlit app sends each problem to the local Ollama model `mightykatun/qwen2.5-math:7b`. It presents the model's understanding, known values, unknown, equation or formula, numbered steps, and final answer. SymPy independently checks a restricted arithmetic expression returned by the model against its numeric answer. The UI includes five examples, validation, and a clear button. No external LLM service or API key is used.

Verification is reported as verified, mismatch, or unavailable. The verifier accepts only a short expression made of numbers and arithmetic operators, uses SymPy rather than Python `eval`, and never labels unsupported expressions verified.

## Technology stack

- Python and Streamlit
- Ollama local HTTP API (`/api/generate`) with Qwen2.5-Math 7B
- SymPy for restricted arithmetic checks
- python-dotenv for optional local `.env` configuration
- requests for local HTTP calls
- pytest and ReportLab for tests and the procedure report

## Architecture and workflow

1. `app.py` validates input and renders the Streamlit interface.
2. `solver.py` sends the prompt and problem to `OLLAMA_BASE_URL/api/generate` using `OLLAMA_MODEL`; it handles connection errors, timeouts, missing models, invalid JSON, and malformed model answers. If Ollama's CUDA runner fails during initialization, it retries once with `num_gpu=0` on the local CPU.
3. `prompts.py` contains the reusable Qwen2.5-Math tutor instructions and requires `Final answer:` on the last non-empty line.
3. `prompts.py` contains the reusable Qwen2.5-Math tutor instructions and requires `Final answer:` on the last non-empty line.
4. If Qwen omits a safe calculation that matches its final answer, the solver asks the same local model once for a constrained numeric expression. It accepts only arithmetic characters before passing that expression to SymPy.
5. `math_verifier.py` accepts only a short arithmetic expression made from digits, decimal points, parentheses, and operators. It evaluates with SymPy and compares the result with the numeric final answer.

## Install and prepare Ollama (Windows)

1. Install Ollama using the [official Windows download](https://ollama.com/download/windows). The Ollama app normally provides the local server.
2. In PowerShell, confirm installation and model availability:

   ```powershell
   ollama --version
   ollama list
   ```

3. If the model is not listed, download it once:

   ```powershell
   ollama pull mightykatun/qwen2.5-math:7b
   ```

   Do not run `ollama pull` if the model is already present. Check the local API at `http://localhost:11434/api/tags`. If Ollama is installed but its server is not running, open the Ollama app or run `ollama serve` in a terminal.

## Install the Python app

Python 3.10 or later is recommended.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The defaults in `.env.example` point to `http://localhost:11434` and `mightykatun/qwen2.5-math:7b`. No API key is needed. Optional settings:

- `OLLAMA_BASE_URL` - local Ollama server URL
- `OLLAMA_MODEL` - exact model name shown by `ollama list`
- `OLLAMA_TIMEOUT_SECONDS` - request limit; default is 300 seconds to allow a cold model load
- `MAX_QUESTION_CHARS` - input limit; default is 2,000 characters

`.env` is excluded from Git.

## Run

Make sure Ollama is running and the model is available, then run:

```powershell
streamlit run app.py
```

Enter a problem or choose an example, then select **Solve problem**. **Clear** resets the input and result. To check whether the model is loaded and whether Ollama reports CPU/GPU use, run `ollama ps` after solving a problem.

## Sample problems

1. Mia has 18 apples, buys 7 more, and gives 5 away. How many remain? (20 apples)
2. 48 pencils are packed equally into 6 boxes. How many per box? (8 pencils)
3. A jacket costs $80 and is discounted 15%. What is the sale price? ($68)
4. A car travels at 60 km/h for 2.5 hours. How far? (150 km)
5. Three times a number plus 5 equals 26. What is the number? (7)

The expected values above are deterministic examples for local SymPy tests. The model generates its response at runtime.

## Tests and actual local model check

Run automated tests with:

```powershell
python -m pytest -q
```

The test suite checks the five sample calculations, mismatches and unsupported expressions, plus mocked Ollama success, server failure, timeout, missing-model, invalid-JSON, malformed-answer handling, and CUDA-runner CPU fallback. Mocked service tests do not replace an actual inference check.

Environment checks found Ollama 0.33.2 and the requested model in `ollama list` and `/api/tags`. Final integrated runs through `solve_problem` produced: addition/subtraction, 20 apples (verified); pens, Rs. 240 (verified after a local expression follow-up); percentage, 68 (verified); speed/time, 150 km (verified); algebra, 7 (the model supplied `21 + 5`, which did not match, so SymPy reported a mismatch). A later browser run solved 48 pencils divided among 6 boxes as 8 pencils, verified by SymPy. A CUDA runner initialization failure (`0xc0000409`) was handled by retrying the same local request on CPU. This records the model's actual responses; no expected answer is treated as verified unless SymPy confirms it. `ollama ps` reported 52% CPU / 48% GPU. `nvidia-smi` was unavailable, so separate NVIDIA telemetry could not be collected.

## Screenshots

See [`screenshots/README.md`](screenshots/README.md) for suggested screenshots from the running local application. No screenshots are fabricated.

## Limitations and future improvements

- SymPy checks one numeric expression and one numeric answer; it does not prove that the model selected the right quantities or units.
- Problems involving symbolic equations, multiple unknowns, or non-arithmetic formulas may be marked unavailable.
- First inference may take longer while the 7B model loads. Increase `OLLAMA_TIMEOUT_SECONDS` if the hardware needs more time.
- Future improvement: add validated structured output and a carefully scoped solver for supported one-variable algebra.

## Submission files

- Updated procedure report: `docs/procedure_report_ollama.pdf` (two A4 pages; the previous PDF is locked by another process)
- Clean archive: `Code.zip`
- GitHub URL placeholder: `github_link.txt`

## Prompt used

The exact system prompt is defined in [`prompts.py`](prompts.py) and included in the procedure report. It asks Qwen2.5-Math to tutor patiently, identify known and unknown values, explain formulas and calculations with units, admit ambiguity, avoid guessing, think step by step, and provide a safe arithmetic expression (or `NONE`) for SymPy. The final non-empty line must start with `Final answer:`.

## GitHub repository

Published repository: [https://github.com/Om-Upadhyay/llm-ps26-ADT23SOCB0662](https://github.com/Om-Upadhyay/llm-ps26-ADT23SOCB0662)
