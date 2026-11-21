"""Streamlit interface for the local Math Word Problem Solver."""
import streamlit as st
from config import MAX_QUESTION_CHARS, PROJECT_TITLE
from math_verifier import verify_answer
from solver import SolverError, solve_problem

st.set_page_config(page_title=PROJECT_TITLE, page_icon="➗", layout="centered")
st.title(PROJECT_TITLE)
st.caption("A patient local math tutor that explains word problems and checks supported arithmetic with SymPy.")

EXAMPLES = {
    "Addition and subtraction": "Mia has 18 apples. She buys 7 more and gives 5 to a friend. How many apples remain?",
    "Multiplication and division": "A teacher packs 48 pencils equally into 6 boxes. How many pencils are in each box?",
    "Percentage": "A jacket costs $80 and is discounted by 15%. What is the sale price?",
    "Speed, distance, time": "A car travels at 60 km/h for 2.5 hours. How far does it travel?",
    "Basic algebra": "Three times a number plus 5 equals 26. What is the number?",
}

if "question" not in st.session_state:
    st.session_state.question = ""
with st.sidebar:
    st.subheader("Try an example")
    for label, example in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.question = example
question = st.text_area(
    "Mathematical word problem", key="question", height=160,
    max_chars=MAX_QUESTION_CHARS,
    placeholder="For example: A shop sells 3 notebooks for $4 each. How much do they cost altogether?",
)
col1, col2 = st.columns(2)
with col1:
    solve_clicked = st.button("Solve problem", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.question = ""
        st.session_state.pop("solution", None)
        st.rerun()

if solve_clicked:
    if not question.strip():
        st.error("Enter a word problem before solving.")
    elif len(question) > MAX_QUESTION_CHARS:
        st.error(f"Please keep the problem under {MAX_QUESTION_CHARS} characters.")
    else:
        with st.spinner("Asking the local Qwen2.5-Math model..."):
            try:
                st.session_state.solution = solve_problem(question)
            except SolverError as exc:
                st.error(str(exc))

if st.session_state.get("solution"):
    result = st.session_state.solution
    st.subheader("Solution")
    st.markdown(result["solution"])
    if result.get("execution_mode") == "cpu_fallback":
        st.info("Ollama's GPU runner failed to initialize, so this answer was generated with the local CPU fallback.")
    st.subheader("Final answer")
    st.success(result["final_answer"])
    check = verify_answer(result.get("verification_expression", ""), result["final_answer"])
    st.subheader("Arithmetic verification")
    if check.status == "verified":
        st.success(f"Verified with SymPy: {check.details}")
    elif check.status == "failed":
        st.error(f"SymPy found a mismatch: {check.details}")
    else:
        st.info(f"Could not verify: {check.details}")

