"""Generate the fixed two-page A4 procedure report."""
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prompts import SYSTEM_PROMPT
OUT = ROOT / "docs" / "procedure_report_ollama.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=5))
styles.add(ParagraphStyle(name="Body12", parent=styles["BodyText"], fontName="Helvetica", fontSize=12, leading=15, spaceAfter=5))
styles.add(ParagraphStyle(name="HeadCustom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#176B87"), spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle(name="Prompt", parent=styles["BodyText"], fontName="Helvetica", fontSize=12, leading=14.2, leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#C9DCE8"), borderWidth=0.5, borderPadding=7, backColor=colors.HexColor("#F4F8FA"), spaceAfter=5))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5E0E7"))
    canvas.line(20*mm, 15*mm, 190*mm, 15*mm)
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#536575"))
    canvas.drawString(20*mm, 10*mm, "Math Word Problem Solver | PS 26")
    canvas.drawRightString(190*mm, 10*mm, f"Page {doc.page} of 2")
    canvas.restoreState()

def p(text, style="Body12"):
    return Paragraph(text, styles[style])

story = [
    p("Math Word Problem Solver", "TitleCustom"),
    p("Procedure Report | GenAI (LLM) Mini Project", "Body12"),
]
metadata = Table([
    [p("<b>Problem Statement:</b> 26", "Body12"), p("<b>Student:</b> Om Upadhyay", "Body12")],
    [p("<b>Roll Number:</b> 26", "Body12"), p("<b>Class:</b> Not provided", "Body12")],
], colWidths=[82*mm, 88*mm])
metadata.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1),colors.HexColor("#EDF4F7")),("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#C9DCE8")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),2)]))
story += [metadata, p("Problem statement", "HeadCustom"),
          p("Students struggle with mathematical word problems. This project builds a tool that uses an LLM to solve a submitted word problem step by step and explain the calculations to the student."),
          p("Tools and model", "HeadCustom"),
          p("Python, Streamlit, the local Ollama HTTP API, Qwen2.5-Math 7B (`mightykatun/qwen2.5-math:7b`), SymPy, and python-dotenv. The local tests use pytest."),
          p("Implementation procedure", "HeadCustom")]
for item in [
    "1. Set up a modular Python project with a Streamlit entry point, an Ollama solver, reusable instructions, and a separate arithmetic verifier.",
    "2. Add a text area, five sample problems, solve and clear controls, and validation that rejects empty input and limits length to 2,000 characters.",
    "3. Load the local Ollama base URL and model name from environment variables; send the prompt to Ollama's local generate API.",
    "4. Check that the response contains a labeled final answer. Display the explanation and final answer separately in the interface.",
    "5. Restrict verification to short arithmetic expressions containing numbers, parentheses, and basic operators. Use SymPy and report verified, mismatch, or unavailable.",
    "6. Add automated tests for five sample arithmetic cases, mismatches, malformed or unsupported expressions, and ambiguous answers. If Ollama's CUDA runner fails during initialization, retry with `num_gpu=0` on the local CPU. Prepare setup documentation and a secret-excluding ZIP."
]:
    story.append(p(item))
story += [p("Prompt improvement made", "HeadCustom"),
          p("The initial requirements asked for a worked solution and a final answer. The prompt was made more reliable by requiring named sections and a separate restricted arithmetic expression (or NONE), so the verifier can check arithmetic without executing generated code."),
          PageBreak(), p("Final system prompt", "HeadCustom")]
prompt = escape(SYSTEM_PROMPT).replace("\n\n", "<br/><br/>").replace("\n", " ")
story += [p(prompt, "Prompt"),
          p("Results", "HeadCustom"),
          p("Automated suite: 29 passed. Actual local model checks: addition/subtraction 20 apples (verified); pens Rs. 240 (verified after a local expression follow-up); 15% discount 68 (verified); 60 km/h for 2.5 hours 150 km (verified); algebra answer 7, but the model expression 21 + 5 did not match, so SymPy reported a mismatch. Ollama reported 52% CPU / 48% GPU during an earlier inference. A browser run after resolving a CUDA runner initialization failure returned 8 pencils for 48 divided among 6 boxes, verified by SymPy.")]
story += [p("Challenges addressed", "HeadCustom"),
          p("A model-generated expression must not be treated as executable code. The verifier therefore accepts a narrow arithmetic character set and uses SymPy instead of Python eval. If Qwen omits an expression, the app asks the local model for a constrained expression. A CUDA shared-object initialization failure caused Ollama's GPU runner to terminate; the app now retries that specific failure on the CPU with `num_gpu=0`. The first model load takes time, so the app uses a configurable request timeout to accommodate cold starts."),
          p("Conclusion and future improvement", "HeadCustom"),
          p("The project implements an LLM math tutor with a limited arithmetic check. Next, add structured output and a scoped solver for supported algebra problems.")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=15*mm, bottomMargin=21*mm, title="Math Word Problem Solver - Procedure Report", author="Om Upadhyay")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)



