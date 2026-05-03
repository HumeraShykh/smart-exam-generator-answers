import gc
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ExamGen AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_streamlit_secrets_to_environ() -> None:
    """Streamlit Community Cloud stores keys in st.secrets; rag_pipeline reads os.environ."""
    try:
        for key in st.secrets:
            val = st.secrets[key]
            if isinstance(val, dict):
                continue
            if isinstance(val, (str, int, float)) and str(val).strip() != "":
                os.environ[key] = str(val)
    except Exception:
        pass


_apply_streamlit_secrets_to_environ()

import rag_pipeline
from orchestrator import AGENT_ROLES, ExamOrchestrator
from pdf_export import (
    build_answer_key_pdf,
    build_questions_pdf,
    pdf_to_bytes,
    safe_filename_part,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DATA_DIR   = Path("app_data")
CHROMA_PARENT  = APP_DATA_DIR / "chroma_db"
UPLOADS_DIR    = APP_DATA_DIR / "uploads"
SYLLABUS_DIR   = UPLOADS_DIR  / "Lectures"
ASSESSMENT_DIR = UPLOADS_DIR  / "Assessment"
CLO_FILE       = UPLOADS_DIR  / "clo_plo.yaml"

# ---------------------------------------------------------------------------
# Global dark CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base dark background ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #0d0f1a !important;
    color: #e2e8f0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#111827 0%,#1a1f35 100%) !important;
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Header strip ── */
[data-testid="stHeader"] { background: #0d0f1a !important; }

/* ── All input widgets ── */
input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background: #1e2235 !important;
    color: #e2e8f0 !important;
    border: 1px solid #3d4a6b !important;
    border-radius: 8px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #1a1f35 !important;
    border: 2px dashed #4a5568 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #cbd5e1 !important; }
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: #94a3b8 !important; }

/* ── All general text ── */
p, span, label, div, li { color: #e2e8f0; }
[data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }
[data-testid="stText"] { color: #e2e8f0 !important; }
.stSlider label, .stTextInput label { color: #a5b4fc !important; }
section[data-testid="stFileUploaderDropzoneInstructions"] span { color: #94a3b8 !important; }

/* ── Buttons ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: all .2s !important;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,.4) !important;
}
[data-testid="baseButton-secondary"] {
    background: #1e2235 !important;
    border: 1px solid #3d4a6b !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #1a1f35;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 0.82em !important;
    font-weight: 600 !important;
    padding: 6px 12px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1a1f35 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #1a1f35;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #a5b4fc !important; font-weight: 800 !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg,#6366f1,#a78bfa,#ec4899) !important;
    border-radius: 99px !important;
}

/* ── Divider ── */
hr { border-color: #2d3748 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f1a; }
::-webkit-scrollbar-thumb { background: #3d4a6b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Design helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _reset_dirs():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for d in (SYLLABUS_DIR, ASSESSMENT_DIR):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)  # ignore locked files on Windows
        d.mkdir(parents=True, exist_ok=True)
    CLO_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        if CLO_FILE.exists():
            CLO_FILE.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Design helpers
# ---------------------------------------------------------------------------

AGENT_COLORS = {
    "A1": "#6366f1", "A2": "#8b5cf6", "A3": "#06b6d4",
    "A4": "#f59e0b", "A5": "#10b981", "A6": "#f43f5e",
}

BLOOM_COLORS = {
    "Remember": "#64748b", "Understand": "#3b82f6",
    "Apply": "#10b981",    "Analyze":    "#06b6d4",
    "Evaluate": "#f59e0b", "Create":     "#f43f5e",
}

DIFF_COLORS = {"Easy": "#10b981", "Medium": "#f59e0b", "Hard": "#f43f5e"}


def badge(text, color="#6366f1"):
    return (
        f'<span style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;padding:2px 10px;'
        f'border-radius:20px;font-size:0.73em;font-weight:700;'
        f'letter-spacing:.4px">{text}</span>'
    )


def question_card(i, q, show_clo=False):
    bloom = q.get("blooms_level", "")
    diff  = q.get("difficulty",   "")
    qtype = q.get("type",         "")
    clo   = q.get("clo",          "")
    plo   = q.get("plo",          "")

    tags = ""
    if bloom: tags += badge(bloom,  BLOOM_COLORS.get(bloom, "#6366f1")) + " "
    if diff:  tags += badge(diff,   DIFF_COLORS.get(diff,   "#94a3b8")) + " "
    if qtype: tags += badge(qtype,  "#94a3b8") + " "
    if show_clo and clo and clo != "UNMAPPED":
        tags += badge(clo, "#6366f1") + " "
    if show_clo and plo and plo != "UNMAPPED":
        tags += badge(plo, "#8b5cf6")

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#1a1f35 0%,#1e2340 100%);
        border:1px solid #2d3748;border-left:4px solid #6366f1;
        border-radius:12px;padding:14px 18px;margin-bottom:10px;
        box-shadow:0 2px 8px rgba(0,0,0,.3)">
      <div style="color:#4a5568;font-size:.76em;margin-bottom:4px;
                  font-weight:600;letter-spacing:.8px">QUESTION {i}</div>
      <div style="color:#e2e8f0;font-size:.95em;
                  line-height:1.6;margin-bottom:10px">{q.get('question','')}</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px">{tags}</div>
    </div>""", unsafe_allow_html=True)


def agent_banner(code, state="running"):
    meta  = AGENT_ROLES[code]
    color = AGENT_COLORS[code]
    dot   = (f'<span style="color:{color};animation:pulse 1s infinite">●</span>'
             if state == "running" else
             f'<span style="color:#10b981">✓</span>')
    return f"""
    <div style="background:linear-gradient(135deg,{color}18,{color}08);
                border:1px solid {color}44;border-left:4px solid {color};
                border-radius:12px;padding:12px 18px;margin:8px 0">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b style="color:{color};font-size:1em">[{code}] {meta['role']}</b>
        <span style="font-size:1.1em">{dot}</span>
      </div>
      <div style="color:#94a3b8;font-size:.8em;margin-top:3px">{meta['goal']}</div>
    </div>"""


def section_header(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:20px 0 12px">
      <h2 style="color:#e2e8f0;margin:0;font-size:1.3em;font-weight:700">{title}</h2>
      {f'<p style="color:#64748b;font-size:.85em;margin:2px 0 0">{subtitle}</p>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="
    background:linear-gradient(135deg,#1a1f35 0%,#1e2340 60%,#2d1b69 100%);
    border:1px solid #3d4a6b;border-radius:16px;
    padding:28px 32px;margin-bottom:24px;
    box-shadow:0 4px 24px rgba(99,102,241,.15)">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="font-size:2.4em">🎓</div>
    <div>
      <h1 style="margin:0;color:#e2e8f0;font-size:1.6em;font-weight:800;
                 background:linear-gradient(135deg,#a5b4fc,#c4b5fd);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        ExamGen AI
      </h1>
      <p style="margin:2px 0 0;color:#64748b;font-size:.88em">
        Multi-Agent Exam Paper Generator &nbsp;|&nbsp;
        RAG + 6 AI Agents &nbsp;|&nbsp; CSC505 · Sukkur IBA University
      </p>
    </div>
  </div>
  <div style="display:flex;gap:8px;margin-top:16px;flex-wrap:wrap">
    <span style="background:#6366f122;color:#a5b4fc;border:1px solid #6366f155;
                 padding:3px 12px;border-radius:20px;font-size:.76em;font-weight:600">
      A1 Generate
    </span>
    <span style="color:#64748b;padding:3px 0;font-size:.76em">→</span>
    <span style="background:#8b5cf622;color:#c4b5fd;border:1px solid #8b5cf655;
                 padding:3px 12px;border-radius:20px;font-size:.76em;font-weight:600">
      A2 Bloom
    </span>
    <span style="color:#64748b;padding:3px 0;font-size:.76em">→</span>
    <span style="background:#06b6d422;color:#67e8f9;border:1px solid #06b6d455;
                 padding:3px 12px;border-radius:20px;font-size:.76em;font-weight:600">
      A3 Balance
    </span>
    <span style="color:#64748b;padding:3px 0;font-size:.76em">→</span>
    <span style="background:#f59e0b22;color:#fcd34d;border:1px solid #f59e0b55;
                 padding:3px 12px;border-radius:20px;font-size:.76em;font-weight:600">
      A4 CLO Map
    </span>
    <span style="color:#64748b;padding:3px 0;font-size:.76em">→</span>
    <span style="background:#10b98122;color:#6ee7b7;border:1px solid #10b98155;
                 padding:3px 12px;border-radius:20px;font-size:.76em;font-weight:600">
      A5 Quality
    </span>
    <span style="color:#64748b;padding:3px 0;font-size:.76em">→</span>
    <span style="background:#f43f5e22;color:#fda4af;border:1px solid #f43f5e55;
                 padding:3px 12px;border-radius:20px;font-size:.76em;font-weight:600">
      A6 Answer Key
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 18px">
      <div style="font-size:2em">⚙️</div>
      <div style="color:#a5b4fc;font-weight:700;font-size:1em">Pipeline Settings</div>
    </div>""", unsafe_allow_html=True)

    topic = st.text_input("Exam Topic", value="Transformer architectures in NLP",
                          help="Main topic for exam question generation")
    initial_question_batch = st.slider(
        "Initial questions (A1)",
        5, 25, 12,
        help="How many questions the first agent asks the model to generate. "
        "Agent A4 can still add more to cover missing CLOs.",
    )
    max_iterations = st.slider("CLO Gap-Fill Iterations", 1, 5, 2,
                               help="How many times to loop and fill missing CLOs")
    st.caption("Upload files in the main area, then press **Run Full Pipeline** below the uploaders.")

    st.markdown("---")
    st.markdown('<div style="color:#64748b;font-size:.78em;font-weight:700;'
                'letter-spacing:.8px;margin-bottom:8px">AGENT PIPELINE</div>',
                unsafe_allow_html=True)
    for code, meta in AGENT_ROLES.items():
        c = AGENT_COLORS[code]
        st.markdown(
            f'<div style="background:{c}11;border-left:3px solid {c};'
            f'padding:5px 10px;margin:3px 0;border-radius:6px;font-size:.78em">'
            f'<b style="color:{c}">{code}</b> '
            f'<span style="color:#94a3b8">{meta["role"]}</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="color:#64748b;font-size:.76em;text-align:center">'
                'Cloud LLMs · ChromaDB · LangChain</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
section_header("Upload Materials", "Provide syllabus, past papers and CLO/PLO outcomes")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p style="color:#a5b4fc;font-size:.85em;font-weight:700;'
                'margin-bottom:6px;letter-spacing:.5px">📚 SYLLABUS / LECTURE FILES</p>',
                unsafe_allow_html=True)
    syllabus_files = st.file_uploader(
        "Syllabus files", type=["pdf", "txt", "md", "pptx", "ppt"],
        accept_multiple_files=True, label_visibility="collapsed",
        key="syllabus_uploader")

with col2:
    st.markdown('<p style="color:#a5b4fc;font-size:.85em;font-weight:700;'
                'margin-bottom:6px;letter-spacing:.5px">📝 PAST PAPERS / ASSIGNMENTS</p>',
                unsafe_allow_html=True)
    assessment_files = st.file_uploader(
        "Assessment files", type=["pdf", "txt", "md", "pptx", "ppt"],
        accept_multiple_files=True, label_visibility="collapsed",
        key="assessment_uploader")

st.markdown('<p style="color:#a5b4fc;font-size:.85em;font-weight:700;'
            'margin-bottom:6px;letter-spacing:.5px">🎯 CLO / PLO YAML</p>',
            unsafe_allow_html=True)
clo_file = st.file_uploader(
    "CLO/PLO YAML",
    type=["yaml", "yml"],
    label_visibility="collapsed",
    key="clo_uploader",
    help="If upload shows an error icon, try a smaller file or check your network; server limit is 500 MB.",
)

# Persist uploads: sidebar Run used to run before widgets resolved; also helps when
# the browser shows files but the latest rerun returns empty lists.
if syllabus_files:
    st.session_state["examgen_syllabus"] = syllabus_files
if assessment_files:
    st.session_state["examgen_assessment"] = assessment_files
if clo_file:
    st.session_state["examgen_clo"] = clo_file

syllabus_eff = syllabus_files or st.session_state.get("examgen_syllabus") or []
assessment_eff = assessment_files or st.session_state.get("examgen_assessment") or []
clo_eff = clo_file or st.session_state.get("examgen_clo")

c_clear, _ = st.columns([1, 3])
with c_clear:
    if st.button("Clear saved uploads", help="Reset cached files from this browser session"):
        for _k in ("examgen_syllabus", "examgen_assessment", "examgen_clo"):
            st.session_state.pop(_k, None)
        st.session_state.pop("examgen_last_result", None)
        st.rerun()

# Show upload status
if syllabus_eff or assessment_eff or clo_eff:
    cols = st.columns(3)
    cols[0].markdown(
        f'<div style="background:#6366f111;border:1px solid #6366f133;'
        f'border-radius:8px;padding:8px 12px;font-size:.82em;color:#a5b4fc">'
        f'📚 {len(syllabus_eff)} syllabus file(s)</div>',
        unsafe_allow_html=True)
    cols[1].markdown(
        f'<div style="background:#10b98111;border:1px solid #10b98133;'
        f'border-radius:8px;padding:8px 12px;font-size:.82em;color:#6ee7b7">'
        f'📝 {len(assessment_eff)} assessment file(s)</div>',
        unsafe_allow_html=True)
    cols[2].markdown(
        f'<div style="background:#f59e0b11;border:1px solid #f59e0b33;'
        f'border-radius:8px;padding:8px 12px;font-size:.82em;color:#fcd34d">'
        f'{"✅" if clo_eff else "❌"} CLO/PLO YAML</div>',
        unsafe_allow_html=True)

run_btn = st.button(
    "🚀  Run Full Pipeline",
    type="primary",
    use_container_width=True,
    key="run_pipeline_main",
)

# ---------------------------------------------------------------------------
# Pipeline run
# ---------------------------------------------------------------------------
if run_btn:
    if not syllabus_eff or not assessment_eff or not clo_eff:
        st.markdown("""
        <div style="background:#f43f5e18;border:1px solid #f43f5e44;
                    border-radius:10px;padding:12px 18px;color:#fda4af">
          ⚠️ Please upload syllabus files, assessment files, and CLO/PLO YAML before running.
        </div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown("---")
    section_header("Live Pipeline Execution", "Watch each agent work in real-time")

    progress_bar  = st.progress(0, text="Initializing...")
    agent_slot    = st.empty()

    st.markdown('<div style="color:#64748b;font-size:.78em;font-weight:700;'
                'letter-spacing:.8px;margin:16px 0 8px">AGENT OUTPUTS</div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["A1 · Questions", "A2 · Bloom",
                    "A3 · Difficulty", "A4 · CLO Map",
                    "A5 · Quality",   "A6 · Answer Key"])
    tab_ph = {
        "A1": tabs[0].empty(), "A2": tabs[1].empty(),
        "A3": tabs[2].empty(), "A4": tabs[3].empty(),
        "A5": tabs[4].empty(), "A6": tabs[5].empty(),
    }

    STEPS = 8
    step  = [0]

    def advance(text):
        step[0] += 1
        progress_bar.progress(min(int(step[0] / STEPS * 100), 99), text=text)

    def on_step(code, event, data):
        if event == "start":
            agent_slot.markdown(agent_banner(code, "running"),
                                unsafe_allow_html=True)
            return

        if event == "iteration":
            it   = data.get("iteration", "?")
            ok   = data.get("coverage_ok", False)
            qs   = data.get("questions", [])
            tab_ph["A4"].markdown(
                f'<div style="background:#f59e0b11;border:1px solid #f59e0b33;'
                f'border-radius:8px;padding:10px 14px;color:#fcd34d;font-size:.85em">'
                f'Iteration {it} · Coverage: {"Complete ✓" if ok else "Incomplete — gap-filling..."}'
                f' · {len(qs)} questions mapped</div>',
                unsafe_allow_html=True)
            return

        if event != "done":
            return

        advance(f"{AGENT_ROLES[code]['role']} done")
        agent_slot.markdown(agent_banner(code, "done"), unsafe_allow_html=True)
        questions  = data.get("questions", [])
        answer_key = data.get("answer_key", [])
        ph         = tab_ph.get(code)
        if not ph:
            return

        if code == "A1":
            with ph.container():
                st.markdown(
                    f'<div style="color:#64748b;font-size:.82em;'
                    f'margin-bottom:6px">{len(questions)} questions in the <b>initial</b> batch (A1)</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Later agents can change the count: A3 rebalances difficulty (may duplicate), "
                    "A4 adds gap-fill questions for CLOs not yet covered — check **Final summary** for the total."
                )
                for i, q in enumerate(questions, 1):
                    question_card(i, q)

        elif code == "A2":
            with ph.container():
                bloom_counts: dict = {}
                for q in questions:
                    b = q.get("blooms_level", "Unknown")
                    bloom_counts[b] = bloom_counts.get(b, 0) + 1
                cols = st.columns(len(bloom_counts) or 1)
                for idx, (level, cnt) in enumerate(bloom_counts.items()):
                    c = BLOOM_COLORS.get(level, "#6366f1")
                    cols[idx].markdown(
                        f'<div style="background:{c}18;border:1px solid {c}44;'
                        f'border-radius:8px;padding:10px;text-align:center">'
                        f'<div style="color:{c};font-size:1.4em;font-weight:800">{cnt}</div>'
                        f'<div style="color:#94a3b8;font-size:.75em">{level}</div></div>',
                        unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                for i, q in enumerate(questions, 1):
                    question_card(i, q)

        elif code == "A3":
            with ph.container():
                c1, c2, c3 = st.columns(3)
                for col, level in zip([c1, c2, c3], ["Easy", "Medium", "Hard"]):
                    cnt = sum(1 for q in questions if q.get("difficulty") == level)
                    c   = DIFF_COLORS[level]
                    col.markdown(
                        f'<div style="background:{c}18;border:1px solid {c}44;'
                        f'border-radius:8px;padding:12px;text-align:center">'
                        f'<div style="color:{c};font-size:1.6em;font-weight:800">{cnt}</div>'
                        f'<div style="color:#94a3b8;font-size:.78em">{level}</div></div>',
                        unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                for i, q in enumerate(questions, 1):
                    question_card(i, q)

        elif code == "A4":
            with ph.container():
                mapped   = [q for q in questions if q.get("clo") != "UNMAPPED"]
                unmapped = [q for q in questions if q.get("clo") == "UNMAPPED"]
                c1, c2   = st.columns(2)
                c1.markdown(
                    f'<div style="background:#10b98118;border:1px solid #10b98144;'
                    f'border-radius:8px;padding:12px;text-align:center">'
                    f'<div style="color:#10b981;font-size:1.6em;font-weight:800">{len(mapped)}</div>'
                    f'<div style="color:#94a3b8;font-size:.78em">Mapped</div></div>',
                    unsafe_allow_html=True)
                c2.markdown(
                    f'<div style="background:#f43f5e18;border:1px solid #f43f5e44;'
                    f'border-radius:8px;padding:12px;text-align:center">'
                    f'<div style="color:#f43f5e;font-size:1.6em;font-weight:800">{len(unmapped)}</div>'
                    f'<div style="color:#94a3b8;font-size:.78em">Unmapped</div></div>',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                for i, q in enumerate(questions, 1):
                    question_card(i, q, show_clo=True)

        elif code == "A5":
            with ph.container():
                st.markdown(
                    f'<div style="background:#10b98118;border:1px solid #10b98144;'
                    f'border-radius:8px;padding:10px 14px;color:#6ee7b7;font-size:.85em;'
                    f'margin-bottom:12px">'
                    f'✓ {len(questions)} high-quality, deduplicated questions</div>',
                    unsafe_allow_html=True)
                for i, q in enumerate(questions, 1):
                    question_card(i, q, show_clo=True)

        elif code == "A6":
            with ph.container():
                for j, entry in enumerate(answer_key, 1):
                    with st.expander(f"Q{j}: {str(entry.get('question',''))[:80]}..."):
                        st.markdown(
                            f'<div style="background:#1a1f35;border-radius:8px;'
                            f'padding:12px;margin-bottom:8px">'
                            f'<div style="color:#a5b4fc;font-size:.78em;font-weight:700;'
                            f'margin-bottom:4px">MODEL ANSWER</div>'
                            f'<div style="color:#e2e8f0;font-size:.9em">'
                            f'{entry.get("model_answer","")}</div></div>',
                            unsafe_allow_html=True)
                        scheme = entry.get("marking_scheme", [])
                        if scheme:
                            st.markdown(
                                '<div style="color:#6ee7b7;font-size:.78em;'
                                'font-weight:700;margin-bottom:4px">MARKING SCHEME</div>',
                                unsafe_allow_html=True)
                            for pt in scheme:
                                st.markdown(
                                    f'<div style="color:#94a3b8;font-size:.85em;'
                                    f'padding:2px 0 2px 12px">▸ {pt}</div>',
                                    unsafe_allow_html=True)
                        marks = entry.get("recommended_marks", "?")
                        st.markdown(
                            f'<div style="background:#f59e0b18;border:1px solid #f59e0b44;'
                            f'border-radius:6px;padding:6px 12px;display:inline-block;'
                            f'color:#fcd34d;font-size:.8em;margin-top:6px">'
                            f'Marks: {marks}</div>',
                            unsafe_allow_html=True)

    # ---- Run ----
    try:
        agent_slot.markdown(
            '<div style="background:#6366f118;border:1px solid #6366f144;'
            'border-radius:10px;padding:12px 18px;color:#a5b4fc">Preparing files...</div>',
            unsafe_allow_html=True)

        _reset_dirs()
        for f in syllabus_eff:
            (SYLLABUS_DIR / f.name).write_bytes(f.getbuffer())
        for f in assessment_eff:
            (ASSESSMENT_DIR / f.name).write_bytes(f.getbuffer())
        CLO_FILE.write_bytes(clo_eff.getbuffer())

        advance("Building RAG pipeline...")
        agent_slot.markdown(
            '<div style="background:#06b6d418;border:1px solid #06b6d444;'
            'border-radius:10px;padding:12px 18px;color:#67e8f9">'
            'Building RAG pipeline — embedding documents...</div>',
            unsafe_allow_html=True)

        gc.collect()
        old = st.session_state.get("last_chroma_path")
        if old:
            shutil.rmtree(old, ignore_errors=True)
        CHROMA_PARENT.mkdir(parents=True, exist_ok=True)
        run_chroma = CHROMA_PARENT / uuid.uuid4().hex
        run_chroma.mkdir(parents=True, exist_ok=True)
        st.session_state["last_chroma_path"] = str(run_chroma)
        rag_pipeline.CHROMA_DB_PATH = str(run_chroma)

        pipeline = rag_pipeline.build_rag_pipeline(
            syllabus_path=str(SYLLABUS_DIR),
            clo_plo_path=str(CLO_FILE),
            past_papers_path=str(ASSESSMENT_DIR),
        )
        advance("RAG pipeline ready. Launching agents...")

        orchestrator = ExamOrchestrator(pipeline)
        result = orchestrator.run_full(
            topic=topic,
            max_iterations=max_iterations,
            initial_questions=initial_question_batch,
            on_step=on_step,
        )
        st.session_state["examgen_last_result"] = {
            "questions": result["questions"],
            "answer_key": result["answer_key"],
            "topic": topic,
        }

        progress_bar.progress(100, text="All agents complete!")
        agent_slot.markdown("""
        <div style="background:linear-gradient(135deg,#10b98122,#6366f122);
                    border:1px solid #10b98155;border-radius:12px;
                    padding:14px 20px;color:#6ee7b7;font-weight:700;font-size:1em">
          ✅ Pipeline complete — all 6 agents finished successfully!
        </div>""", unsafe_allow_html=True)

    except Exception as exc:
        progress_bar.empty()
        st.markdown(
            f'<div style="background:#f43f5e18;border:1px solid #f43f5e55;'
            f'border-radius:10px;padding:14px 18px;color:#fda4af">'
            f'<b>Pipeline failed:</b> {exc}</div>',
            unsafe_allow_html=True)
        st.stop()

# ---------------------------------------------------------------------------
# Latest pipeline results (persists across reruns; PDF + JSON export)
# ---------------------------------------------------------------------------
_last = st.session_state.get("examgen_last_result")
if _last:
    questions = _last["questions"]
    answer_key = _last["answer_key"]
    topic_saved = _last.get("topic") or ""

    st.markdown("---")
    section_header("Final summary", "Counts from your last successful pipeline run")
    st.caption(
        "The number can exceed A1’s initial batch: **A4** may append questions for uncovered CLOs, "
        "and **A3** may duplicate some items to hit the Easy / Medium / Hard mix."
    )

    s1, s2, s3 = st.columns(3)
    q_total = len(questions)
    ak_total = len(answer_key)
    mapped = sum(1 for q in questions if q.get("clo") != "UNMAPPED")

    for col, val, label, color in [
        (s1, q_total, "Questions generated", "#6366f1"),
        (s2, ak_total, "Answer key entries", "#10b981"),
        (s3, mapped, "CLO mapped", "#f59e0b"),
    ]:
        col.markdown(
            f'<div style="background:{color}18;border:1px solid {color}44;'
            f'border-radius:12px;padding:16px;text-align:center">'
            f'<div style="color:{color};font-size:2em;font-weight:800">{val}</div>'
            f'<div style="color:#94a3b8;font-size:.8em;margin-top:4px">{label}</div></div>',
            unsafe_allow_html=True,
        )

    section_header("Export", "Download printable PDFs or raw JSON for your report")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    slug = safe_filename_part(topic_saved)

    try:
        pdf_q = pdf_to_bytes(build_questions_pdf(questions, topic=topic_saved))
        pdf_a = pdf_to_bytes(build_answer_key_pdf(answer_key, topic=topic_saved))
    except Exception as pdf_exc:
        pdf_q = pdf_a = None
        st.markdown(
            f'<div style="background:#f59e0b18;border:1px solid #f59e0b44;'
            f'border-radius:10px;padding:12px 18px;color:#fcd34d;font-size:.9em">'
            f"<b>PDF export unavailable:</b> {pdf_exc}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div style="margin:12px 0 8px;color:#64748b;font-size:.78em;font-weight:700;
                letter-spacing:.6px">PDF DOCUMENTS</div>
    """,
        unsafe_allow_html=True,
    )

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(
            """
        <div style="background:linear-gradient(145deg,#1a1f35,#1e2746);
                    border:1px solid #6366f155;border-radius:14px;padding:18px 20px;
                    min-height:120px;border-left:4px solid #6366f1">
          <div style="font-size:1.6em;margin-bottom:6px">📄</div>
          <div style="color:#e2e8f0;font-weight:700;font-size:1.05em;margin-bottom:6px">
            Question paper</div>
          <div style="color:#94a3b8;font-size:.82em;line-height:1.45">
            All questions with Bloom, difficulty, type & CLO/PLO tags — ready to print or share.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if pdf_q is not None:
            st.download_button(
                "⬇  Download questions (PDF)",
                data=pdf_q,
                file_name=f"exam_questions_{slug}_{stamp}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="dl_pdf_questions",
            )

    with pc2:
        st.markdown(
            """
        <div style="background:linear-gradient(145deg,#1a1f35,#1e2746);
                    border:1px solid #10b98155;border-radius:14px;padding:18px 20px;
                    min-height:120px;border-left:4px solid #10b981">
          <div style="font-size:1.6em;margin-bottom:6px">🔑</div>
          <div style="color:#e2e8f0;font-weight:700;font-size:1.05em;margin-bottom:6px">
            Answer key</div>
          <div style="color:#94a3b8;font-size:.82em;line-height:1.45">
            Model answers, marking scheme & recommended marks — separate PDF for instructors.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if pdf_a is not None:
            st.download_button(
                "⬇  Download answer key (PDF)",
                data=pdf_a,
                file_name=f"exam_answer_key_{slug}_{stamp}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="dl_pdf_answers",
            )

    st.markdown(
        """
    <div style="margin:18px 0 8px;color:#64748b;font-size:.78em;font-weight:700;
                letter-spacing:.6px">JSON DATA</div>
    """,
        unsafe_allow_html=True,
    )

    j1, j2 = st.columns(2)
    with j1:
        st.download_button(
            "⬇  Questions (JSON)",
            data=json.dumps(questions, indent=2),
            file_name=f"questions_{slug}_{stamp}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_questions",
        )
    with j2:
        st.download_button(
            "⬇  Answer key (JSON)",
            data=json.dumps(answer_key, indent=2),
            file_name=f"answer_key_{slug}_{stamp}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_answers",
        )
