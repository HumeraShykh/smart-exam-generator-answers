"""
Generate printable PDF exports for exam questions and answer keys (fpdf2).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fpdf import FPDF

# Prefer a Unicode-capable font on macOS so syllabus text survives; fallback to core fonts.
_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
)


def _register_font(pdf: FPDF) -> tuple[str, bool]:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            try:
                path = str(p)
                pdf.add_font("ExamFont", "", path)
                # Same file for synthetic bold/italic so set_font(..., style="B") works.
                try:
                    pdf.add_font("ExamFont", "B", path)
                    pdf.add_font("ExamFont", "I", path)
                except Exception:
                    pass
                return "ExamFont", True
            except Exception:
                continue
    return "Helvetica", False


def _txt(unicode_font: bool, text: str) -> str:
    """Ensure text is PDF-safe for the active font."""
    if not text:
        return ""
    if unicode_font:
        return text
    return text.encode("latin-1", "replace").decode("latin-1")


def _write_heading(
    pdf: FPDF, font: str, unicode_font: bool, title: str, subtitle: str = ""
) -> None:
    pdf.set_font(font, size=18)
    pdf.multi_cell(0, 10, txt=_txt(unicode_font, title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    if subtitle:
        pdf.set_font(font, size=10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, txt=_txt(unicode_font, subtitle), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    pdf.set_font(font, size=11)
    pdf.ln(4)


def build_questions_pdf(
    questions: List[Dict[str, Any]],
    *,
    topic: str = "",
    institution: str = "ExamGen AI — CSC505 · Sukkur IBA",
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    font, unicode_font = _register_font(pdf)
    pdf.add_page()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    _write_heading(
        pdf,
        font,
        unicode_font,
        "Exam Question Paper",
        f"{institution}\nTopic: {topic or '—'}\nGenerated: {stamp}",
    )
    if not questions:
        pdf.set_font(font, size=11)
        pdf.multi_cell(
            0, 8, txt=_txt(unicode_font, "No questions available."),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        return pdf.output()

    for i, q in enumerate(questions, start=1):
        bloom = q.get("blooms_level") or ""
        diff = q.get("difficulty") or ""
        qtype = q.get("type") or ""
        clo = q.get("clo") or ""
        plo = q.get("plo") or ""
        meta_parts = [p for p in (bloom, diff, qtype) if p]
        if clo and clo != "UNMAPPED":
            meta_parts.append(f"CLO: {clo}")
        if plo and plo != "UNMAPPED":
            meta_parts.append(f"PLO: {plo}")
        meta = " · ".join(meta_parts)

        pdf.set_font(font, style="B", size=12)
        pdf.multi_cell(
            0, 8, txt=_txt(unicode_font, f"Question {i}"),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_font(font, size=11)
        if meta:
            pdf.set_text_color(60, 60, 60)
            pdf.set_font(font, size=9)
            pdf.multi_cell(0, 5, txt=_txt(unicode_font, meta), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font(font, size=11)
        body = (q.get("question") or "").strip()
        pdf.multi_cell(0, 6, txt=_txt(unicode_font, body), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    return pdf.output()


def build_answer_key_pdf(
    answer_key: List[Dict[str, Any]],
    *,
    topic: str = "",
    institution: str = "ExamGen AI — CSC505 · Sukkur IBA",
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    font, unicode_font = _register_font(pdf)
    pdf.add_page()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    _write_heading(
        pdf,
        font,
        unicode_font,
        "Answer Key & Marking Guide",
        f"{institution}\nTopic: {topic or '—'}\nGenerated: {stamp}",
    )
    if not answer_key:
        pdf.set_font(font, size=11)
        pdf.multi_cell(
            0, 8, txt=_txt(unicode_font, "No answer key entries."),
            new_x="LMARGIN", new_y="NEXT",
        )
        return pdf.output()

    for j, entry in enumerate(answer_key, start=1):
        qtext = (entry.get("question") or "").strip()
        ans = (entry.get("model_answer") or "").strip()
        marks = entry.get("recommended_marks", "?")
        scheme = entry.get("marking_scheme") or []

        pdf.set_font(font, style="B", size=12)
        pdf.multi_cell(
            0, 8, txt=_txt(unicode_font, f"Question {j}"),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_font(font, size=10)
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 5, txt=_txt(unicode_font, qtext), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        pdf.set_font(font, style="B", size=10)
        pdf.cell(0, 6, txt="Model answer", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font, size=11)
        pdf.multi_cell(0, 6, txt=_txt(unicode_font, ans), new_x="LMARGIN", new_y="NEXT")

        if scheme:
            pdf.ln(1)
            pdf.set_font(font, style="B", size=10)
            pdf.cell(0, 6, txt="Marking scheme", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font, size=10)
            for pt in scheme:
                line = f"• {pt}"
                pdf.multi_cell(0, 5, txt=_txt(unicode_font, line), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font(font, style="B", size=10)
        pdf.set_text_color(180, 90, 20)
        pdf.cell(
            0, 7, txt=_txt(unicode_font, f"Recommended marks: {marks}"),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)

    return pdf.output()


def pdf_to_bytes(pdf_output) -> bytes:
    """fpdf2 output() returns bytes or bytearray depending on version."""
    if isinstance(pdf_output, bytes):
        return pdf_output
    return bytes(pdf_output)


def safe_filename_part(text: str, max_len: int = 40) -> str:
    keep = []
    for ch in (text or "").strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    s = "".join(keep).strip("_") or "exam"
    return s[:max_len]
