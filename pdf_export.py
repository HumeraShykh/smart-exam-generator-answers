"""
Generate printable PDF exports for exam questions and answer keys (fpdf2).
Sukkur IBA–style headers (IBA logo + formal titles).
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fpdf import FPDF

_BRAND_DIR = Path(__file__).resolve().parent / "assets" / "brand"
# Prefer bundled official artwork (PNG or JPG).
_IBA_FILES = (
    _BRAND_DIR / "iba_logo.png",
    _BRAND_DIR / "iba_logo.jpg",
)

_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
)

_TIMES_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
    Path("/Library/Fonts/Times New Roman.ttf"),
)


def _register_font(pdf: FPDF) -> tuple[str, bool]:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            try:
                path = str(p)
                pdf.add_font("ExamFont", "", path)
                try:
                    pdf.add_font("ExamFont", "B", path)
                    pdf.add_font("ExamFont", "I", path)
                except Exception:
                    pass
                return "ExamFont", True
            except Exception:
                continue
    return "Helvetica", False


def _register_times(pdf: FPDF) -> str:
    for p in _TIMES_CANDIDATES:
        if p.is_file():
            try:
                path = str(p)
                pdf.add_font("ExamSerif", "", path)
                pdf.add_font("ExamSerif", "B", path)
                pdf.add_font("ExamSerif", "I", path)
                pdf.add_font("ExamSerif", "BI", path)
                return "ExamSerif"
            except Exception:
                continue
    return "Times"


def _txt(unicode_font: bool, text: str) -> str:
    if not text:
        return ""
    if unicode_font:
        return text
    return text.encode("latin-1", "replace").decode("latin-1")


def _iba_logo_png_bytes() -> bytes:
    """Fallback wordmark when assets/brand/iba_logo.png is missing."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 200, 90
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    purple_dark = (76, 29, 149)
    purple_mid = (109, 40, 217)

    def load_font(path: str | None, size: int) -> Any:
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
        return ImageFont.load_default()

    tnr = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    font_s = load_font(tnr, 12)
    font_l = load_font(tnr, 30)
    font_m = load_font(tnr, 12)

    draw.text((w // 2, 12), "Sukkur", fill=purple_dark, anchor="mm", font=font_s)
    draw.text((w // 2, 42), "IBA", fill=purple_mid, anchor="mm", font=font_l)
    draw.text((w // 2, 72), "University", fill=purple_dark, anchor="mm", font=font_m)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _iba_logo_raw_bytes() -> bytes:
    for path in _IBA_FILES:
        if path.is_file():
            return path.read_bytes()
    return _iba_logo_png_bytes()


def _iba_logo_dimensions_mm(
    raw: bytes, *, max_w_mm: float = 44.0, max_h_mm: float = 44.0
) -> tuple[float, float]:
    """Preserve aspect ratio (square/cropped thumbnails scale cleanly)."""
    from PIL import Image

    im = Image.open(BytesIO(raw))
    w_px, h_px = im.size
    aspect = w_px / max(h_px, 1)
    w_mm = max_w_mm
    h_mm = w_mm / aspect
    if h_mm > max_h_mm:
        h_mm = max_h_mm
        w_mm = h_mm * aspect
    return float(w_mm), float(h_mm)


def _count_by_type(questions: List[Dict[str, Any]]) -> Tuple[int, int]:
    mcq = sum(
        1 for q in questions if "MCQ" in str(q.get("type", "")).strip().upper()
    )
    return mcq, len(questions) - mcq


def _write_iba_formal_header(
    pdf: FPDF,
    *,
    serif: str,
    title_line: str,
    subtitle_paren: str,
    topic: str,
    stamp: str,
    section_source: List[Dict[str, Any]],
    show_section_table: bool,
    block_heading: str,
    instruction_line: str,
) -> None:
    """
    Centered letterhead: IBA logo centered on top; all header lines centered below.
    Bundled logo: assets/brand/iba_logo.png or iba_logo.jpg (see README).
    """
    margin = pdf.l_margin
    y0 = pdf.get_y()

    raw_logo = _iba_logo_raw_bytes()
    logo_w, logo_h = _iba_logo_dimensions_mm(raw_logo)

    x_logo = (pdf.w - logo_w) / 2
    pdf.image(BytesIO(raw_logo), x=x_logo, y=y0, w=logo_w, h=logo_h)

    pdf.set_y(y0 + logo_h + 5)
    pdf.set_x(margin)

    pdf.set_font(serif, style="B", size=13)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(
        0, 7, txt="Sukkur IBA University", align="C", new_x="LMARGIN", new_y="NEXT"
    )

    pdf.set_font(serif, style="I", size=9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(
        0, 4, txt="Merit-Quality-Excellence", align="C", new_x="LMARGIN", new_y="NEXT"
    )

    pdf.set_font(serif, style="B", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 7, txt=title_line, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(serif, style="B", size=10)
    pdf.multi_cell(
        0, 6, txt=subtitle_paren, align="C", new_x="LMARGIN", new_y="NEXT"
    )

    pdf.set_font(serif, size=8)
    pdf.set_text_color(90, 90, 90)
    meta = f"Course topic: {topic or '—'}  ·  Generated: {stamp}"
    pdf.multi_cell(0, 4, txt=meta, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    y_after = pdf.get_y() + 4
    pdf.set_y(y_after)
    pdf.line(margin, y_after - 1, pdf.w - margin, y_after - 1)

    if show_section_table and section_source:
        mcq, part2 = _count_by_type(section_source)
        pdf.ln(2)
        pdf.set_font(serif, style="B", size=10)
        pdf.multi_cell(
            0,
            6,
            txt=f"PART-I (MCQ / OBJECTIVE)  ·  {mcq} QUESTIONS",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.multi_cell(
            0,
            6,
            txt=f"PART-II (SHORT & LONG)  ·  {part2} QUESTIONS",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(3)
        pdf.line(margin, pdf.get_y(), pdf.w - margin, pdf.get_y())
        pdf.ln(4)

    pdf.set_font(serif, style="B", size=11)
    pdf.multi_cell(0, 6, txt=block_heading, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(serif, style="B", size=9)
    pdf.multi_cell(
        0,
        5,
        txt=instruction_line,
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)


def build_questions_pdf(
    questions: List[Dict[str, Any]],
    *,
    topic: str = "",
    institution: str = "CSC505 · Sukkur IBA University",
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    font, unicode_font = _register_font(pdf)
    serif = _register_times(pdf)
    pdf.add_page()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    _write_iba_formal_header(
        pdf,
        serif=serif,
        title_line="EXAM QUESTION PAPER",
        subtitle_paren="(Generated assessment — ExamGen AI)",
        topic=f"{topic}  ·  {institution}" if topic else institution,
        stamp=stamp,
        section_source=questions,
        show_section_table=True,
        block_heading="QUESTIONS",
        instruction_line=(
            "Read each question carefully and answer in the required format."
        ),
    )

    if not questions:
        pdf.set_font(font, size=11)
        pdf.multi_cell(
            0,
            8,
            txt=_txt(unicode_font, "No questions available."),
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
            0,
            8,
            txt=_txt(unicode_font, f"Question {i}"),
            new_x="LMARGIN",
            new_y="NEXT",
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
    institution: str = "CSC505 · Sukkur IBA University",
    questions: List[Dict[str, Any]] | None = None,
) -> bytes:
    """
    Same letterhead as the question paper. Pass `questions` (final pipeline list)
    so PART-I / PART-II counts match the student paper.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    font, unicode_font = _register_font(pdf)
    serif = _register_times(pdf)
    pdf.add_page()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    section_src = questions if questions is not None else []
    show_table = bool(section_src)

    _write_iba_formal_header(
        pdf,
        serif=serif,
        title_line="ANSWER SHEET",
        subtitle_paren="(Faculty / examiner copy — ExamGen AI)",
        topic=f"{topic}  ·  {institution}" if topic else institution,
        stamp=stamp,
        section_source=section_src,
        show_section_table=show_table,
        block_heading="MARKING KEY & MODEL RESPONSES",
        instruction_line=(
            "Confidential. Contains suggested answers and marking criteria "
            "aligned with the question paper above."
        ),
    )

    if not answer_key:
        pdf.set_font(font, size=11)
        pdf.multi_cell(
            0,
            8,
            txt=_txt(unicode_font, "No answer key entries."),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        return pdf.output()

    for j, entry in enumerate(answer_key, start=1):
        qtext = (entry.get("question") or "").strip()
        ans = (entry.get("model_answer") or "").strip()
        marks = entry.get("recommended_marks", "?")
        scheme = entry.get("marking_scheme") or []

        pdf.set_font(font, style="B", size=12)
        pdf.multi_cell(
            0,
            8,
            txt=_txt(unicode_font, f"Question {j}"),
            new_x="LMARGIN",
            new_y="NEXT",
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
            0,
            7,
            txt=_txt(unicode_font, f"Recommended marks: {marks}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)

    return pdf.output()


def pdf_to_bytes(pdf_output) -> bytes:
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
