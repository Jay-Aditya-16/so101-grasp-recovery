"""Render the project report Markdown and referenced figures to a standalone PDF.

Install reportlab separately, then run from the repository root:
    python scripts/render_project_report.py --output reports/project-report.pdf
"""
import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    """Escape markup and link plain HTTP URLs."""
    text = escape(text)
    return re.sub(r"https?://[^\s<]+", lambda m: f'<link href="{m.group(0)}" color="#17609a">{m.group(0)}</link>', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if not font_path.exists() or not bold_path.exists():
        raise SystemExit("Arial regular and bold fonts are required to render this report on macOS")
    pdfmetrics.registerFont(TTFont("ReportArial", str(font_path)))
    pdfmetrics.registerFont(TTFont("ReportArialBold", str(bold_path)))
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontName="ReportArial", fontSize=9.2, leading=13.4, spaceAfter=7)
    title = ParagraphStyle("ReportTitle", parent=body, fontName="ReportArialBold", fontSize=19, leading=23, spaceAfter=12, textColor=colors.HexColor("#173751"))
    heading = ParagraphStyle("ReportHeading", parent=body, fontName="ReportArialBold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#173751"), keepWithNext=True)
    bullet = ParagraphStyle("ReportBullet", parent=body, leftIndent=12, firstLineIndent=-8, spaceAfter=4)
    cell = ParagraphStyle("ReportCell", parent=body, fontSize=7.8, leading=10.5, spaceAfter=0)
    cell_header = ParagraphStyle("ReportCellHeader", parent=cell, fontName="ReportArialBold", textColor=colors.white)
    caption = ParagraphStyle("ReportCaption", parent=body, fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#505c65"))
    doc = SimpleDocTemplate(str(args.output), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=19*mm, bottomMargin=18*mm,
                            title="SO-101 Vision-Guided Failure Detection and Recovery", author="SO-101 Recovery Project")
    lines = (ROOT / "docs/project-report.md").read_text().splitlines()
    story = []
    paragraph = []

    def flush():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), body))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush()
        elif line.startswith("# "):
            flush(); story.append(Paragraph(inline(line[2:]), title))
        elif line.startswith("## "):
            flush(); story.append(Paragraph(inline(line[3:]), heading))
        elif line.startswith("- "):
            flush(); story.append(Paragraph("• " + inline(line[2:]), bullet))
        elif line.startswith("[figure:") and line.endswith("]"):
            flush()
            path = ROOT / line[len("[figure:"):-1]
            graphic = Image(str(path))
            graphic._restrictSize(doc.width, 100*mm)
            story.append(Spacer(1, 4)); story.append(graphic)
            story.append(Paragraph("MATLAB analysis of recorded MuJoCo simulator traces.", caption))
        elif line.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                items = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", x) for x in items):
                    rows.append(items)
                i += 1
            widths = [doc.width * x for x in (0.19, 0.16, 0.25, 0.40)]
            table_data = [[Paragraph(inline(value), cell_header if row == 0 else cell) for value in values] for row, values in enumerate(rows)]
            table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#245b78")),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d0d5")),
                                       ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f8")]),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.append(table); story.append(Spacer(1, 7)); continue
        else:
            paragraph.append(line)
        i += 1
    flush()

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("ReportArial", 8)
        canvas.setFillColor(colors.HexColor("#64727b"))
        canvas.drawString(document.leftMargin, 10*mm, "SO-101 recovery | Simulation research report")
        canvas.drawRightString(A4[0] - document.rightMargin, 10*mm, str(document.page))
        canvas.restoreState()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
