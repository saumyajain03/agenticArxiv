import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)


class NumberedCanvas(canvas.Canvas):
    """Canvas implementation that dynamically computes total page count and adds footers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Don't draw headers/footers on the cover or page 1 if it's a cover page
        # In our case, we will draw it on all pages for consistency
        
        # Footer text
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_text) # Letter width is 612, 0.75 in margin = 54 pt
        
        self.drawString(54, 36, "arXiv Paper Curator — Agentic RAG Report")
        
        # Footer line divider
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 48, 612 - 54, 48)
        
        # Header text
        self.drawString(54, 792 - 36, "Literature Synthesis & Literature Review Report")
        self.drawRightString(612 - 54, 792 - 36, datetime.now().strftime("%Y-%m-%d"))
        
        # Header line divider
        self.line(54, 792 - 42, 612 - 54, 792 - 42)
        
        self.restoreState()


class MarkdownPDFGenerator:
    """Helper service to compile scientific Markdown reports into structured PDF reports."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define customized styles for the academic report look and feel."""
        # Modify existing or add custom styles
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1e293b"), # Slate 800
            spaceAfter=15,
            alignment=0 # Left
        ))
        
        self.styles.add(ParagraphStyle(
            name="ReportSubtitle",
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#475569"), # Slate 600
            spaceAfter=20,
            alignment=0
        ))

        self.styles.add(ParagraphStyle(
            name="AcademicH1",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#0f172a"), # Slate 900
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True
        ))

        self.styles.add(ParagraphStyle(
            name="AcademicH2",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e293b"), # Slate 800
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        ))

        self.styles.add(ParagraphStyle(
            name="AcademicH3",
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#334155"), # Slate 700
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        ))

        self.styles.add(ParagraphStyle(
            name="AcademicBody",
            fontName="Helvetica",
            fontSize=10,
            leading=14.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=10
        ))

        self.styles.add(ParagraphStyle(
            name="AcademicBullet",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
            leftIndent=20,
            firstLineIndent=-10,
            spaceAfter=4
        ))

        self.styles.add(ParagraphStyle(
            name="AcademicCitation",
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#475569"),
            leftIndent=15,
            spaceAfter=6
        ))

    def _clean_markdown(self, text: str) -> str:
        """Escape XML entities and convert basic markdown to HTML-like tags for ReportLab Paragraphs."""
        if not text:
            return ""
            
        # Basic XML entity escape (except we want to allow our tags later)
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        
        # Restore/convert specific markdown patterns to ReportLab tags
        # Bold: **word** -> <b>word</b> (handling escaped entities)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Italic: *word* -> <i>word</i>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Links: [text](url) -> <a> tag
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<font color="#0284c7"><u><a href="\2">\1</a></u></font>', text)
        
        # Let's restore raw bracket escapes if any
        # E.g. we might have converted <a href... to &lt;a href... so let's parse after entity replacement
        text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
        text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
        text = text.replace("&lt;u&gt;", "<u>").replace("&lt;/u&gt;", "</u>")
        text = text.replace("&lt;font", "<font").replace("&lt;/font&gt;", "</font>")
        text = text.replace("&lt;a ", "<a ").replace("&lt;/a&gt;", "</a>")
        text = text.replace("&quot;", '"')
        
        return text

    def generate_pdf(self, query: str, answer_markdown: str, output_path: Path) -> Path:
        """Parse raw query and answer markdown and compile into a styled PDF report."""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=54,  # 0.75 in
            rightMargin=54,
            topMargin=54,   # space for header
            bottomMargin=54 # space for footer
        )
        
        story = []
        
        # Title Banner Block
        story.append(Spacer(1, 10))
        story.append(Paragraph("arXiv Agent Research Report", self.styles["ReportSubtitle"]))
        
        clean_query = self._clean_markdown(query)
        story.append(Paragraph(f"Topic Synthesis: {clean_query}", self.styles["ReportTitle"]))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Ingestion Source: arXiv cs.AI", self.styles["ReportSubtitle"]))
        
        # Draw a divider
        divider = Table([[""]], colWidths=[612 - 108])
        divider.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.HexColor("#0f172a")),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(divider)
        story.append(Spacer(1, 15))
        
        # Process the markdown body line by line
        lines = answer_markdown.split("\n")
        in_bullet_list = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Headers
            if stripped.startswith("# "):
                title_text = self._clean_markdown(stripped[2:])
                story.append(Paragraph(title_text, self.styles["AcademicH1"]))
                story.append(Spacer(1, 4))
            elif stripped.startswith("## "):
                title_text = self._clean_markdown(stripped[3:])
                story.append(Paragraph(title_text, self.styles["AcademicH2"]))
                story.append(Spacer(1, 3))
            elif stripped.startswith("### "):
                title_text = self._clean_markdown(stripped[4:])
                story.append(Paragraph(title_text, self.styles["AcademicH3"]))
                story.append(Spacer(1, 2))
                
            # Bullet lists
            elif stripped.startswith("- ") or stripped.startswith("* "):
                bullet_content = self._clean_markdown(stripped[2:])
                story.append(Paragraph(f"&bull; {bullet_content}", self.styles["AcademicBullet"]))
            elif re.match(r'^\d+\.\s', stripped):
                # Numbered list
                match = re.match(r'^(\d+)\.\s(.*)', stripped)
                num = match.group(1)
                num_content = self._clean_markdown(match.group(2))
                story.append(Paragraph(f"{num}. {num_content}", self.styles["AcademicBullet"]))
                
            # Citations bibliography line
            elif re.match(r'^\[\d+\].*', stripped):
                citation_text = self._clean_markdown(stripped)
                story.append(Paragraph(citation_text, self.styles["AcademicCitation"]))
                
            # Normal paragraph
            else:
                body_text = self._clean_markdown(stripped)
                story.append(Paragraph(body_text, self.styles["AcademicBody"]))
                story.append(Spacer(1, 6))
                
        # Build the document using the NumberedCanvas to add page count dynamically
        doc.build(story, canvasmaker=NumberedCanvas)
        logger.info(f"✓ PDF report successfully compiled at: {output_path}")
        return output_path
