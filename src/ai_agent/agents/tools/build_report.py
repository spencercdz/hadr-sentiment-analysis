from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from datetime import datetime

stylesheets = getSampleStyleSheet()

class SentimentReport(SimpleDocTemplate):
    """Creates a Sentiment Report Class with Proper Formatting"""

    def __init__(self, filename="sentiment_report.pdf", pagesize=A4):
        super().__init__(filename, pagesize=pagesize)
        self.story = []

        # Initialize titles and footings
        self.title = (f"SENTIMENT REPORT (CAA: {datetime.now().strftime('%d %b %Y').upper()}, 2359 HRS)")
        self.logo = Image("../assets/rhcc.jpg", width=85, height=85)
        self.classification = "OFFICIAL (OPEN)"
        self.classification_width = pdfmetrics.stringWidth(self.classification, "Times-Roman", 14)

        # Build styles
        self.title_style     = self.build_title_style()
        self.header_style    = self.build_header_style()
        self.paragraph_style = self.build_normal_text_style()

    def generate_title(self):
        # Add RHCC logo
        self.story.append(self.logo)
        self.story.append(Spacer(1, 24))

        # Add Title
        self.story.append(Paragraph(f"<u>{self.title}</u>", self.title_style))
        self.story.append(Spacer(1, 24))

    def generate_report(self):
        self.generate_title()
        # Add headers & paragraphs with self.header_style, self.paragraph_style…
        self.build(self.story, onFirstPage=self.footers, onLaterPages=self.footers)

    def build_section(self):
        pass

    def build_background(self):
        pass
    
    def build_sentiment(self):
        pass

    def build_monitoring(self):
        pass

    def build_summary(self):
        pass

    def build_title_style(self):
        # Title-text, centered
        s = stylesheets['Title']
        s.fontName = "Times-Bold"
        s.fontSize = 14
        s.leading  = 22
        s.alignment = TA_CENTER

        # Add underline
        s.underlineWidth  = 0.5
        s.underlineOffset = -1
        s.underlineGap    = 0
        s.underlineColor  = colors.black

        return s

    def build_header_style(self):
        # Header text, left-aligned
        s = stylesheets['Heading1']
        s.fontName  = "Times-Bold"
        s.fontSize  = 14
        s.leading   = 22
        s.alignment = TA_LEFT
        return s

    def build_normal_text_style(self):
        # Base body text, left‑aligned
        s = stylesheets['BodyText']
        s.fontName  = "Times-Roman"
        s.fontSize  = 14
        s.leading   = 14
        s.alignment = TA_LEFT

        return s

    def footers(self, canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFont("Times-Roman", 14)

        # Top classification stamp
        canvas_obj.drawString((A4[0] - self.classification_width) / 2,A4[1] - 35, self.classification)

        # Bottom classification stamp
        canvas_obj.drawString((A4[0] - self.classification_width) / 2, 35, self.classification)

        # Footer page number
        page_number_width = pdfmetrics.stringWidth(f"{doc.page}", "Times-Roman", 14)
        canvas_obj.drawString((A4[0] - page_number_width) / 2, 35 + 12, f"{doc.page}")
        canvas_obj.restoreState()

if __name__ == "__main__":
    report = SentimentReport()
    report.generate_report()
