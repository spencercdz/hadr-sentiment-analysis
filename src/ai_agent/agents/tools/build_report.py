from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from datetime import datetime

stylesheets = getSampleStyleSheet()

class SentimentReport(SimpleDocTemplate):
    """Creates a Sentiment Report Class with Proper Formatting"""

    def __init__(self, filename="sentiment_report.pdf", pagesize=A4):
        super().__init__(filename, pagesize=pagesize)
        self.story = []
        self.data = None

        # Initialize titles and footings
        self.title = (f"CHANGI RHCC SENTIMENT REPORT (CAA: {datetime.now().strftime('%d%b%Y').upper()}, 2359HRS)")
        self.logo = Image("../assets/rhcc.jpg", width=85, height=85)
        self.classification = "OFFICIAL (OPEN)"
        self.classification_width = pdfmetrics.stringWidth(self.classification, "Times-Roman", 14)

        # Build styles
        self.title_style     = self.build_title_style()
        self.header_style    = self.build_header_style()
        self.text_style = self.build_normal_text_style()

    def generate_report(self):
        # Add Front Title
        self.generate_title()

        # Add sections
        self.build_sections()

        # Add headers & paragraphs with self.header_style, self.paragraph_style…
        self.build(self.story, onFirstPage=self.footers, onLaterPages=self.footers)

    def add_data(self, input_data):
        self.data = input_data

    def build_sections(self):
        if isinstance(self.data, dict):
            for header, content in self.data.items():
                # Add Header
                self.story.append(Paragraph(f"{header.upper()}", self.header_style))

                # Add Content
                self.story.append(Paragraph(f"{content}", self.text_style))

                # Check if the section has a Table/Graph/Chart
                if header.lower() == "tweet overview" or header.lower() == "results" or header.lower() == "sentiment over time":
                    self.story.append(Spacer(1, 13))
                    if header.lower() == "tweet overview":
                        # Add Tweet Table
                        self.story.append(self.tweet_table())
                    elif header.lower() == "results":
                        # Add Results Graph
                        pass
                    else:
                        # Add Sentiment Chart
                        pass
                
                # Add spacing for the next section
                self.story.append(Spacer(1, 13))

    def tweet_table(self):
        # Format tweet data
        data = [[Paragraph(cell, self.text_style) for cell in row] for row in table_data]

        # Adjust column widths
        col_widths = [75, 75, 75, 300]

        # Create Tweet Table using data
        t = Table(data, colWidths=col_widths, repeatRows=1)

        # Style tweet table
        t.setStyle(self.build_tweet_table_style())

        return t
    
    def build_results_table(self):
        pass

    def build_sentiment_chart(self):
        pass

    def build_monitoring(self):
        pass

    def generate_title(self):
        # Add RHCC logo
        self.story.append(self.logo)

        # Add spacing
        self.story.append(Spacer(1, 24))

        # Add Title
        self.story.append(Paragraph(f"<u>{self.title}</u>", self.title_style))

    def build_title_style(self):
        # Title-text, centered
        title = stylesheets['Heading1']
        title.fontName = "Times-Bold"
        title.fontSize = 14
        title.leading  = 22
        title.alignment = TA_JUSTIFY

        # Add underline
        title.underlineWidth  = 0.5
        title.underlineOffset = -1
        title.underlineGap    = 0
        title.underlineColor  = colors.black

        return title

    def build_header_style(self):
        # Header text, left-aligned
        header = stylesheets['Heading1']
        header.fontName  = "Times-Bold"
        header.fontSize  = 14
        header.leading   = 22
        header.alignment = TA_LEFT
        
        return header

    def build_normal_text_style(self):
        # Base body text, left‑aligned
        text = stylesheets['BodyText']
        text.fontName  = "Times-Roman"
        text.fontSize  = 13
        text.leading   = 14
        text.alignment = TA_JUSTIFY

        return text
    
    def build_tweet_table_style(self):
        style = TableStyle([
            # grid around every cell
            ('GRID',         (0,0), (-1,-1),    0.5, colors.black),
            # header row (row 0) background and centering
            ('BACKGROUND',   (0,0), (-1,0),      colors.lightgrey),
            ('ALIGN',        (0,0), (-1,0),      'CENTER'),
            ('FONTNAME',     (0,0), (-1,0),      'Times-Bold'),
            # all cells top‑aligned
            ('VALIGN',       (0,0), (-1,-1),     'TOP'),
            # cell padding
            ('LEFTPADDING',  (0,0), (-1,-1),     6),
            ('RIGHTPADDING', (0,0), (-1,-1),     6),
            ('TOPPADDING',   (0,0), (-1,-1),     4),
            ('BOTTOMPADDING',(0,0), (-1,-1),     4),
        ])

        return style

    def footers(self, canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFont("Times-Roman", 14)

        # Top classification stamp
        canvas_obj.drawString((A4[0] - self.classification_width) / 2, A4[1] - 35, self.classification)

        # Bottom classification stamp
        canvas_obj.drawString((A4[0] - self.classification_width) / 2, 35, self.classification)

        # Footer page number
        page_number_width = pdfmetrics.stringWidth(f"{doc.page}", "Times-Roman", 14)
        canvas_obj.drawString((A4[0] - page_number_width) / 2, 35 + 12, f"{doc.page}")
        canvas_obj.restoreState()

test_data = {
    "Background": "On 28 March 2025, a magnitude 7.7–7.9 earthquake struck the Sagaing Region of Myanmar, with an epicenter close to Mandalay, the country's second-largest city. It was the most powerful earthquake to strike Myanmar since 1912, and the second deadliest in Myanmar's modern history, surpassed only by upper estimates of the 1930 Bago earthquake.",
    "Tweet Overview": "A total of 1026 Tweets were collected as part of this report. These are the 5 most impactful tweets made during this period of time...",
    "Results": "Overall Sentiment Distribution was negative with an average score of 0.12, whereby 0 indicates the most negative sentiment and 1 indicates the most positive sentiment. Based on request classification, here are the results on a whole...",
    "Sentiment Over Time": "The chart plot displays the changes in sentiment over time. The table indicates the frequency of requests and its request type reported on the respective days...",
    "Discussion": "Here is an overview of the situation with an understanding on the current sentiment and requirements on the ground...",
    "Recommendation": "In order to address public concerns, here are recommended solutions that we may consider...",
    "Summary" : "In overall, this is a holistic view of the sentiment regarding Myanmar Earthquake 2025..."
}

table_data = [
    [
        "Username",
        "Date",
        "Retweets",
        "Tweet"
    ],
    [
        "Myanmar Now",
        "29/03/2025",
        "133",
        "Yesterday's #earthquakes in #Myanmar and #Thailand killed/injured many hundreds and destroyed several homes and civilian buildings. They need your urgent help and how can you help those affected. Share this post and Donate now 💌 https://t.co/8fSxe6XcmC #Myanmarquake https://t.co/Pkpt3em0ip",
    ],
    [
        "Myanmar Now",
        "29/03/2025",
        "133",
        "Yesterday's #earthquakes in #Myanmar and #Thailand killed/injured many hundreds and destroyed several homes and civilian buildings. They need your urgent help and how can you help those affected. Share this post and Donate now 💌 https://t.co/8fSxe6XcmC #Myanmarquake https://t.co/Pkpt3em0ip",
    ],
    [
        "Myanmar Now",
        "29/03/2025",
        "133",
        "Yesterday's #earthquakes in #Myanmar and #Thailand killed/injured many hundreds and destroyed several homes and civilian buildings. They need your urgent help and how can you help those affected. Share this post and Donate now 💌 https://t.co/8fSxe6XcmC #Myanmarquake https://t.co/Pkpt3em0ip",
    ],
    [
        "Myanmar Now",
        "29/03/2025",
        "133",
        "Yesterday's #earthquakes in #Myanmar and #Thailand killed/injured many hundreds and destroyed several homes and civilian buildings. They need your urgent help and how can you help those affected. Share this post and Donate now 💌 https://t.co/8fSxe6XcmC #Myanmarquake https://t.co/Pkpt3em0ip",
    ],
    [
        "Myanmar Now",
        "29/03/2025",
        "133",
        "Yesterday's #earthquakes in #Myanmar and #Thailand killed/injured many hundreds and destroyed several homes and civilian buildings. They need your urgent help and how can you help those affected. Share this post and Donate now 💌 https://t.co/8fSxe6XcmC #Myanmarquake https://t.co/Pkpt3em0ip",
    ],
]

if __name__ == "__main__":
    report = SentimentReport()
    report.add_data(test_data)
    report.generate_report()