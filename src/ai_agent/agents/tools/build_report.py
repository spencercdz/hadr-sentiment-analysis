from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.colors import purple, PCMYKColor, black, pink, green, blue
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.legends import LineLegend
from reportlab.graphics.shapes import Drawing, _DrawingEditorMixin
from reportlab.lib.validators import Auto
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.pdfbase.pdfmetrics import stringWidth, EmbeddedType1Face, registerTypeFace, Font, registerFont
from reportlab.graphics.charts.axes import XValueAxis, YValueAxis, AdjYValueAxis, NormalDateXValueAxis
from datetime import datetime
from reportlab.graphics.charts.textlabels import Label

test_data = {
    "Background": "On 28 March 2025, a magnitude 7.7–7.9 earthquake struck the Sagaing Region of Myanmar, with an epicenter close to Mandalay, the country's second-largest city. It was the most powerful earthquake to strike Myanmar since 1912, and the second deadliest in Myanmar's modern history, surpassed only by upper estimates of the 1930 Bago earthquake.",
    "Tweet Overview": "A total of 1026 Tweets were collected as part of this report. These are the 5 most impactful tweets made during this period of time...",
    "Results": "Overall Sentiment Distribution was negative with an average score of 0.12, whereby 0 indicates the most negative sentiment and 1 indicates the most positive sentiment. Based on request classification, here are the results on a whole...",
    "Day Overview": "The chart plot displays the changes in sentiment over time. The table indicates the frequency of requests and its request type reported on the respective days...",
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

sentiment_over_time = {
    # Sample data for sentiment over time (0-1 range)
    "dates": ["29/03/2025", "30/03/2025", "31/03/2025", "01/04/2025", "02/04/2025"],
    "sentiment": [0.35, 0.42, 0.51, 0.48, 0.55]
}

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
                if header.lower() == "tweet overview" or header.lower() == "results" or header.lower() == "day overview":
                    self.story.append(Spacer(1, 13))
                    if header.lower() == "tweet overview":
                        # Add Tweet Table
                        self.story.append(self.tweet_table())
                    elif header.lower() == "day overview":
                        # Add Sentiment Chart
                        self.story.append(self.sentiment_chart())
                    else:
                        # Add Results Chart/Table
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

    def sentiment_chart(self):
        """Creates a centered sentiment chart that's properly positioned within the document."""
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.lineplots import LinePlot
        from reportlab.graphics.charts.textlabels import Label
        from reportlab.graphics.charts.legends import LineLegend
        from reportlab.graphics.widgets.markers import makeMarker
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        # Create a drawing that's exactly the width of the content area
        # A4 is 8.27 × 11.69 inches, with typical margins of 1 inch
        content_width = A4[0] - 2*inch  # Page width minus margins
        drawing_width = content_width
        drawing_height = 250
        
        # Chart dimensions - make it slightly smaller than the drawing
        chart_width = drawing_width * 0.8
        chart_height = 150
        
        # Center the chart within the drawing
        chart_x = (drawing_width - chart_width) / 2
        chart_y = 50
        
        # Create the drawing with exact content width
        drawing = Drawing(drawing_width, drawing_height)

        # Prepare data
        if sentiment_over_time and 'sentiment' in sentiment_over_time and 'dates' in sentiment_over_time:
            sentiments = sentiment_over_time['sentiment']
            dates = sentiment_over_time['dates']
        else:
            sentiments = [0.35, 0.42, 0.51, 0.48, 0.55]
            dates = [str(i+1) for i in range(len(sentiments))]
        data_points = [(i+1, val) for i, val in enumerate(sentiments)]

        # Plot
        lp = LinePlot()
        lp.x = chart_x
        lp.y = chart_y
        lp.width = chart_width
        lp.height = chart_height
        lp.data = [data_points]
        lp.lines[0].strokeColor = colors.red
        lp.lines[0].strokeWidth = 2
        lp.lines.symbol = makeMarker('FilledCircle')
        lp.lines[0].symbol.fillColor = colors.red
        lp.lines[0].symbol.size = 5

        # X axis: show axis line and tick marks for each data point
        lp.xValueAxis.visible = True
        lp.xValueAxis.labels.visible = False
        lp.xValueAxis.tickDown = 5  # Show tick marks
        lp.xValueAxis.tickUp = 0
        lp.xValueAxis.strokeColor = colors.black
        lp.xValueAxis.valueMin = 1
        lp.xValueAxis.valueMax = len(sentiments)
        lp.xValueAxis.valueStep = 1  # Tick at each data point

        # Y axis
        lp.yValueAxis.valueMin = 0
        lp.yValueAxis.valueMax = 1
        lp.yValueAxis.valueStep = 0.1
        lp.yValueAxis.labelTextFormat = '%0.1f'
        lp.yValueAxis.labels.fontName = 'Times-Roman'
        lp.yValueAxis.labels.fontSize = 8

        drawing.add(lp)

        # Add date labels centered under each data point
        n_points = max(1, len(dates)-1)
        for i, date in enumerate(dates):
            lbl = Label()
            x_pos = chart_x + (i * chart_width / n_points)
            lbl.setOrigin(x_pos, chart_y - 10)
            lbl.boxAnchor = 'n'
            lbl.fontName = 'Times-Roman'
            lbl.fontSize = 8
            lbl.setText(date)
            drawing.add(lbl)

        # X axis name
        x_axis_label = Label()
        x_axis_label.setOrigin(chart_x + chart_width/2, chart_y - 30)
        x_axis_label.boxAnchor = 'n'
        x_axis_label.fontName = 'Times-Bold'
        x_axis_label.fontSize = 10
        x_axis_label.setText('Date')
        drawing.add(x_axis_label)

        # Y axis name
        y_axis_label = Label()
        y_axis_label.setOrigin(chart_x - 35, chart_y + chart_height/2)
        y_axis_label.boxAnchor = 'c'
        y_axis_label.angle = 90
        y_axis_label.fontName = 'Times-Bold'
        y_axis_label.fontSize = 10
        y_axis_label.setText('Sentiment Score')
        drawing.add(y_axis_label)

        # Legend
        legend = LineLegend()
        legend.x = chart_x + chart_width/2 - 25  # Center legend
        legend.y = chart_y + chart_height + 10
        legend.fontName = 'Times-Roman'
        legend.fontSize = 8
        legend.colorNamePairs = [(colors.red, 'Sentiment')]
        drawing.add(legend)

        return drawing

    def build_sentiment_chart_style(self, chart):
        """This function is kept for backwards compatibility but no longer used."""
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

if __name__ == "__main__":
    report = SentimentReport()
    report.add_data(test_data)
    report.generate_report()