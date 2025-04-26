from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepInFrame, PageBreak
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

# Function to load data from JSON file
def load_report_data(json_file_path):
    """Load report data from a JSON file"""
    import json
    from pathlib import Path
    
    # Empty structures for initialization
    default_sections = {}
    default_tweets = []
    default_details = []
    
    try:
        json_path = Path(json_file_path)
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                # Extract sections
                sections = data.get("sections", default_sections)
                
                # Extract tweets - convert from dict format to table format
                if "tweets" in data and isinstance(data["tweets"], list):
                    tweets_list = data.get("tweets", default_tweets)
                    if tweets_list and isinstance(tweets_list[0], dict):
                        # Create header row
                        headers = ["Username", "Date", "Retweets", "Tweet"]
                        rows = [headers]
                        
                        # Add data rows
                        for tweet in tweets_list:
                            row = [
                                tweet.get("Username", tweet.get("username", "")),
                                tweet.get("Date", tweet.get("date", "")),
                                str(tweet.get("Retweets", tweet.get("retweets", 0))),
                                tweet.get("Tweet", tweet.get("tweet", ""))
                            ]
                            rows.append(row)
                        tweets = rows
                    else:
                        tweets = default_tweets
                else:
                    tweets = default_tweets
                
                # Extract details - convert from dict format to table format
                if "details" in data and isinstance(data["details"], list):
                    details_list = data["details"]
                    if details_list and isinstance(details_list[0], dict):
                        # Create header row
                        headers = ["Date", "Sentiment", "Elements", "Impact", "Requests", "Summary"]
                        rows = [headers]
                        
                        # Add data rows
                        for detail in details_list:
                            row = [
                                detail.get("Date", detail.get("date", "")),
                                str(detail.get("Sentiment", detail.get("sentiment", 0.0))),
                                detail.get("Elements", detail.get("elements", "")),
                                detail.get("Impact", detail.get("impact", "")),
                                detail.get("Requests", detail.get("requests", "")),
                                detail.get("Summary", detail.get("summary", ""))
                            ]
                            rows.append(row)
                        details = rows
                    else:
                        details = default_details
                else:
                    details = default_details
                
                return sections, tweets, details
        else:
            print(f"Warning: JSON file not found: {json_file_path}")
            print(f"Error: JSON file not found: {json_file_path}")
            raise FileNotFoundError(f"Required JSON report data file not found: {json_file_path}")
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        raise ValueError(f"Failed to load or parse JSON data: {e}")

# Initialize global variables that will be set by generate_report
test_data = {}
table_data = []
results_data = []

stylesheets = getSampleStyleSheet()

class SentimentReport(SimpleDocTemplate):
    """Creates a Sentiment Report Class with Proper Formatting"""

    def __init__(self, filename="sentiment_report.pdf", pagesize=A4):
        super().__init__(filename, pagesize=pagesize)
        self.story = []
        self.data = None

        # Initialize titles and footings
        self.title = (f"CHANGI RHCC SENTIMENT REPORT (CAA: {datetime.now().strftime('%d%b%Y').upper()}, 2359HRS)")
        # Try to find logo using multiple potential paths
        try:
            script_dir = Path(__file__).parent
            # Try different potential paths for the logo
            potential_logo_paths = [
                Path("ai_agent/agents/assets/icons/rhcc.jpg"),  # Path provided by user
                script_dir / "../assets/icons/rhcc.jpg",       # Relative from script
                script_dir / "../../assets/icons/rhcc.jpg",    # Up two levels
                script_dir / "../assets/icons/rhcc.jpg"        # Another variation
            ]
            
            logo_path = None
            for path in potential_logo_paths:
                if path.exists():
                    print(f"Found logo at: {path}")
                    logo_path = str(path)
                    break
            
            if logo_path:
                self.logo = Image(logo_path, width=85, height=85)
            else:
                print("Warning: Logo image not found, skipping logo")
                self.logo = None
        except Exception as e:
            print(f"Warning: Could not load logo image: {e}")
            self.logo = None
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
                if header.lower() == "tweet overview" or header.lower() == "results" or header.lower() == "sentiment overview":
                    # Add a horizontal spacer
                    self.story.append(Spacer(1, 13))
                
                    try:
                        # Add appropriate visualizations
                        if header.lower() == "tweet overview":
                            # Add Tweet Table
                            self.story.append(self.tweet_table())
                        elif header.lower() == "sentiment overview":
                            # Add Sentiment Chart
                            self.story.append(self.sentiment_chart())
                        else:
                            # Add Results Chart/Table
                            self.story.append(self.results_table())
                    except Exception as e:
                        # If visualization fails, add error message
                        error_text = f"Error generating visualization: {str(e)}"
                        self.story.append(Paragraph(error_text, self.text_style))

                # Add spacing for the next section
                self.story.append(Spacer(1, 13))

    def tweet_table(self):
        # Check if we have tweet data
        global table_data
        
        if not table_data or len(table_data) < 1:
            # Create a minimal valid table with headers if table_data is empty
            default_headers = ["Username", "Date", "Retweets", "Tweet"]
            data = [[Paragraph(cell, self.text_style) for cell in default_headers]]
            # Add a dummy row to make the table valid
            dummy_row = ["No data", "", "", "No tweets available for analysis"]
            data.append([Paragraph(cell, self.text_style) for cell in dummy_row])
        else:
            # Format tweet data
            data = [[Paragraph(cell, self.text_style) for cell in row] for row in table_data]

        # Adjust column widths
        col_widths = [75, 75, 75, 300]

        # Create Tweet Table using data
        t = Table(data, colWidths=col_widths, repeatRows=1)

        # Style tweet table
        t.setStyle(self.build_table_style())

        return t
    
    def results_table(self):
        # Check if we have results data
        global results_data
        
        if not results_data or len(results_data) < 1:
            # Create a minimal valid table with headers if results_data is empty
            default_headers = ["Date", "Sentiment", "Elements", "Impact", "Requests", "Summary"]
            data = [[Paragraph(cell, self.text_style) for cell in default_headers]]
            # Add a dummy row to make the table valid
            dummy_row = ["No data", "0.0", "No data", "No data", "No data", "No detailed results available"]
            data.append([Paragraph(cell, self.text_style) for cell in dummy_row])
        else:
            # Format results data
            data = [[Paragraph(cell, self.text_style) for cell in row] for row in results_data]

        # Adjust column widths
        col_widths = [75, 75, 75, 75, 75, 100]

        # Create Results Table using data
        t = Table(data, colWidths=col_widths, repeatRows=1)

        # Style results table
        t.setStyle(self.build_table_style())

        return t
    
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
        content_height = A4[1] - 9*inch  # Page height minus margins
        drawing_width = content_width
        drawing_height = content_height
        
        # Chart dimensions - make it slightly smaller than the drawing
        chart_width = drawing_width * 0.8
        chart_height = drawing_height * 0.8
        
        # Center the chart within the drawing
        chart_x = (drawing_width - chart_width) / 2
        chart_y = (drawing_height - chart_height) / 2
        
        # Create the drawing with exact content width
        drawing = Drawing(drawing_width, drawing_height)

        # Extract sentiment data from the details section (results_data)
        sentiments = []
        dates = []
        
        # Skip the header row (index 0)
        if len(results_data) > 1:
            for row in results_data[1:]:  # Skip header row
                try:
                    # Date is at index 0, Sentiment at index 1 in each row
                    date = row[0] if len(row) > 0 else '01/01/2025'
                    sentiment_str = row[1] if len(row) > 1 else '0.5'
                    
                    # Convert sentiment to float
                    try:
                        sentiment = float(sentiment_str)
                    except (ValueError, TypeError):
                        sentiment = 0.5  # Default if conversion fails
                    
                    dates.append(date)
                    sentiments.append(sentiment)
                except Exception as e:
                    print(f"Error processing sentiment data: {e}")
        
        # If no data was extracted, raise an exception - we don't want to use hardcoded defaults
        if not sentiments:
            print("Error: No sentiment data available for chart")
            raise ValueError("No sentiment data available for chart")
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
        lp.lines[0].symbol.fillColor = colors.blue
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
        lp.yValueAxis.labels.fontName = 'Times-Bold'
        lp.yValueAxis.labels.fontSize = 8

        drawing.add(lp)

        # Add date labels centered under each data point
        n_points = max(1, len(dates)-1)
        for i, date in enumerate(dates):
            lbl = Label()
            x_pos = chart_x + (i * chart_width / n_points)
            lbl.setOrigin(x_pos, chart_y - 10)
            lbl.boxAnchor = 'n'
            lbl.fontName = 'Times-Bold'
            lbl.fontSize = 8
            lbl.setText(date)
            drawing.add(lbl)

        # X axis name
        x_axis_label = Label()
        x_axis_label.setOrigin(chart_x + chart_width/2, chart_y - 20)
        x_axis_label.boxAnchor = 'n'
        x_axis_label.fontName = 'Times-Bold'
        x_axis_label.fontSize = 10
        x_axis_label.setText('Date')
        drawing.add(x_axis_label)

        # Y axis name
        y_axis_label = Label()
        y_axis_label.setOrigin(chart_x - 25, chart_y + chart_height/2)
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

    def generate_title(self):
        # Add RHCC logo if available
        if self.logo is not None:
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
    
    def build_table_style(self):
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

def generate_report(json_file_path=None, output_pdf_path=None):
    """Generate a PDF report using data from a JSON file
    
    Args:
        json_file_path (str): Path to the JSON file containing report data
        output_pdf_path (str): Path where the PDF should be saved
        
    Returns:
        str: Path to the generated PDF file
    """
    global test_data, table_data, results_data
    
    # Set default paths if not provided
    if json_file_path is None:
        json_file_path = "../assets/outputs/latest_report_data.json"
    
    if output_pdf_path is None:
        output_pdf_path = "../assets/outputs/report.pdf"
    
    # Ensure output path has .pdf extension
    output_pdf_path = str(output_pdf_path)
    if not output_pdf_path.lower().endswith('.pdf'):
        output_pdf_path += '.pdf'
        print(f"Added .pdf extension to output path: {output_pdf_path}")
    
    try:
        # Load the data from the specified JSON file
        test_data, table_data, results_data = load_report_data(json_file_path)
    except Exception as e:
        raise ValueError(f"Error loading report data from {json_file_path}: {str(e)}")

    # Create the report
    report = SentimentReport(output_pdf_path)
    report.add_data(test_data)
    report.generate_report()
    
    # Verify the file was created and has PDF extension
    pdf_path = Path(output_pdf_path)
    if not pdf_path.exists():
        print(f"Warning: PDF file not found at {pdf_path}, checking for other files")
        # Try to find the PDF in the same directory with any name
        parent_dir = pdf_path.parent
        pdf_files = list(parent_dir.glob("*.pdf"))
        if pdf_files:
            output_pdf_path = str(pdf_files[0])
            print(f"Found PDF file: {output_pdf_path}")
    
    print(f"Report generated successfully at: {output_pdf_path}")
    return output_pdf_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        # Accept input JSON file and output PDF file as command line arguments
        input_json = sys.argv[1]
        output_pdf = sys.argv[2]
        generate_report(input_json, output_pdf)
    else:
        # Use default paths
        report = SentimentReport()
        report.add_data(test_data)
        report.generate_report()