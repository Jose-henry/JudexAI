import os
import re
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.colors import black
from langchain_core.tools import tool

try:
    from PyPDF2 import PdfWriter, PdfReader
except ImportError:
    PdfWriter = None
    PdfReader = None

class PDFManager:
    """Manages PDF creation and appending across chat instances"""
    _instance = None
    _created_files = set()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._created_files = set()
        return cls._instance

    def _create_styles(self):
        """Create custom paragraph styles"""
        styles = getSampleStyleSheet()
        
        # Avoid redefinition error by checking first
        if 'Title' not in styles:
            styles.add(ParagraphStyle(
                name='Title',
                fontSize=16,
                textColor=black,
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        if 'BodyText' not in styles:
            styles.add(ParagraphStyle(
                name='BodyText',
                fontSize=10,
                textColor=black,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                fontName='Helvetica'
            ))
        
        return styles

    def _process_markdown(self, text):
        """Process markdown-like formatting"""
        # Remove double asterisks for bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Convert numbered lists
        text = re.sub(r'^(\d+)\. (.*)$', r'\1. \2', text, flags=re.MULTILINE)
        
        return text

    def save_note(self, note, filename=None, append=False):
        """
        Save or append a note to a PDF file
        
        Args:
            note (str): The text to save
            filename (str, optional): Name of the PDF file
            append (bool, optional): Whether to append to existing file
        
        Returns:
            str: Confirmation message
        """
        # Create notes directory if it doesn't exist
        if not os.path.exists('notes'):
            os.makedirs('notes')

        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'conversation_{timestamp}.pdf'
        
        # Ensure filename ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        # Full path for saving
        full_path = os.path.join('notes', filename)

        # Process the note
        processed_note = self._process_markdown(note)

        # Get custom styles
        styles = self._create_styles()

        # Determine if this is a new file or an append
        if append and os.path.exists(full_path):
            if not PdfWriter or not PdfReader:
                raise ImportError("PyPDF2 is not installed. Please install it to enable appending functionality.")

            from reportlab.pdfgen import canvas

            # Create a new PDF with ReportLab
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)
            
            # Add the new content
            text_object = can.beginText(72, 720)  # Start near the top of the page
            text_object.setFont("Helvetica", 10)
            
            # Add new content lines
            for line in processed_note.split('\n'):
                text_object.textLine(line)
            
            can.drawText(text_object)
            can.showPage()
            can.save()
            
            # Move to the beginning of the StringIO buffer
            packet.seek(0)

            # Create a new PDF writer
            output = PdfWriter()

            # Read existing PDF
            existing_pdf = PdfReader(full_path)
            
            # Add all pages from existing PDF
            for page in existing_pdf.pages:
                output.add_page(page)

            # Add the new page with content
            new_pdf = PdfReader(packet)
            output.add_page(new_pdf.pages[0])

            # Write the merged PDF
            with open(full_path, 'wb') as merged_file:
                output.write(merged_file)

            return f"Note appended to {filename}"
        else:
            # Create a new PDF
            doc = SimpleDocTemplate(full_path, pagesize=letter)

            # Story to build the PDF
            story = []

            # Add title
            story.append(Paragraph(f"Document: {filename}", styles['Title']))

            # Add timestamp
            story.append(Paragraph(f"Created/Updated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyText']))
            story.append(Spacer(1, 12))

            # Split the note into paragraphs
            paragraphs = processed_note.split('\n')

            for para in paragraphs:
                # Skip empty lines
                if not para.strip():
                    story.append(Spacer(1, 6))
                    continue

                # Add paragraph
                story.append(Paragraph(para, styles['BodyText']))

            # Build PDF
            doc.build(story)

            # Track created files
            self._created_files.add(filename)

            return f"Note saved as {filename}"

# Create the tool with the PDF manager
@tool
def note_tool(note, filename=None, append=False):
    """
    Tool to save or append notes to PDFs
    
    Args:
        note (str): The text to save
        filename (str, optional): Name of the PDF file
        append (bool, optional): Whether to append to existing file
    """
    pdf_manager = PDFManager()
    return pdf_manager.save_note(note, filename, append)
