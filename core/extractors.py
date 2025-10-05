"""
Document content extraction utilities
"""
import fitz
import docx
import pandas as pd
import openpyxl
from pptx import Presentation
from utils.image_utils import extract_text_from_image


def extract_from_pdf(path):
    """Extract text from PDF file"""
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        import traceback
        traceback.print_exc()
        return ""


def extract_from_docx(path):
    """Extract text from DOCX file"""
    try:
        docx_file = docx.Document(path)
        text_content = []
        
        for paragraph in docx_file.paragraphs:
            text_content.append(paragraph.text)
        
        for table in docx_file.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                text_content.append("\t".join(row_text))
        
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        import traceback
        traceback.print_exc()
        return ""


def extract_from_excel(path):
    """Extract text from Excel file"""
    try:
        if path.endswith('.xlsx'):
            wb = openpyxl.load_workbook(path, data_only=True)
            text_content = []
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_content.append(f"\n=== Sheet: {sheet_name} ===\n")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = [str(cell) if cell is not None else "" for cell in row]
                    if any(row_text):
                        text_content.append("\t".join(row_text))
            
            return "\n".join(text_content)
        else:
            excel_file = pd.ExcelFile(path)
            text_content = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet_name)
                text_content.append(f"\n=== Sheet: {sheet_name} ===\n")
                df = df.fillna('')
                text_content.append(df.to_string(index=False))
            return "\n".join(text_content)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return f"Error reading Excel file: {e}"


def extract_from_ppt(path):
    """Extract text from PowerPoint file"""
    try:
        prs = Presentation(path)
        text_content = []
        
        for slide_num, slide in enumerate(prs.slides, 1):
            text_content.append(f"\n=== Slide {slide_num} ===\n")
            
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text_content.append(shape.text)
                
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        row_text = [cell.text for cell in row.cells]
                        text_content.append("\t".join(row_text))
                
                if hasattr(shape, "text_frame") and shape.text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            if run.text:
                                text_content.append(run.text)
        
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error reading PowerPoint file: {e}")
        import traceback
        traceback.print_exc()
        return f"Error reading PowerPoint file: {e}"