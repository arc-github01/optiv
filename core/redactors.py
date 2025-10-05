"""
Document redaction functions for different file types
"""
import io
import fitz
import docx
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill, Font
from pptx import Presentation
from PIL import ImageFilter
from utils.text_utils import is_sensitive_text, redact_text
from utils.image_utils import (
    blur_faces, detect_and_blur_codes, create_clean_image, 
    blur_small_images
)


def redact_pdf(input_path, output_path):
    """Create redacted PDF by overlaying black rectangles on sensitive text"""
    try:
        doc = fitz.open(input_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Redact text
            text_instances = page.get_text("dict")
            blocks = text_instances.get("blocks", [])
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = " ".join([span.get("text", "") for span in line["spans"]])
                        if is_sensitive_text(line_text):
                            for span in line["spans"]:
                                rect = fitz.Rect(span["bbox"])
                                page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
                        else:
                            for span in line["spans"]:
                                text = span.get("text", "")
                                if is_sensitive_text(text):
                                    rect = fitz.Rect(span["bbox"])
                                    page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
            
            # Process images
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    from PIL import Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    processed_image = pil_image.convert("RGB")
                    
                    # Blur small images
                    processed_image = blur_small_images(processed_image)
                    
                    # Apply face detection and code detection
                    processed_image = blur_faces(processed_image)
                    processed_image = detect_and_blur_codes(processed_image)
                    
                    # Create clean version without metadata
                    processed_image = create_clean_image(processed_image)
                    
                    # Save processed image
                    img_byte_arr = io.BytesIO()
                    processed_image.save(img_byte_arr, format='PNG', exif=b'')
                    img_byte_arr.seek(0)
                    
                    # Try to get image rectangle
                    try:
                        img_rect = page.get_image_rects(img_info)[0]
                    except (IndexError, Exception):
                        print(f"Could not get image rect for image {img_index}")
                        try:
                            img_rect = page.get_image_bbox(img_info)
                        except:
                            print(f"Skipping image {img_index}")
                            continue
                    
                    # Validate rectangle before inserting
                    if img_rect and img_rect.is_valid and not img_rect.is_empty:
                        page.insert_image(img_rect, stream=img_byte_arr.getvalue(), keep_proportion=False)
                    else:
                        print(f"Invalid image rectangle for image {img_index}")
                        
                except Exception as img_error:
                    print(f"Error processing image {img_index} on page {page_num}: {img_error}")
                    continue
        
        doc.save(output_path)
        doc.close()
        return output_path
        
    except Exception as e:
        print(f"Error redacting PDF: {e}")
        import traceback
        traceback.print_exc()
        return None


def redact_docx(input_path, output_path):
    """Create redacted DOCX by replacing sensitive text"""
    try:
        doc = docx.Document(input_path)
        
        for paragraph in doc.paragraphs:
            if paragraph.text:
                redacted_text = redact_text(paragraph.text)
                if redacted_text != paragraph.text:
                    paragraph.text = redacted_text
        
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        redacted_text = redact_text(cell.text)
                        if redacted_text != cell.text:
                            cell.text = redacted_text
        
        doc.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error redacting DOCX: {e}")
        import traceback
        traceback.print_exc()
        return None


def redact_excel(input_path, output_path):
    """Redact sensitive information in Excel files"""
    try:
        if input_path.endswith('.xlsx'):
            wb = openpyxl.load_workbook(input_path)
            black_fill = PatternFill(start_color="000000", end_color="000000", fill_type="solid")
            white_font = Font(color="FFFFFF")
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value is None:
                            continue
                        
                        if is_sensitive_text(str(cell.value)):
                            cell.fill = black_fill
                            cell.font = white_font
                            cell.value = "[REDACTED]"
            
            wb.save(output_path)
            return output_path
        else:
            output_path = output_path.replace('.xls', '.xlsx')
            excel_file = pd.ExcelFile(input_path)
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(input_path, sheet_name=sheet_name)
                    
                    for col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: "[REDACTED]" if pd.notna(x) and is_sensitive_text(str(x)) else x
                        )
                    
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return output_path
            
    except Exception as e:
        print(f"Error redacting Excel: {e}")
        import traceback
        traceback.print_exc()
        return None


def redact_ppt(input_path, output_path):
    """Redact sensitive information in PowerPoint files"""
    try:
        prs = Presentation(input_path)
        
        for slide in prs.slides:
            shapes_to_replace = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    original_text = shape.text
                    redacted_text = redact_text(original_text)
                    
                    if shape.has_text_frame and original_text != redacted_text:
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0] if shape.text_frame.paragraphs else shape.text_frame.add_paragraph()
                        run = p.add_run()
                        run.text = redacted_text
                
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text:
                                original_text = cell.text
                                redacted_text = redact_text(original_text)
                                if original_text != redacted_text:
                                    cell.text_frame.clear()
                                    p = cell.text_frame.paragraphs[0] if cell.text_frame.paragraphs else cell.text_frame.add_paragraph()
                                    run = p.add_run()
                                    run.text = redacted_text
                
                if shape.shape_type == 13:
                    shapes_to_replace.append(shape)

            for shape in shapes_to_replace:
                from PIL import Image
                image = shape.image
                image_bytes = image.blob
                pil_image = Image.open(io.BytesIO(image_bytes))
                processed_image = pil_image.convert("RGB")
                
                processed_image = blur_small_images(processed_image)
                processed_image = blur_faces(processed_image)
                processed_image = detect_and_blur_codes(processed_image)
                processed_image = create_clean_image(processed_image)
                
                with io.BytesIO() as output:
                    processed_image.save(output, format='PNG', exif=b'')
                    output.seek(0)
                    slide.shapes.add_picture(output, shape.left, shape.top, width=shape.width, height=shape.height)
                    sp = shape._element
                    sp.getparent().remove(sp)

        prs.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error redacting PowerPoint: {e}")
        import traceback
        traceback.print_exc()
        return None