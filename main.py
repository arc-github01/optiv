import os
import io
import re
import zipfile
import tempfile
import shutil
import fitz
import docx
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pytesseract
from facenet_pytorch import MTCNN
from transformers import ViTFeatureExtractor, ViTForImageClassification

import spacy

import openpyxl
from openpyxl.styles import PatternFill, Font
from pptx import Presentation
from pptx.util import Inches

import gradio as gr
import google.generativeai as genai

import torch

from ctypes import CDLL
from pyzbar import pyzbar


dll_path = os.path.join(os.path.dirname(__file__), "libzbar-64.dll")
CDLL(dll_path)


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


nlp = spacy.load("en_core_web_sm")

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
mtcnn = MTCNN(keep_all=True, device="cpu")
genai.configure(api_key="your_api_key")


def extract_from_pdf(path):
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
    docx_file = docx.Document(path)
    text_content = []
    
    for paragraph in docx_file.paragraphs:
        text_content.append(paragraph.text)
    
    for table in docx_file.tables:
        for row in table.rows:
            row_text = [cell.text for cell in row.cells]
            text_content.append("\t".join(row_text))
    
    return "\n".join(text_content)

def extract_from_excel(path):
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

def redact_excel(input_path, output_path):
    try:
        if input_path.endswith('.xlsx'):
            wb = openpyxl.load_workbook(input_path)
            black_fill = PatternFill(start_color="000000", end_color="000000", fill_type="solid")
            white_font = Font(color="FFFFFF")
            
            sensitive_patterns = [
                r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
                r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
                r"\b\d{4}\s?\d{4}\s?\d{4}\b",
                r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                r"\b(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16}\b",
                r"\bAIza[0-9A-Za-z\-_]{35}\b",
                r"\barn:aws:",
                r"\b\d{12}\b",
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                r"\b\d{3}-\d{2}-\d{4}\b",
                r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}",
                r"\b[A-Za-z0-9]{30,}\b",
            ]
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                
                # FIXED: Iterate correctly over rows
                for row in sheet.iter_rows():
                    for cell in row:  # Changed from row.cells to row
                        if cell.value is None:
                            continue
                        
                        cell_text = str(cell.value)
                        is_sensitive = False
                        
                        for pattern in sensitive_patterns:
                            if re.search(pattern, cell_text, re.IGNORECASE):
                                is_sensitive = True
                                break
                        
                        if not is_sensitive and len(cell_text) > 2 and not cell_text.isdigit():
                            try:
                                doc = nlp(cell_text)
                                if any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC"] for ent in doc.ents):
                                    is_sensitive = True
                            except:
                                pass
                        
                        if is_sensitive:
                            cell.fill = black_fill
                            cell.font = white_font
                            cell.value = "[REDACTED]"
            
            wb.save(output_path)
            return output_path
        else:
            excel_file = pd.ExcelFile(input_path)
            output_path = output_path.replace('.xls', '.xlsx')
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(input_path, sheet_name=sheet_name)
                    
                    for col in df.columns:
                        df[col] = df[col].apply(lambda x: "[REDACTED]" if pd.notna(x) and is_sensitive_text(str(x)) else x)
                    
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return output_path
            
    except Exception as e:
        print(f"Error redacting Excel: {e}")
        import traceback
        traceback.print_exc()
        return None

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
                        for span in line["spans"]:
                            text = span.get("text", "")
                            if is_sensitive_text(text):
                                rect = fitz.Rect(span["bbox"])
                                page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
            
            # Redact images (blur faces)
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes))
                processed_image = pil_image.convert("RGB")
                processed_image = blur_faces(processed_image)
                processed_image = detect_and_blur_codes(processed_image)
                
                # Convert blurred image to a format PyMuPDF understands (like PNG)
                img_byte_arr = io.BytesIO()
                processed_image.save(img_byte_arr, format='PNG')
                
                # Replace image
                img_rect = page.get_image_bbox(img_info)
                page.insert_image(img_rect, stream=img_byte_arr.getvalue(), keep_proportion=False)
        
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
        
        # Redact paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text:
                redacted_text = redact_text(paragraph.text)
                if redacted_text != paragraph.text:
                    paragraph.text = redacted_text
        
        # Redact tables
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

def extract_from_ppt(path):
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

def redact_ppt(input_path, output_path):
    try:
        prs = Presentation(input_path)
        
        for slide in prs.slides:
            shapes_to_replace = []

            for shape in slide.shapes:
                # Redact text in shapes
                if hasattr(shape, "text") and shape.text:
                    original_text = shape.text
                    redacted_text = redact_text(original_text)
                    
                    if shape.has_text_frame and original_text != redacted_text:
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0] if shape.text_frame.paragraphs else shape.text_frame.add_paragraph()
                        run = p.add_run()
                        run.text = redacted_text
                
                # Redact text in tables
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
                
                # Redact images (blur faces)
                if shape.shape_type == 13: # 13 is the type for a Picture
                    shapes_to_replace.append(shape)

            for shape in shapes_to_replace:
                image = shape.image
                image_bytes = image.blob
                pil_image = Image.open(io.BytesIO(image_bytes))
                processed_image = pil_image.convert("RGB")
                processed_image = blur_faces(processed_image)
                processed_image = detect_and_blur_codes(processed_image)
                
                with io.BytesIO() as output:
                    processed_image.save(output, format='PNG')
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

def extract_from_image(path):
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text, img

def is_sensitive_text(text):
    sensitive_patterns = [
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        r"\b(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16}\b",
        r"\bAIza[0-9A-Za-z\-_]{35}\b",
        r"\b[A-Za-z0-9]{30,}\b",
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, str(text), re.IGNORECASE):
            return True
    
    try:
        if len(str(text)) > 2:
            doc = nlp(str(text))
            if any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC"] for ent in doc.ents):
                return True
    except:
        pass
    
    return False

def redact_text(text):
    patterns = {
        "EMAIL": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
        "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "VEHICLE": r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}\b",
        "DOB": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "DATE": r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))?\b",
        "IPV4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:/\d{1,2})?\b",
        "IPV6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
        "MAC_ADDRESS": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b",
        "PORT": r"\b:\d{2,5}\b(?=\s|,|$)",
        "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|APKA)[A-Z0-9]{16}\b",
        "AWS_SECRET_KEY": r"\b[A-Za-z0-9/+=]{40}\b(?=\s|,|\"|}|$)",
        "AWS_ARN": r"\barn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:[a-zA-Z0-9\-/]+",
        "AWS_ACCOUNT_ID": r"\b\d{12}(?=:|\"|\s|,|})",
        "API_KEY": r"\b[A-Za-z0-9_\-]{32,45}\b",
        "JWT_TOKEN": r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
        "BEARER_TOKEN": r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*",
        "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z\-_]{35}\b",
        "GITHUB_TOKEN": r"\bghp_[A-Za-z0-9]{36}\b",
        "GITHUB_OAUTH": r"\bgho_[A-Za-z0-9]{36}\b",
        "PASSWORD_FIELD": r"(?i)(?:password|passwd|pwd|secret)[\"\s]*[:=][\"\s]*[^\s\"}{,]{6,}",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "DB_CONNECTION": r"(?i)(?:mongodb|mysql|postgresql|jdbc):\/\/[^\s\"'<>]+",
        "PRIVATE_KEY": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[^-]+-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "LONG_ALPHANUMERIC": r"\b[A-Za-z0-9]{50,}\b",
    }
    
    priority_patterns = [
        "PRIVATE_KEY", "JWT_TOKEN", "BEARER_TOKEN", "AWS_ARN", 
        "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "GOOGLE_API_KEY",
        "GITHUB_TOKEN", "GITHUB_OAUTH", "DB_CONNECTION",
        "EMAIL", "IPV4", "IPV6", "MAC_ADDRESS", 
        "AWS_ACCOUNT_ID", "AADHAAR", "PAN", "CREDIT_CARD", "SSN",
        "PHONE", "VEHICLE", "DATE", "DOB"
    ]
    
    for label in priority_patterns:
        if label in patterns:
            text = re.sub(patterns[label], f"[REDACTED_{label}]", text, flags=re.IGNORECASE if label == "PASSWORD_FIELD" else 0)
    
    for label, pat in patterns.items():
        if label not in priority_patterns:
            text = re.sub(pat, f"[REDACTED_{label}]", text)
    
    try:
        doc = nlp(text)
        redacted_positions = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG", "LOC", "FAC"]:
                overlap = any(not (ent.start_char >= end or ent.end_char <= start) for start, end in redacted_positions)
                if not overlap and len(ent.text) > 2:
                    text = text[:ent.start_char] + f"[REDACTED_{ent.label_}]" + text[ent.end_char:]
                    new_end = ent.start_char + len(f"[REDACTED_{ent.label_}]")
                    redacted_positions.append((ent.start_char, new_end))
    except Exception as e:
        print(f"SpaCy NER error: {e}")
    
    return text

def redact_image_text(img):
    # New approach: Process text blocks to handle multi-word patterns like Aadhaar
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Group words by block
    blocks = {}
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            block_num = data['block_num'][i]
            if block_num not in blocks:
                blocks[block_num] = []
            blocks[block_num].append(i)

    sensitive_patterns = [
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",  # Aadhaar
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}",
    ]
    
    # Add more comprehensive patterns from redact_text
    sensitive_patterns.extend([
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:/\d{1,2})?\b", # IPV4
        r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b", # MAC_ADDRESS
        r"\b(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16}\b", # AWS_ACCESS_KEY
        r"\bAIza[0-9A-Za-z\-_]{35}\b", # GOOGLE_API_KEY
        r"\bghp_[A-Za-z0-9]{36}\b", # GITHUB_TOKEN
        r"\beyJ[A-Za-z0-9_\-]+\.", # JWT
        r"\b\d{12}\b", # AWS_ACCOUNT_ID
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b", # CREDIT_CARD
    ])

    for block_num, word_indices in blocks.items():
        block_text = " ".join([data['text'][i] for i in word_indices])
        
        for pattern in sensitive_patterns:
            for match in re.finditer(pattern, block_text):
                start, end = match.span()
                
                # Find which words correspond to the match
                current_pos = 0
                matched_indices = []
                for i in word_indices:
                    word = data['text'][i]
                    word_len = len(word)
                    if start <= current_pos < end or start < current_pos + word_len <= end:
                        matched_indices.append(i)
                    current_pos += word_len + 1 # +1 for space

                if matched_indices:
                    x1 = min(data['left'][i] for i in matched_indices)
                    y1 = min(data['top'][i] for i in matched_indices)
                    x2 = max(data['left'][i] + data['width'][i] for i in matched_indices)
                    y2 = max(data['top'][i] + data['height'][i] for i in matched_indices)
                    
                    padding = 8
                    x = max(0, x1 - padding)
                    y = max(0, y1 - padding)
                    w = min(img.width - x, (x2 - x1) + 2 * padding)
                    h = min(img.height - y, (y2 - y1) + 2 * padding)
                    
                    if w > 0 and h > 0:
                        region = img.crop((x, y, x + w, y + h))
                        region = region.filter(ImageFilter.GaussianBlur(30))
                        img.paste(region, (x, y))
    
    return img

def detect_and_blur_codes(img):
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        decoded_objects = pyzbar.decode(img_cv)
        
        for obj in decoded_objects:
            points = obj.polygon
            if len(points) == 4:
                x = min([p.x for p in points])
                y = min([p.y for p in points])
                w = max([p.x for p in points]) - x
                h = max([p.y for p in points]) - y
                
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.width - x, w + 2 * padding)
                h = min(img.height - y, h + 2 * padding)
                
                region = img.crop((x, y, x + w, y + h))
                region = region.filter(ImageFilter.GaussianBlur(35))
                img.paste(region, (x, y))
    except Exception as e:
        print(f"QR/Barcode detection error: {e}")
    
    return img

def blur_faces(img):
    try:
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = img.crop((x1, y1, x2, y2))
                face = face.filter(ImageFilter.GaussianBlur(30))
                img.paste(face, (x1, y1))
    except Exception as e:
        print(f"Face detection error: {e}")
    return img

def analyze_with_gemini(text, file_type, file_name):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        prompt = f"""Analyze the following content from a {file_type} file named "{file_name}".

Content:
{text}

Please provide a structured analysis in the following format:
1. File Description: A brief description of what this file contains (2-3 sentences)
2. Key Findings: List 3-4 bullet points of the most important findings or insights

Be concise and focus on the main purpose and key information."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        
        return f"Analysis failed: {e}"

def describe_image_with_gemini(img_path):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        prompt = """Analyze this image and provide:
1. File Description: What is shown in the image? (2-3 sentences)
2. Key Findings: List 3-4 key observations or insights about what's in the image

Format your response clearly with these two sections."""

        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": img_bytes}
        ])
        return response.text
    except Exception as e:
       
        return f"Image analysis failed: {e}"

def process_single_file(file_path):
    fname = str(file_path)
    file_name = os.path.basename(fname)
    ext = fname.split(".")[-1].lower()
    
    result = {
        "File Name": file_name,
        "File Type": f".{ext}",
        "File Description": "",
        "Key Findings": "",
        "Processed Image": None,
        "Redacted File": None
    }

    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(fname), "redacted_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"redacted_{file_name}")
        
        if ext == "pdf":
            raw_text = extract_from_pdf(fname)
            if not raw_text or len(raw_text.strip()) < 10:
                result["File Description"] = "PDF appears empty or could not be read"
                result["Key Findings"] = "Check if PDF is valid and not password protected"
                return result
                
            redacted_text = redact_text(raw_text)
            analysis = analyze_with_gemini(redacted_text[:5000], "PDF", file_name)
            
            # Create redacted PDF
            redacted_file = redact_pdf(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext == "docx":
            raw_text = extract_from_docx(fname)
            redacted_text = redact_text(raw_text)
            analysis = analyze_with_gemini(redacted_text[:5000], "DOCX", file_name)
            
            # Create redacted DOCX
            redacted_file = redact_docx(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["xlsx", "xls"]:
            raw_text = extract_from_excel(fname)
            redacted_text = redact_text(raw_text)
            analysis = analyze_with_gemini(redacted_text[:5000], "Excel", file_name)
            
            redacted_file = redact_excel(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["pptx", "ppt"]:
            raw_text = extract_from_ppt(fname)
            redacted_text = redact_text(raw_text)
            analysis = analyze_with_gemini(redacted_text[:5000], "PowerPoint", file_name)
            
            redacted_file = redact_ppt(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["jpg", "jpeg", "png"]:
            ocr_text, img = extract_from_image(fname)
            img = detect_and_blur_codes(img)
            img = redact_image_text(img)
            img = blur_faces(img)
            result["Processed Image"] = img
            
            img.save(output_path)
            result["Redacted File"] = output_path
            
            if len(ocr_text.strip()) < 10:
                analysis = describe_image_with_gemini(fname)
            else:
                redacted_text = redact_text(ocr_text)
                analysis = analyze_with_gemini(redacted_text[:3000], "Image", file_name)
        else:
            result["File Description"] = "Unsupported file type"
            return result
        
        # Parse analysis results
        lines = analysis.split('\n')
        description_lines = []
        findings_lines = []
        in_description = False
        in_findings = False
        
        for line in lines:
            line = line.strip()
            if 'file description' in line.lower() or line.startswith('1.'):
                in_description = True
                in_findings = False
                if ':' in line:
                    desc_text = line.split(':', 1)[1].strip()
                    if desc_text:
                        description_lines.append(desc_text)
                continue
            elif 'key findings' in line.lower() or line.startswith('2.'):
                in_description = False
                in_findings = True
                continue
            
            if in_description and line:
                description_lines.append(line)
            elif in_findings and line:
                findings_lines.append(line)
        
        result["File Description"] = ' '.join(description_lines) if description_lines else "Analysis completed"
        result["Key Findings"] = '\n'.join(findings_lines) if findings_lines else "See description"
        
    except Exception as e:
        result["File Description"] = f"Error processing file"
        result["Key Findings"] = str(e)
        import traceback
        traceback.print_exc()
    
    return result

def process_directory(directory_path):
    if not directory_path or not os.path.exists(directory_path):
        return "Invalid directory path", None, None
    
    supported_extensions = ['.pdf', '.docx', '.xlsx', '.xls', '.pptx', '.ppt', '.jpg', '.jpeg', '.png']
    files = []
    
    path_obj = Path(directory_path)
    for ext in supported_extensions:
        files.extend(path_obj.glob(f'*{ext}'))
        files.extend(path_obj.glob(f'*{ext.upper()}'))
    
    if not files:
        return "No supported files found in directory", None, None
    
    results = []
    processed_images = []
    redacted_files_list = []
    
    for file_path in sorted(files):
        print(f"Processing: {file_path}")
        result = process_single_file(file_path)
        results.append(result)
        if result["Processed Image"] is not None:
            processed_images.append((result["File Name"], result["Processed Image"]))
        if result["Redacted File"] is not None:
            redacted_files_list.append(result["Redacted File"])
    
    df = pd.DataFrame(results)
    df = df[["File Name", "File Type", "File Description", "Key Findings"]]
    
    html_output = """
    <style>
        .analysis-table {
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .analysis-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 12px 15px;
            text-align: left;
            border: 1px solid #ddd;
        }
        .analysis-table td {
            padding: 12px 15px;
            border: 1px solid #ddd;
            vertical-align: top;
            line-height: 1.4;
        }
        .analysis-table tr:nth-child(even) {
            background-color: #000000;
        }
        .analysis-table tr:nth-child(odd) {
            background-color: #000000;
        }
        
        .file-type {
            font-weight: bold;
            color: #5b9bd5;
        }
        .key-findings {
            white-space: pre-line;
        }
        .key-findings::before {
            content: "• ";
        }
    </style>
    
    <table class="analysis-table">
        <thead>
            <tr>
                <th>File Name</th>
                <th>File Type</th>
                <th>File Description</th>
                <th>Key Findings</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df.iterrows():
        findings_formatted = row['Key Findings'].replace('\n', '\n• ')
        if findings_formatted and not findings_formatted.startswith('•'):
            findings_formatted = '• ' + findings_formatted
        
        html_output += f"""
            <tr>
                <td>{row['File Name']}</td>
                <td class="file-type">{row['File Type']}</td>
                <td>{row['File Description']}</td>
                <td class="key-findings">{findings_formatted}</td>
            </tr>
        """
    
    html_output += """
        </tbody>
    </table>
    
    
    """
    
    image_gallery = None
    if processed_images:
        image_gallery = [img for _, img in processed_images]
    
    download_info = f"Redacted files saved in: {os.path.join(directory_path, 'redacted_output')}" if redacted_files_list else "No files required redaction"
    
    return html_output, image_gallery, download_info

with gr.Blocks(title="Enhanced Batch File Analyzer", theme=gr.themes.Soft()) as iface:
    
    with gr.Row():
        directory_input = gr.Textbox(
            label="Directory Path",
            placeholder="C:/Users/YourName/Documents/FilesToAnalyze",
            lines=1,
            info="Enter the full path to the directory containing your files"
        )
    
    analyze_btn = gr.Button("Analyze All Files", variant="primary", size="lg")
    
    with gr.Row():
        output_html = gr.HTML(label="Analysis Results")
    
    
    
    analyze_btn.click(
        fn=process_directory,
        inputs=[directory_input],
        outputs=[output_html]
    )

if __name__ == "__main__":
    iface.launch(share=False, server_name="0.0.0.0", server_port=7860)