"""
Main file processing logic for different document types
"""
import os
from core.extractors import (
    extract_from_pdf, extract_from_docx, extract_from_excel,
    extract_from_ppt
)
from core.redactors import (
    redact_pdf, redact_docx, redact_excel, redact_ppt
)
from core.analyzers import (
    analyze_with_gemini, describe_image_with_gemini,
    parse_analysis_response
)
from utils.text_utils import redact_text
from utils.image_utils import (
    extract_text_from_image, redact_image_text, blur_faces,
    detect_and_blur_codes, create_clean_image, save_secure_image
)
from config.settings import (
    OUTPUT_DIR_NAME, MAX_TEXT_LENGTH, IMAGE_TEXT_LENGTH
)


def process_single_file(file_path):
    """Process a single file: extract text, redact sensitive info, analyze content"""
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
        output_dir = os.path.join(os.path.dirname(fname), OUTPUT_DIR_NAME)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"redacted_{file_name}")
        
        if ext == "pdf":
            raw_text = extract_from_pdf(fname)
            if not raw_text or len(raw_text.strip()) < 10:
                result["File Description"] = "PDF appears empty or could not be read"
                result["Key Findings"] = "Check if PDF is valid and not password protected"
                return result
                
            redacted_text = redact_text(raw_text)
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "PDF", file_name)
            
            redacted_file = redact_pdf(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext == "docx":
            raw_text = extract_from_docx(fname)
            redacted_text = redact_text(raw_text)
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "DOCX", file_name)
            
            redacted_file = redact_docx(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["xlsx", "xls"]:
            raw_text = extract_from_excel(fname)
            redacted_text = redact_text(raw_text)
            
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "Excel", file_name)
            
            redacted_file = redact_excel(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["pptx", "ppt"]:
            raw_text = extract_from_ppt(fname)
            redacted_text = redact_text(raw_text)
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "PowerPoint", file_name)
            
            redacted_file = redact_ppt(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["jpg", "jpeg", "png"]:
            ocr_text, img = extract_text_from_image(fname)
            img = detect_and_blur_codes(img)
            img = redact_image_text(img)
            img = blur_faces(img)
            
            # Create clean version and save securely
            img = create_clean_image(img)
            result["Processed Image"] = img
            
            save_secure_image(img, output_path)
            result["Redacted File"] = output_path
            
            # Analyze using the secure clean image path
            if len(ocr_text.strip()) < 10:
                analysis = describe_image_with_gemini(output_path)
            else:
                redacted_text = redact_text(ocr_text)
                # print(redacted_text[:IMAGE_TEXT_LENGTH])
                analysis = analyze_with_gemini(redacted_text[:IMAGE_TEXT_LENGTH], "Image", file_name)
        else:
            result["File Description"] = "Unsupported file type"
            return result
        
        # Parse analysis response
        description, findings = parse_analysis_response(analysis)
        result["File Description"] = description
        result["Key Findings"] = findings
        
    except Exception as e:
        result["File Description"] = f"Error processing file"
        result["Key Findings"] = str(e)
        import traceback
        traceback.print_exc()
    
    return result