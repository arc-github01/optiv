"""
Main file processing logic for different document types
FIXED: Now uses secure pre-redaction for images
SECURE: Filename anonymization for LLM analysis
"""
import re
import os
import hashlib
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
    process_image_secure, save_secure_image
)
from config.settings import (
    OUTPUT_DIR_NAME, MAX_TEXT_LENGTH, IMAGE_TEXT_LENGTH
)


def anonymize_filename(file_name, ext):
    """
    Create an anonymous filename for LLM analysis.
    LLM will never see the real filename.
    
    Args:
        file_name: Original filename
        ext: File extension
    
    Returns:
        str: Anonymous filename like "document_abc123.pdf"
    """
    # Create a short hash of the filename for tracking (optional)
    file_hash = hashlib.md5(file_name.encode()).hexdigest()[:6]
    
    # Map extensions to generic names
    ext_map = {
        'pdf': 'document',
        'docx': 'document',
        'doc': 'document',
        'xlsx': 'spreadsheet',
        'xls': 'spreadsheet',
        'pptx': 'presentation',
        'ppt': 'presentation',
        'jpg': 'image',
        'jpeg': 'image',
        'png': 'image'
    }
    
    generic_name = ext_map.get(ext.lower(), 'file')
    anonymous_name = f"{generic_name}_{file_hash}.{ext}"
    
    return anonymous_name


def process_single_file(file_path):
    """Process a single file: extract text, redact sensitive info, analyze content"""
    fname = str(file_path)
    file_name = os.path.basename(fname)
    ext = fname.split(".")[-1].lower()
    
    # Create anonymous filename for LLM (LLM never sees real name)
    anonymous_name = anonymize_filename(file_name, ext)
    
    print(f"\n🔒 FILENAME ANONYMIZATION:")
    print(f"  Real filename: {file_name}")
    print(f"  LLM sees: {anonymous_name}")
    print(f"  ✅ Real filename protected from LLM\n")
    
    result = {
        "File Name": file_name,  # Keep real name for user display
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
            # Use anonymous filename for LLM
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "PDF", anonymous_name)
            
            redacted_file = redact_pdf(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext == "docx":
            raw_text = extract_from_docx(fname)
            redacted_text = redact_text(raw_text)
            # Use anonymous filename for LLM
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "DOCX", anonymous_name)
            
            redacted_file = redact_docx(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["xlsx", "xls"]:
            raw_text = extract_from_excel(fname)
            redacted_text = redact_text(raw_text)
            
            # Use anonymous filename for LLM
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "Excel", anonymous_name)
            
            redacted_file = redact_excel(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["pptx", "ppt"]:
            raw_text = extract_from_ppt(fname)
            redacted_text = redact_text(raw_text)
            # Use anonymous filename for LLM
            analysis = analyze_with_gemini(redacted_text[:MAX_TEXT_LENGTH], "PowerPoint", anonymous_name)
            
            redacted_file = redact_ppt(fname, output_path)
            result["Redacted File"] = redacted_file
            
        elif ext in ["jpg", "jpeg", "png"]:
            # ✅ FIXED: Use secure image processing
            print("\n" + "="*60)
            print("🔒 SECURE IMAGE PROCESSING")
            print("="*60)
            
            # Process image securely - redacts FIRST, then extracts
            secure_result = process_image_secure(fname, return_original=False)
            
            redacted_text = secure_result['redacted_text']
            img = secure_result['redacted_image']
            is_valid, issues = secure_result['validation']
            
            # Log validation results
            if not is_valid:
                print(f"⚠️  Validation found {len(issues)} issues:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✅ All sensitive data successfully redacted!")
            
            print("="*60 + "\n")
            
            # Save the redacted image
            result["Processed Image"] = img
            save_secure_image(img, output_path)
            result["Redacted File"] = output_path
            
            # Analyze using the redacted text only
            if len(redacted_text.strip()) < 10:
                # If no text extracted, use image description with anonymous name
                analysis = describe_image_with_gemini(output_path)
            else:
                # Analyze the already-redacted text with anonymous filename
                analysis = analyze_with_gemini(
                    redacted_text[:IMAGE_TEXT_LENGTH], 
                    "Image", 
                    anonymous_name  # ✅ LLM sees anonymous name
                )
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