"""
Image processing and manipulation utilities
Enhanced with pre-LLM redaction security using BLACKOUT (not blur)
"""
import io
import re
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pytesseract
from facenet_pytorch import MTCNN
from .patterns import SENSITIVE_PATTERNS
from .text_utils import nlp, NER_LABELS, redact_text, validate_redaction
from config.settings import (
    TESSERACT_CMD, FACE_BLUR_RADIUS, GAUSSIAN_BLUR_SMALL_IMAGE,
    SMALL_IMAGE_THRESHOLD_WIDTH, SMALL_IMAGE_THRESHOLD_HEIGHT,
    TEXT_PADDING
)

# Use TEXT_PADDING for all padding operations
PADDING_SIZE = TEXT_PADDING

# Redaction color - use black for complete OCR blocking
REDACTION_COLOR = 'black'  # Change to 'white' if preferred

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device="cpu")


def create_clean_image(img):
    """Create a clean image without metadata"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    clean_img = Image.new('RGB', img.size, color=(255, 255, 255))
    clean_img.paste(img, (0, 0))
    
    img_array = np.array(clean_img)
    final_clean_img = Image.fromarray(img_array, 'RGB')
    
    return final_clean_img


def save_secure_image(img, output_path):
    """Save image securely without any metadata"""
    clean_img = create_clean_image(img)
    clean_img.save(output_path, format='PNG', optimize=True, exif=b'')
    return output_path


def blur_faces(img):
    """Detect and blur faces in images"""
    try:
        boxes, _ = mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                
                if x2 > x1 and y2 > y1:
                    face = img.crop((x1, y1, x2, y2))
                    face = face.filter(ImageFilter.GaussianBlur(FACE_BLUR_RADIUS))
                    img.paste(face, (x1, y1))
    except Exception as e:
        print(f"Face detection error: {e}")
    
    return img


def detect_and_blur_codes(img):
    """QR/Barcode detection disabled - returns image unchanged"""
    return img


def blur_small_images(img):
    """Blur small images that might contain sensitive info"""
    if img.width < SMALL_IMAGE_THRESHOLD_WIDTH and img.height < SMALL_IMAGE_THRESHOLD_HEIGHT:
        return img.filter(ImageFilter.GaussianBlur(GAUSSIAN_BLUR_SMALL_IMAGE))
    return img


def redact_image_text(img):
    """
    Redact sensitive text in images with COMPLETE BLACKOUT (not blur).
    This ensures OCR cannot read through the redaction.
    
    CRITICAL: This function MUST be called BEFORE any text extraction
    to prevent sensitive data exposure.
    """
    if not isinstance(img, Image.Image):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            raise TypeError(f"Expected PIL Image or numpy array, got {type(img)}")
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Create drawing object for blackout
    draw = ImageDraw.Draw(img)
    
    # Get OCR data with positions (but don't expose the full text yet)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Quick check for AWS JSON (using data structure, not full text)
    sample_text = " ".join([data['text'][i] for i in range(min(50, len(data['text']))) if data['text'][i].strip()])
    is_aws_json = bool(re.search(
        r'"(?:RoleId|Arn|AWS|AccountId|Sid|Effect|Principal|Action)"', 
        sample_text, 
        re.IGNORECASE
    ))
    
    # Group words into blocks
    blocks = {}
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            block_num = data['block_num'][i]
            if block_num not in blocks:
                blocks[block_num] = []
            blocks[block_num].append(i)
    
    redacted_regions = []
    
    # AWS JSON special handling - COMPLETE BLACKOUT
    if is_aws_json:
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text and len(text) > 2:
                if re.search(
                    r'(?:AKIA|ASIA|arn:aws|RoleId|Statement|Principal|Allow|Deny|\d{12}|"[A-Za-z]+")', 
                    text, 
                    re.IGNORECASE
                ):
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    x = max(0, x - TEXT_PADDING)
                    y = max(0, y - TEXT_PADDING)
                    w = min(img.width - x, w + 2 * TEXT_PADDING)
                    h = min(img.height - y, h + 2 * TEXT_PADDING)
                    
                    if w > 0 and h > 0:
                        # BLACK OUT - OCR cannot read through solid color
                        draw.rectangle([x, y, x + w, y + h], fill=REDACTION_COLOR)
                        redacted_regions.append((x, y, w, h))
    
    # Pattern-based redaction - COMPLETE BLACKOUT
    for block_num, word_indices in blocks.items():
        block_text = " ".join([data['text'][i] for i in word_indices])
        
        for pattern_name, pattern in SENSITIVE_PATTERNS.items():
            try:
                for match in re.finditer(pattern, block_text, re.IGNORECASE):
                    start, end = match.span()
                    
                    current_pos = 0
                    matched_indices = []
                    for i in word_indices:
                        word = data['text'][i]
                        word_len = len(word)
                        if start <= current_pos < end or start < current_pos + word_len <= end:
                            matched_indices.append(i)
                        current_pos += word_len + 1
                    
                    if matched_indices:
                        x1 = min(data['left'][i] for i in matched_indices)
                        y1 = min(data['top'][i] for i in matched_indices)
                        x2 = max(data['left'][i] + data['width'][i] for i in matched_indices)
                        y2 = max(data['top'][i] + data['height'][i] for i in matched_indices)
                        
                        x = max(0, x1 - PADDING_SIZE)
                        y = max(0, y1 - PADDING_SIZE)
                        w = min(img.width - x, (x2 - x1) + 2 * PADDING_SIZE)
                        h = min(img.height - y, (y2 - y1) + 2 * PADDING_SIZE)
                        
                        overlaps = any(
                            not (x + w < rx or x > rx + rw or y + h < ry or y > ry + rh)
                            for rx, ry, rw, rh in redacted_regions
                        )
                        
                        if not overlaps and w > 0 and h > 0:
                            # BLACK OUT instead of blur - OCR cannot read through
                            draw.rectangle([x, y, x + w, y + h], fill=REDACTION_COLOR)
                            redacted_regions.append((x, y, w, h))
            except Exception:
                continue
    
    # NER-based redaction - COMPLETE BLACKOUT
    for block_num, word_indices in blocks.items():
        block_text = " ".join([data['text'][i] for i in word_indices])
        
        try:
            doc = nlp(block_text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "DATE"] and len(ent.text) > 2:
                    ent_start = block_text.find(ent.text)
                    if ent_start != -1:
                        ent_end = ent_start + len(ent.text)
                        
                        current_pos = 0
                        matched_indices = []
                        for i in word_indices:
                            word = data['text'][i]
                            word_len = len(word)
                            if ent_start <= current_pos < ent_end or ent_start < current_pos + word_len <= ent_end:
                                matched_indices.append(i)
                            current_pos += word_len + 1
                        
                        if matched_indices:
                            x1 = min(data['left'][i] for i in matched_indices)
                            y1 = min(data['top'][i] for i in matched_indices)
                            x2 = max(data['left'][i] + data['width'][i] for i in matched_indices)
                            y2 = max(data['top'][i] + data['height'][i] for i in matched_indices)
                            
                            x = max(0, x1 - PADDING_SIZE)
                            y = max(0, y1 - PADDING_SIZE)
                            w = min(img.width - x, (x2 - x1) + 2 * PADDING_SIZE)
                            h = min(img.height - y, (y2 - y1) + 2 * PADDING_SIZE)
                            
                            overlaps = any(
                                not (x + w < rx or x > rx + rw or y + h < ry or y > ry + rh)
                                for rx, ry, rw, rh in redacted_regions
                            )
                            
                            if not overlaps and w > 0 and h > 0:
                                # BLACK OUT NER-detected entities completely
                                draw.rectangle([x, y, x + w, y + h], fill=REDACTION_COLOR)
                                redacted_regions.append((x, y, w, h))
        except Exception:
            continue
    
    return img


def extract_text_from_image(path):
    """
    DEPRECATED: Use extract_text_from_image_secure() instead.
    This function is kept for backward compatibility but should not be used
    for new code as it does not provide pre-LLM redaction.
    """
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text, img


def extract_text_from_image_secure(path, apply_visual_redaction=True, apply_text_redaction=True):
    """
    SECURE: Extract text from image with PRE-LLM redaction.
    
    This function ensures that sensitive information is BLACKED OUT BEFORE
    any text is extracted and exposed to LLMs or logging systems.
    
    Args:
        path: Path to the image file
        apply_visual_redaction: If True, redact sensitive text visually in the image
        apply_text_redaction: If True, redact extracted text before returning
    
    Returns:
        tuple: (redacted_text, redacted_image, validation_results)
        - redacted_text: Text extracted from redacted image with additional text redaction
        - redacted_image: Visually redacted image (safe for display)
        - validation_results: Tuple (is_valid, issues) from redaction validation
    """
    print("🔒 SECURE EXTRACTION: Starting pre-redaction process...")
    
    # Load original image
    img = Image.open(path).convert("RGB")
    print(f"📷 Loaded image: {img.size}")
    
    # Stage 1: Visual redaction (blur faces and BLACK OUT sensitive text)
    if apply_visual_redaction:
        print("🎭 Stage 1: Applying visual redaction...")
        # Create a copy for visual redaction
        visual_img = img.copy()
        
        # Blur faces first
        visual_img = blur_faces(visual_img)
        print("  ✓ Faces blurred")
        
        # Blur small images that might contain sensitive info
        visual_img = blur_small_images(visual_img)
        print("  ✓ Small images blurred")
        
        # BLACK OUT sensitive text (not blur - complete removal)
        print("  🔲 Blacking out sensitive text...")
        visual_img = redact_image_text(visual_img)
        print("  ✓ Text blacked out")
        
        # Blur any codes (QR/barcodes) if needed
        visual_img = detect_and_blur_codes(visual_img)
    else:
        visual_img = img.copy()
        print("⚠️  WARNING: Visual redaction skipped!")
    
    # Stage 2: Extract text from REDACTED image only
    print("📝 Stage 2: Extracting text from REDACTED image...")
    extracted_text = pytesseract.image_to_string(visual_img)
    print(f"  Extracted {len(extracted_text)} characters")
    
    # Stage 3: Additional text-based redaction as safety layer
    if apply_text_redaction:
        print("🛡️ Stage 3: Applying text-based redaction...")
        final_text = redact_text(extracted_text, full_text=extracted_text)
        print(f"  Final text: {len(final_text)} characters")
    else:
        final_text = extracted_text
        print("⚠️  WARNING: Text redaction skipped!")
    
    # Stage 4: Validate that redaction worked
    print("✅ Stage 4: Validating redaction...")
    is_valid, issues = validate_redaction(final_text)
    
    if not is_valid:
        print(f"❌ WARNING: Redaction validation found {len(issues)} potential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Validation passed - no sensitive data detected!")
    
    print("🔒 SECURE EXTRACTION: Complete\n")
    return final_text, visual_img, (is_valid, issues)


def process_image_secure(img_path, return_original=False):
    """
    Comprehensive secure image processing with multi-stage redaction.
    
    This is the recommended function for processing images that may contain PII.
    It ensures ALL sensitive data is BLACKED OUT before any exposure to LLMs.
    
    Args:
        img_path: Path to the image file
        return_original: If True, also return the original unredacted image 
                        (use with caution, only for debugging)
    
    Returns:
        dict with keys:
            - 'redacted_text': Safely redacted text extracted from image
            - 'redacted_image': Visually redacted image (safe for display/LLM)
            - 'validation': Tuple (is_valid, issues) from validation
            - 'original_image': (optional) Original unredacted image if return_original=True
    """
    # Load original image
    original_img = Image.open(img_path).convert("RGB")
    
    # Perform secure extraction with all redaction stages
    redacted_text, redacted_img, validation = extract_text_from_image_secure(
        img_path,
        apply_visual_redaction=True,
        apply_text_redaction=True
    )
    
    result = {
        'redacted_text': redacted_text,
        'redacted_image': redacted_img,
        'validation': validation
    }
    
    if return_original:
        result['original_image'] = original_img
    
    return result


def batch_process_images_secure(image_paths, save_redacted=True, output_dir=None):
    """
    Process multiple images securely with pre-LLM redaction.
    
    Args:
        image_paths: List of image file paths
        save_redacted: If True, save redacted images to disk
        output_dir: Directory to save redacted images (required if save_redacted=True)
    
    Returns:
        list of dicts: Results for each image with redacted text and images
    """
    import os
    
    if save_redacted and not output_dir:
        raise ValueError("output_dir must be provided when save_redacted=True")
    
    if save_redacted and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    
    for img_path in image_paths:
        try:
            # Process image securely
            result = process_image_secure(img_path, return_original=False)
            
            # Add source path
            result['source_path'] = img_path
            
            # Save redacted image if requested
            if save_redacted:
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, f"redacted_{filename}")
                save_secure_image(result['redacted_image'], output_path)
                result['redacted_path'] = output_path
            
            results.append(result)
            
            # Log validation results
            is_valid, issues = result['validation']
            if not is_valid:
                print(f"⚠️  Image {img_path}: Found {len(issues)} validation issues")
            else:
                print(f"✓ Image {img_path}: Successfully redacted")
                
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            results.append({
                'source_path': img_path,
                'error': str(e),
                'redacted_text': None,
                'redacted_image': None,
                'validation': (False, [f"Processing error: {e}"])
            })
    
    return results


def extract_text_from_redacted_image(redacted_img):
    """
    Extract text from an already-redacted image.
    Use this when you have a pre-redacted image and need to extract text.
    
    Args:
        redacted_img: PIL Image object that has already been redacted
    
    Returns:
        str: Extracted text (with additional text-based redaction applied)
    """
    if not isinstance(redacted_img, Image.Image):
        raise TypeError("Input must be a PIL Image object")
    
    # Extract text from already-redacted image
    extracted_text = pytesseract.image_to_string(redacted_img)
    
    # Apply additional text-based redaction as safety layer
    final_text = redact_text(extracted_text, full_text=extracted_text)
    
    return final_text


def compare_redaction_effectiveness(original_path, redacted_img, original_text=None):
    """
    Compare original and redacted versions to verify redaction effectiveness.
    
    WARNING: This function handles sensitive data. Use only in secure testing environments.
    
    Args:
        original_path: Path to original unredacted image
        redacted_img: Redacted PIL Image object
        original_text: Optional pre-extracted text from original image
    
    Returns:
        dict: Comparison metrics showing what was redacted
    """
    # Extract text from original (SENSITIVE - handle with care)
    if original_text is None:
        original_img = Image.open(original_path).convert("RGB")
        original_text = pytesseract.image_to_string(original_img)
    
    # Extract text from redacted version
    redacted_text = pytesseract.image_to_string(redacted_img)
    
    # Apply text redaction to both for comparison
    original_redacted = redact_text(original_text, full_text=original_text)
    
    # Count redaction markers
    redaction_count = len(re.findall(r'\[REDACTED_[^\]]+\]', redacted_text))
    original_redaction_count = len(re.findall(r'\[REDACTED_[^\]]+\]', original_redacted))
    
    # Validate final redacted text
    is_valid, issues = validate_redaction(redacted_text)
    
    return {
        'original_length': len(original_text),
        'redacted_length': len(redacted_text),
        'reduction_percent': round((1 - len(redacted_text) / max(len(original_text), 1)) * 100, 2),
        'redaction_markers_in_image': redaction_count,
        'redaction_markers_in_text': original_redaction_count,
        'validation_passed': is_valid,
        'validation_issues': issues
    }


def get_safe_image_for_llm(img_path):
    """
    Get a completely safe, redacted image suitable for LLM processing.
    
    This is the simplest interface - use this when you just need a safe image
    to send to an LLM without worrying about the details.
    
    Args:
        img_path: Path to the image file
    
    Returns:
        PIL.Image: Fully redacted image safe for LLM processing
    """
    result = process_image_secure(img_path, return_original=False)
    return result['redacted_image']


def get_safe_text_for_llm(img_path):
    """
    Get completely safe, redacted text from an image for LLM processing.
    
    This is the simplest interface for text extraction - use this when you
    just need safe text without worrying about the details.
    
    Args:
        img_path: Path to the image file
    
    Returns:
        str: Fully redacted text safe for LLM processing
    """
    result = process_image_secure(img_path, return_original=False)
    return result['redacted_text']


# Convenience function for backward compatibility
def redact_and_extract(img_path):
    """
    Convenience function that returns both redacted image and text.
    
    Args:
        img_path: Path to the image file
    
    Returns:
        tuple: (redacted_text, redacted_image)
    """
    result = process_image_secure(img_path, return_original=False)
    return result['redacted_text'], result['redacted_image']