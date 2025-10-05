"""
Image processing and manipulation utilities
"""
import io
import re
import numpy as np
from PIL import Image, ImageFilter
import pytesseract
from facenet_pytorch import MTCNN
from .patterns import SENSITIVE_PATTERNS
from .text_utils import nlp, NER_LABELS
from config.settings import (
    TESSERACT_CMD, FACE_BLUR_RADIUS, GAUSSIAN_BLUR_SMALL_IMAGE,
    SMALL_IMAGE_THRESHOLD_WIDTH, SMALL_IMAGE_THRESHOLD_HEIGHT,
    TEXT_PADDING, PADDING_SIZE
)

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
    """Redact sensitive text in images with enhanced multi-word pattern matching"""
    if not isinstance(img, Image.Image):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            raise TypeError(f"Expected PIL Image or numpy array, got {type(img)}")
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    full_text = pytesseract.image_to_string(img)
    is_aws_json = bool(re.search(
        r'"(?:RoleId|Arn|AWS|AccountId|Sid|Effect|Principal|Action)"', 
        full_text, 
        re.IGNORECASE
    ))
    
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Group words into blocks
    blocks = {}
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            block_num = data['block_num'][i]
            if block_num not in blocks:
                blocks[block_num] = []
            blocks[block_num].append(i)
    
    redacted_regions = []
    
    # AWS JSON special handling
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
                        region = img.crop((x, y, x + w, y + h))
                        region = region.filter(ImageFilter.GaussianBlur(30))
                        img.paste(region, (x, y))
                        redacted_regions.append((x, y, w, h))
    
    # Pattern-based redaction
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
                            region = img.crop((x, y, x + w, y + h))
                            region = region.filter(ImageFilter.GaussianBlur(30))
                            img.paste(region, (x, y))
                            redacted_regions.append((x, y, w, h))
            except Exception:
                continue
    
    # NER-based redaction
    for block_num, word_indices in blocks.items():
        block_text = " ".join([data['text'][i] for i in word_indices])
        
        try:
            doc = nlp(block_text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"] and len(ent.text) > 2:
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
                                region = img.crop((x, y, x + w, y + h))
                                region = region.filter(ImageFilter.GaussianBlur(30))
                                img.paste(region, (x, y))
                                redacted_regions.append((x, y, w, h))
        except Exception:
            continue
    
    return img


def extract_text_from_image(path):
    """Extract text from image using OCR"""
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text, img