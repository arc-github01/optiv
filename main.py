import os
import io
import re
import fitz
import docx
from PIL import Image, ImageFilter
import pytesseract
import spacy
import torch
from facenet_pytorch import MTCNN
from transformers import ViTFeatureExtractor, ViTForImageClassification
import gradio as gr
import google.generativeai as genai

# =======================
# Setup
# =======================
# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- SpaCy model ---
nlp = spacy.load("en_core_web_sm")

# --- ViT Model ---
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# --- Face Detector ---
mtcnn = MTCNN(keep_all=True, device="cpu")

# --- Gemini API ---
genai.configure(api_key="YOUR-API-KEY")  # <-- put your Gemini API key here

# =======================
# Helper functions
# =======================
def extract_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_from_docx(path):
    docx_file = docx.Document(path)
    return "\n".join([p.text for p in docx_file.paragraphs])

def extract_from_image(path):
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text, img

def redact_text(text):
    patterns = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"\b\d{10}\b",
        "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "DOB": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        "VEHICLE": r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}"
    }
    for label, pat in patterns.items():
        text = re.sub(pat, f"[REDACTED_{label}]", text)

    doc = nlp(text)
    redacted_positions = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            overlap = any(not (ent.start_char >= end or ent.end_char <= start) for start, end in redacted_positions)
            if not overlap:
                text = text[:ent.start_char] + f"[REDACTED_{ent.label_}]" + text[ent.end_char:]
                new_end = ent.start_char + len(f"[REDACTED_{ent.label_}]")
                redacted_positions.append((ent.start_char, new_end))
    return text

def redact_image_text(img):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if not text:
            continue

        # Sensitive patterns
        patterns = [
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",  # email
            r"\b\d{10}\b",                                      # phone
            r"\b\d{4}\s?\d{4}\s?\d{4}\b",                       # Aadhaar
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                       # PAN
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",              # DOB
            r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}",  # Vehicle plate
            r"\b\d{4}\b",
        ]

        sensitive = False
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                sensitive = True
                break

        if re.search(r'\b\d{3,}\b', text):
            sensitive = True

        if not sensitive:
            try:
                doc = nlp(text)
                sensitive = any(ent.label_ in ["PERSON", "GPE", "ORG"] for ent in doc.ents)
            except:
                if re.search(r'[A-Za-z]{2,}', text):
                    sensitive = True

        if sensitive:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.width - x, w + 2 * padding)
            h = min(img.height - y, h + 2 * padding)
            region = img.crop((x, y, x + w, y + h))
            region = region.filter(ImageFilter.GaussianBlur(25))
            img.paste(region, (x, y))
    
    return img

def blur_faces(img):
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img.crop((x1, y1, x2, y2))
            face = face.filter(ImageFilter.GaussianBlur(30))
            img.paste(face, (x1, y1))
    return img

def classify_image(img):
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = vit_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()
    return vit_model.config.id2label[pred]

# =======================
# Gemini Summarization / Image description
# =======================
def summarize_with_gemini(text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(f"Summarize the following text:\n\n{text}")
        return response.text
    except Exception as e:
        return f"Summarization failed: {e}"

# NEW: ask Gemini to describe an image if no text
def describe_image_with_gemini(img_path):
    model = genai.GenerativeModel("gemini-2.5-flash")
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    response = model.generate_content(
        [
            "Describe the following image in one sentence.",
            {"mime_type": "image/png", "data": img_bytes}
        ]
    )
    return response.text

# =======================
# Gradio processing
# =======================
def process_file(file):
    fname = file.name
    ext = fname.split(".")[-1].lower()

    if ext == "pdf":
        raw_text = extract_from_pdf(fname)
        redacted_text = redact_text(raw_text)
        summary = summarize_with_gemini(redacted_text)
        return redacted_text, summary, None

    elif ext == "docx":
        raw_text = extract_from_docx(fname)
        redacted_text = redact_text(raw_text)
        summary = summarize_with_gemini(redacted_text)
        return redacted_text, summary, None

    elif ext in ["jpg", "jpeg", "png"]:
        ocr_text, img = extract_from_image(fname)
        img = redact_image_text(img)
        img = blur_faces(img)
        redacted_text = redact_text(ocr_text)

        if len(ocr_text.strip()) < 5:  # NEW: fallback to Gemini vision
            # Try Gemini image description
            description = describe_image_with_gemini(fname)
            redacted_text = f"{description}\nSensitive content blurred."
            summary = ""  # No extra summary needed
        else:
            summary = summarize_with_gemini(redacted_text)

        return redacted_text, summary, img

    else:
        return "Unsupported file type.", "", None

# =======================
# Gradio interface
# =======================
iface = gr.Interface(
    fn=process_file,
    inputs=gr.File(file_types=[".pdf", ".docx", ".jpg", ".jpeg", ".png"]),
    outputs=[
        gr.Textbox(label="Redacted Text"),
        gr.Textbox(label="Summary"),
        gr.Image(label="Processed Image", type="pil")
    ],
    title="Document & Image Redactor + Summarizer",
    description="Upload a PDF, DOCX, or image. Redacts sensitive text, blurs faces in images, and summarizes or describes the remaining content using Gemini."
)

iface.launch()
