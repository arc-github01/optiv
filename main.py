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

# =======================
# Setup
# =======================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
nlp = spacy.load("en_core_web_sm")

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
mtcnn = MTCNN(keep_all=True, device="cpu")

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
    """
    Blur sensitive text in images including Aadhaar, PAN, DOB, emails, phone, names, orgs, GPE, and vehicle plates
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if not text:
            continue

        # Normalize text (remove spaces for numbers)
        text_norm = re.sub(r"\D", "", text)  # remove non-digits
        patterns = [
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",  # email
            r"\b\d{10}\b",                                      # phone
            r"\b\d{12}\b",                                      # Aadhaar 12 digits
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                       # PAN
            r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",                     # DOB
            r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}"    # Vehicle plate
        ]

        sensitive = any(re.fullmatch(pat.replace(" ", ""), text_norm) for pat in patterns)
        
        # spaCy NER for names, orgs, GPE
        doc = nlp(text)
        sensitive |= any(ent.label_ in ["PERSON", "GPE", "ORG"] for ent in doc.ents)

        if sensitive:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            region = img.crop((x, y, x+w, y+h))
            region = region.filter(ImageFilter.GaussianBlur(20))
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
# Gradio processing
# =======================
def process_file(file):
    fname = file.name
    ext = fname.split(".")[-1].lower()
    
    if ext == "pdf":
        text = extract_from_pdf(fname)
        return redact_text(text), None
    elif ext == "docx":
        text = extract_from_docx(fname)
        return redact_text(text), None
    elif ext in ["jpg", "jpeg", "png"]:
        ocr_text, img = extract_from_image(fname)
        img = redact_image_text(img)      # blur sensitive text
        img = blur_faces(img)             # blur faces
        text_output = redact_text(ocr_text)
        if len(ocr_text.strip()) < 5:
            label = classify_image(img)
            text_output = f"Image classified as {label}. Sensitive content blurred."
        return text_output, img
    else:
        return "Unsupported file type.", None

# =======================
# Gradio interface
# =======================
iface = gr.Interface(
    fn=process_file,
    inputs=gr.File(file_types=[".pdf", ".docx", ".jpg", ".jpeg", ".png"]),
    outputs=[gr.Textbox(label="Redacted Text"), gr.Image(label="Processed Image", type="pil")],
    title="Document & Image Redactor",
    description="Upload a PDF, DOCX, or image. Redacts sensitive text and blurs faces in images."
)

iface.launch()
