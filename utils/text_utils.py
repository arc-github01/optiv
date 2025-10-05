"""
Text processing and redaction utilities
"""
import re
import spacy
from .patterns import SENSITIVE_PATTERNS, PRIORITY_PATTERNS, NER_LABELS
from config.settings import SPACY_MODEL

# Load spaCy model
nlp = spacy.load(SPACY_MODEL)


def is_sensitive_text(text):
    if not text or len(str(text)) < 2:
        return False
    
    text_str = str(text)
    
    # Check against regex patterns
    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        try:
            if re.search(pattern, text_str, re.IGNORECASE):
                return True
        except:
            pass
    
    # Check using NER
    try:
        if len(text_str) > 2 and not text_str.isdigit():
            doc = nlp(text_str)
            if any(ent.label_ in NER_LABELS for ent in doc.ents):
                return True
    except:
        pass
    
    return False


def redact_text(text):
    """Redact sensitive information from text"""
    if not text:
        return text
    
    # Apply priority patterns first
    for label in PRIORITY_PATTERNS:
        if label in SENSITIVE_PATTERNS:
            text = re.sub(
                SENSITIVE_PATTERNS[label], 
                f"[REDACTED_{label}]", 
                text, 
                flags=re.IGNORECASE
            )
    
    # Apply remaining patterns
    for label, pattern in SENSITIVE_PATTERNS.items():
        if label not in PRIORITY_PATTERNS:
            text = re.sub(
                pattern, 
                f"[REDACTED_{label}]", 
                text, 
                flags=re.IGNORECASE
            )
    
    # Apply NER-based redaction
    try:
        doc = nlp(text)
        redacted_positions = []
        for ent in doc.ents:
            if ent.label_ in NER_LABELS:
                overlap = any(
                    not (ent.start_char >= end or ent.end_char <= start) 
                    for start, end in redacted_positions
                )
                if not overlap and len(ent.text) > 2:
                    text = text[:ent.start_char] + f"[REDACTED_{ent.label_}]" + text[ent.end_char:]
                    new_end = ent.start_char + len(f"[REDACTED_{ent.label_}]")
                    redacted_positions.append((ent.start_char, new_end))
    except Exception as e:
        print(f"SpaCy NER error: {e}")
    
    return text