"""
Text processing and redaction utilities with configurable settings
Enhanced with pre-LLM redaction security
"""
import re
import spacy
from .patterns import SENSITIVE_PATTERNS, PRIORITY_PATTERNS, NER_LABELS
from .redaction_config import (
    REDACTION_CONFIG, DOB_CONTEXT_KEYWORDS, BUSINESS_ROLES, 
    BUSINESS_TERMS, should_redact, is_dob_context, get_name_confidence_threshold
)
from config.settings import SPACY_MODEL

# Load spaCy model
nlp = spacy.load(SPACY_MODEL)


def contains_pii_pattern(text):
    """Check if text matches specific PII patterns based on config"""
    if not text:
        return False
    
    patterns_to_check = []
    
    # Email
    if should_redact('ALWAYS_REDACT', 'email'):
        patterns_to_check.append(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b')
    
    # Phone
    if should_redact('ALWAYS_REDACT', 'phone'):
        patterns_to_check.extend([
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\b\+?91[-.\s]?\d{10}\b',  # Indian phone numbers
        ])
    
    # SSN
    if should_redact('ALWAYS_REDACT', 'ssn'):
        patterns_to_check.append(r'\b\d{3}-\d{2}-\d{4}\b')
    
    # PAN (Indian Permanent Account Number)
    # Format: ABCDE1234F (5 letters, 4 digits, 1 letter)
    if should_redact('ALWAYS_REDACT', 'pan'):
        patterns_to_check.extend([
            r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
            r'\b[A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]\b',  # More strict PAN format
        ])
    
    # Aadhaar (Indian identification number)
    # Format: 1234 5678 9012 or 123456789012 (12 digits)
    if should_redact('ALWAYS_REDACT', 'aadhaar'):
        patterns_to_check.extend([
            r'\b\d{4}\s\d{4}\s\d{4}\b',  # With spaces
            r'\b\d{12}\b',                # Without spaces (but be careful with false positives)
        ])
    
    # Credit Card
    if should_redact('ALWAYS_REDACT', 'credit_card'):
        patterns_to_check.append(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    
    # Address
    if should_redact('ADDRESS', 'street_address'):
        patterns_to_check.append(
            r'\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|lane|ln|drive|dr|way|court|ct|boulevard|blvd|place|pl)\b'
        )
    
    if should_redact('ADDRESS', 'po_box'):
        patterns_to_check.append(r'\bP\.?O\.?\s+Box\s+\d+\b')
    
    # IP Address
    if should_redact('ALWAYS_REDACT', 'ip_address'):
        patterns_to_check.append(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    
    # Credentials
    if should_redact('ALWAYS_REDACT', 'credentials'):
        patterns_to_check.append(r'\b(?:password|passwd|pwd|secret|api[_-]?key|token)\s*[:=]\s*\S+')
    
    # Passport
    if should_redact('ALWAYS_REDACT', 'passport'):
        patterns_to_check.extend([
            r'\b[A-Z]\d{8}\b',           # US format
            r'\b[A-Z]{1,2}\d{7}\b',      # Indian format
        ])
    
    # Check all applicable patterns
    for pattern in patterns_to_check:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def is_person_name(text, doc=None, full_text=None, position=0):
    """
    Determine if text is a person's name with confidence scoring.
    
    Args:
        text: The potential name text
        doc: spaCy doc object (optional)
        full_text: The full text context (optional, for context analysis)
        position: Position of the name in full_text (optional)
    """
    if not text or len(text) < 3:
        return False
    
    # Check config
    if not should_redact('NAMES', 'person_names'):
        return False
    
    text = text.strip()
    words = text.split()
    
    # Must have at least 2 words for full name
    if len(words) < 2:
        return False
    
    # All words must be capitalized
    if not all(word[0].isupper() for word in words if word):
        return False
    
    # Check if it's a business role/title
    if should_redact('NAMES', 'whitelist_roles'):
        if any(word.lower() in BUSINESS_ROLES for word in words):
            return False
        if any(word.lower() in BUSINESS_TERMS for word in words):
            return False
    
    # If require_context is enabled, check surrounding text
    if should_redact('NAMES', 'require_context') and full_text and position > 0:
        # Look for PII context keywords near the name
        context_start = max(0, position - 100)
        context_end = min(len(full_text), position + len(text) + 100)
        context = full_text[context_start:context_end].lower()
        
        # PII context indicators
        pii_indicators = [
            'employee', 'staff', 'patient', 'customer', 'client', 'resident',
            'member', 'contact', 'signature', 'signed by', 'submitted by',
            'name:', 'from:', 'to:', 'cc:', 'attn:', 'attention:'
        ]
        
        has_context = any(indicator in context for indicator in pii_indicators)
        if not has_context:
            return False  # No PII context, likely just a business reference
    
    # Use NER confidence scoring
    if doc:
        for ent in doc.ents:
            if ent.text == text and ent.label_ == 'PERSON':
                # Check confidence threshold
                # Note: spaCy doesn't provide confidence scores directly,
                # so we use heuristics
                confidence_score = 0.9  # High confidence if NER detects it
                threshold = get_name_confidence_threshold()
                return confidence_score >= threshold
    
    return True  # Passed all checks


def contains_date_pattern(text, full_text=None, position=0):
    """Check if text contains dates based on config"""
    if not text:
        return False
    
    # If all_dates is enabled, redact all dates
    if should_redact('DATES', 'all_dates'):
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        ]
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
    
    # If dob_keywords enabled, only redact dates in DOB context
    elif should_redact('DATES', 'dob_keywords') and full_text:
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and is_dob_context(full_text, position + match.start()):
                return True
    
    # Check for age indicators
    if should_redact('DATES', 'age_indicators'):
        age_pattern = r'\b(?:age|aged)\s+\d{1,3}\b|\b\d{1,3}\s+years?\s+old\b'
        if re.search(age_pattern, text, re.IGNORECASE):
            return True
    
    return False


def is_sensitive_text(text, full_text=None):
    """
    Detect if text contains sensitive information based on configuration.
    
    Args:
        text: The text to check
        full_text: Optional full document context for better decisions
    """
    if not text or len(str(text)) < 2:
        return False
    
    text_str = str(text).strip()
    
    # Very short text is likely not sensitive
    if len(text_str) < 3:
        return False
    
    # Check high-confidence PII patterns
    if contains_pii_pattern(text_str):
        return True
    
    # Check dates if configured
    position = full_text.find(text_str) if full_text else 0
    if contains_date_pattern(text_str, full_text, position):
        return True
    
    # Check regex patterns from config (filter by categories)
    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        try:
            # Map pattern names to config categories
            if 'email' in pattern_name.lower() and should_redact('ALWAYS_REDACT', 'email'):
                if re.search(pattern, text_str, re.IGNORECASE):
                    return True
            elif 'phone' in pattern_name.lower() and should_redact('ALWAYS_REDACT', 'phone'):
                if re.search(pattern, text_str, re.IGNORECASE):
                    return True
            elif 'pan' in pattern_name.lower() and should_redact('ALWAYS_REDACT', 'pan'):
                if re.search(pattern, text_str, re.IGNORECASE):
                    return True
            elif 'aadhaar' in pattern_name.lower() and should_redact('ALWAYS_REDACT', 'aadhaar'):
                if re.search(pattern, text_str, re.IGNORECASE):
                    return True
        except:
            pass
    
    # Check for person names using NER
    if should_redact('NAMES', 'person_names') and len(text_str) > 5:
        try:
            doc = nlp(text_str)
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    if is_person_name(ent.text, doc, full_text, position):
                        return True
        except:
            pass
    
    return False


def redact_text(text, full_text=None):
    """
    Redact sensitive information from text based on configuration.
    
    Args:
        text: Text to redact
        full_text: Optional full document context
    """
    if not text:
        return text
    
    original_text = text
    
    # Apply PII patterns based on config
    for label in PRIORITY_PATTERNS:
        if label in SENSITIVE_PATTERNS:
            # Check if this pattern category is enabled
            should_apply = False
            if 'email' in label.lower() and should_redact('ALWAYS_REDACT', 'email'):
                should_apply = True
            elif 'phone' in label.lower() and should_redact('ALWAYS_REDACT', 'phone'):
                should_apply = True
            elif 'ssn' in label.lower() and should_redact('ALWAYS_REDACT', 'ssn'):
                should_apply = True
            elif 'credit' in label.lower() and should_redact('ALWAYS_REDACT', 'credit_card'):
                should_apply = True
            elif 'address' in label.lower() and should_redact('ADDRESS', 'street_address'):
                should_apply = True
            elif 'ip' in label.lower() and should_redact('ALWAYS_REDACT', 'ip_address'):
                should_apply = True
            elif 'pan' in label.lower() and should_redact('ALWAYS_REDACT', 'pan'):
                should_apply = True
            elif 'aadhaar' in label.lower() and should_redact('ALWAYS_REDACT', 'aadhaar'):
                should_apply = True
            
            if should_apply:
                text = re.sub(
                    SENSITIVE_PATTERNS[label], 
                    f"[REDACTED_{label}]", 
                    text, 
                    flags=re.IGNORECASE
                )
    
    # Apply direct PAN and Aadhaar redaction (in case they're not in SENSITIVE_PATTERNS)
    if should_redact('ALWAYS_REDACT', 'pan'):
        # PAN format: ABCDE1234F
        text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[REDACTED_PAN]', text)
    
    if should_redact('ALWAYS_REDACT', 'aadhaar'):
        # Aadhaar with spaces
        text = re.sub(r'\b\d{4}\s\d{4}\s\d{4}\b', '[REDACTED_AADHAAR]', text)
        # Aadhaar without spaces (be careful - only if surrounded by word boundaries)
        # Add context checking to avoid false positives
        
        matches = list(re.finditer(r'\b\d{12}\b', text))
        # Process in reverse to maintain string indices
        for match in reversed(matches):
            num = match.group()
            # Basic Aadhaar validation: first digit should not be 0 or 1
            if num[0] not in ['0', '1']:
                text = text[:match.start()] + '[REDACTED_AADHAAR]' + text[match.end():]
    
    # Apply NER-based redaction for person names
    if should_redact('NAMES', 'person_names'):
        try:
            doc = nlp(text)
            redacted_positions = []
            position = full_text.find(text) if full_text else 0
            
            # Process entities in reverse order to maintain string indices
            entities = [(ent.start_char, ent.end_char, ent.text, ent.label_) for ent in doc.ents]
            entities.sort(reverse=True, key=lambda x: x[0])
            
            for start_char, end_char, ent_text, label in entities:
                if label == 'PERSON':
                    if is_person_name(ent_text, doc, full_text, position):
                        # Check for overlap
                        overlap = any(
                            not (start_char >= end or end_char <= start) 
                            for start, end in redacted_positions
                        )
                        
                        if not overlap:
                            text = text[:start_char] + f"[REDACTED_PERSON]" + text[end_char:]
                            new_end = start_char + len(f"[REDACTED_PERSON]")
                            redacted_positions.append((start_char, new_end))
        except Exception as e:
            print(f"SpaCy NER error: {e}")
    
    return text


def validate_redaction(text):
    """
    Validate that sensitive data was properly redacted.
    Returns True if validation passes (no sensitive data found), False otherwise.
    
    Args:
        text: The redacted text to validate
    
    Returns:
        tuple: (is_valid, list of issues found)
    """
    if not text:
        return True, []
    
    issues = []
    
    # Check for common PII patterns that should have been redacted
    validation_patterns = {
        'Aadhaar': r'\b\d{4}\s\d{4}\s\d{4}\b',
        'PAN': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
        'Email': r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b',
        'Phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'Credit Card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
    }
    
    for pii_type, pattern in validation_patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            issues.append(f"{pii_type} found: {match.group()}")
    
    is_valid = len(issues) == 0
    return is_valid, issues