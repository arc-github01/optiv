"""
Configuration for redaction settings
Allows fine-grained control over what gets redacted
"""

# ============================================
# REDACTION SETTINGS - CUSTOMIZE AS NEEDED
# ============================================

REDACTION_CONFIG = {
    # High-confidence PII that should ALWAYS be redacted
    'ALWAYS_REDACT': {
        'email': True,           # john@example.com
        'phone': True,           # +1-555-123-4567
        'ssn': True,             # 123-45-6789
        'credit_card': True,     # 4532-1234-5678-9010
        'passport': True,        # A12345678
        'drivers_license': True, # DL12345678
        'account_number': True,  # Bank accounts, etc.
        'ip_address': True,      # 192.168.1.1
        'credentials': True,     # password=secret123
        'pan': True,             # ABCDE1234F (Indian PAN)
        'aadhaar': True,         # 1234 5678 9012 (Indian Aadhaar)
    },
    
    # Address information
    'ADDRESS': {
        'street_address': True,  # 123 Main Street
        'po_box': True,          # P.O. Box 456
        'zip_code': True,       # 12345 (often not sensitive)
    },
    
    # Dates - BE CAREFUL: Hard to distinguish DOB from regular dates
    'DATES': {
        'all_dates': True,      # ❌ DON'T enable - will redact ALL dates
        'age_indicators': True,  # "age 35", "35 years old"
        'dob_keywords': True,    # Only redact dates near keywords like "born", "DOB:"
    },
    
    # Names - Most controversial category
    'NAMES': {
        'person_names': True,        # John Doe (uses NER with validation)
        'require_context': True,     # Only redact names in PII context
        'min_confidence': 'medium',    # 'low', 'medium', 'high'
        'whitelist_roles': True,     # Don't redact titles like "Chief Officer"
    },
    
    # Organization names
    'ORGANIZATIONS': {
        'company_names': False,      # ❌ Usually NOT sensitive (GILAC, Microsoft, etc.)
        'internal_orgs': False,      # Department names, divisions
    },
    
    # Location information
    'LOCATIONS': {
        'cities': False,         # New York, London (usually not sensitive)
        'states': False,         # California, Texas
        'countries': False,      # United States, India
        'specific_locations': True,  # GPS coordinates, landmarks
    },
    
    # Financial information
    'FINANCIAL': {
        'bank_names': False,     # Wells Fargo, HDFC (not sensitive)
        'amounts': True,        # $1,234.56 (usually not sensitive in policy docs)
        'account_numbers': True, # Actual account numbers
    },
    
    # Other data
    'OTHER': {
        'medical_info': True,    # Medical record numbers, health data
        'license_plates': True,  # ABC-1234
        'custom_ids': True,      # Employee IDs if pattern matches
    }
}


# Keywords that indicate a date is likely a DOB (only used if dob_keywords=True)
DOB_CONTEXT_KEYWORDS = [
    'born', 'birth', 'dob', 'date of birth', 'birthdate',
    'born on', 'birthday', 'age', 'years old'
]

# Role titles that should NOT be redacted as person names
BUSINESS_ROLES = {
    'chief', 'officer', 'director', 'manager', 'executive', 'president',
    'vice', 'senior', 'junior', 'assistant', 'head', 'lead',
    'coordinator', 'administrator', 'specialist', 'analyst', 'engineer',
    'developer', 'consultant', 'supervisor', 'superintendent'
}

# Business terms that indicate non-PII content
BUSINESS_TERMS = {
    'department', 'company', 'corporation', 'organization', 'division',
    'team', 'group', 'committee', 'board', 'council', 'inc', 'ltd', 'llc',
    'management', 'information', 'security', 'policy', 'compliance',
    'governance', 'risk', 'asset', 'change', 'access', 'control',
    'audit', 'review', 'approval', 'implementation', 'maintenance'
}


def should_redact(category, subcategory=None):
    """
    Check if a category/subcategory should be redacted based on config
    
    Args:
        category: Main category (e.g., 'ALWAYS_REDACT', 'NAMES')
        subcategory: Specific type (e.g., 'email', 'person_names')
    
    Returns:
        bool: True if should be redacted
    """
    if category not in REDACTION_CONFIG:
        return False
    
    if subcategory:
        return REDACTION_CONFIG[category].get(subcategory, False)
    
    # If no subcategory, check if any in category is True
    return any(REDACTION_CONFIG[category].values())


def get_name_confidence_threshold():
    """Get the minimum confidence level for name redaction"""
    confidence = REDACTION_CONFIG['NAMES'].get('min_confidence', 'high')
    
    thresholds = {
        'low': 0.5,    # Redact most names (more false positives)
        'medium': 0.7, # Balanced approach
        'high': 0.85   # Only very confident matches (fewer false positives)
    }
    
    return thresholds.get(confidence, 0.85)


def is_dob_context(text, date_position):
    """Check if a date appears in a DOB context"""
    if not REDACTION_CONFIG['DATES'].get('dob_keywords', False):
        return False
    
    # Check text around the date (50 chars before and after)
    start = max(0, date_position - 50)
    end = min(len(text), date_position + 50)
    context = text[start:end].lower()
    
    # Check if any DOB keyword appears near the date
    return any(keyword in context for keyword in DOB_CONTEXT_KEYWORDS)