# ============================================
# core/__init__.py
# ============================================
"""
Core document processing modules
"""
from .extractors import (
    extract_from_pdf,
    extract_from_docx,
    extract_from_excel,
    extract_from_ppt
)

from .redactors import (
    redact_pdf,
    redact_docx,
    redact_excel,
    redact_ppt
)

from .analyzers import (
    analyze_with_gemini,
    describe_image_with_gemini,
    parse_analysis_response
)

__all__ = [
    # Extractors
    'extract_from_pdf',
    'extract_from_docx',
    'extract_from_excel',
    'extract_from_ppt',
    
    # Redactors
    'redact_pdf',
    'redact_docx',
    'redact_excel',
    'redact_ppt',
    
    # Analyzers
    'analyze_with_gemini',
    'describe_image_with_gemini',
    'parse_analysis_response'
]
