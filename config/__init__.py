"""
Configuration package for document redaction system
"""
from .settings import (
    GEMINI_API_KEY,
    TESSERACT_CMD,
    SPACY_MODEL,
    GEMINI_MODEL,
    MAX_TEXT_LENGTH,
    IMAGE_TEXT_LENGTH,
    OUTPUT_DIR_NAME,
    SUPPORTED_EXTENSIONS,
    SERVER_HOST,
    SERVER_PORT
)

__all__ = [
    'GEMINI_API_KEY',
    'TESSERACT_CMD',
    'SPACY_MODEL',
    'GEMINI_MODEL',
    'MAX_TEXT_LENGTH',
    'IMAGE_TEXT_LENGTH',
    'OUTPUT_DIR_NAME',
    'SUPPORTED_EXTENSIONS',
    'SERVER_HOST',
    'SERVER_PORT'
]