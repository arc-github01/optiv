"""
Configuration settings for the document redaction system
"""
import os

# API Configuration
GEMINI_API_KEY = "AIzaSyDt7DD7ySYBoqiWor9kend7wsbq1xKTviA" 

# Tesseract Configuration
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Model Configuration
SPACY_MODEL = "en_core_web_sm"
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Processing Configuration
MAX_TEXT_LENGTH = 30000
IMAGE_TEXT_LENGTH = 5000
BLUR_RADIUS = 30
GAUSSIAN_BLUR_SMALL_IMAGE = 20
FACE_BLUR_RADIUS = 30

# File Configuration
OUTPUT_DIR_NAME = "redacted_output"
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls', '.pptx', '.ppt', '.jpg', '.jpeg', '.png']

# Image Processing
SMALL_IMAGE_THRESHOLD_WIDTH = 500
SMALL_IMAGE_THRESHOLD_HEIGHT = 500
PADDING_SIZE = 10
TEXT_PADDING = 8

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860