"""
AI-powered document analysis using Gemini
"""
import io
from PIL import Image
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, MAX_TEXT_LENGTH, IMAGE_TEXT_LENGTH
from utils.image_utils import create_clean_image

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def analyze_with_gemini(text, file_type, file_name):
    """Analyze document content using Gemini AI"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""Analyze the following content from a {file_type} file named "{file_name}".

Content:
{text}

Please provide a structured analysis in the following format:
1. File Description: A brief description of what this file contains (2-3 sentences)
2. Key Findings: List 3-4 bullet points of the most important findings or insights

Be concise and focus on the main purpose and key information."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Analysis failed: {e}"


def describe_image_with_gemini(img_path):
    """Analyze image content using Gemini AI - with secure clean image"""
    try:
        # Load the original image
        original_img = Image.open(img_path).convert("RGB")
        
        # Create a clean version without metadata
        clean_img = create_clean_image(original_img)
        
        # Save to temporary buffer
        img_byte_arr = io.BytesIO()
        clean_img.save(img_byte_arr, format='PNG', exif=b'')
        img_byte_arr.seek(0)
        img_bytes = img_byte_arr.read()
        
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = """Analyze this image and provide:
1. File Description: What is shown in the image? (2-3 sentences)
2. Key Findings: List 3-4 key observations or insights about what's in the image

Format your response clearly with these two sections."""

        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": img_bytes}
        ])
        return response.text
    except Exception as e:
        return f"Image analysis failed: {e}"


def parse_analysis_response(analysis):
    """Parse Gemini response into description and findings"""
    lines = analysis.split('\n')
    description_lines = []
    findings_lines = []
    in_description = False
    in_findings = False
    
    for line in lines:
        line = line.strip()
        if 'file description' in line.lower() or line.startswith('1.'):
            in_description = True
            in_findings = False
            if ':' in line:
                desc_text = line.split(':', 1)[1].strip()
                if desc_text:
                    description_lines.append(desc_text)
            continue
        elif 'key findings' in line.lower() or line.startswith('2.'):
            in_description = False
            in_findings = True
            continue
        
        if in_description and line:
            description_lines.append(line)
        elif in_findings and line:
            findings_lines.append(line)
    
    description = ' '.join(description_lines) if description_lines else "Analysis completed"
    findings = '\n'.join(findings_lines) if findings_lines else "See description"
    
    return description, findings