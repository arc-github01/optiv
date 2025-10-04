# Enhanced Batch File Analyzer & Redactor

[![Standard Readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Supported File Formats](#supported-file-formats)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project
The Enhanced Batch File Analyzer & Redactor is a powerful tool to automatically detect and redact sensitive information across various document and image formats. It incorporates advanced pattern matching, AI-powered content analysis, and image processing to protect personally identifiable information (PII), credentials, addresses, and more. The tool features a user-friendly Gradio web interface for batch processing directories of files, making it ideal for privacy-focused data handling and compliance workflows.

## Features
- Automatic detection and redaction of PII including emails, phone numbers, government IDs, financial details, addresses, and names.
- Redaction methods include black-box censoring and NLP-based masking.
- Image processing to detect and blur faces, QR codes, and barcodes.
- AI-driven analysis and summarization of document contents using Google Gemini.
- Batch redaction and analysis for whole directories of files.
- Intuitive browser-based UI via Gradio.
- Saves redacted documents in a structured output folder.

## Supported File Formats
| Document Type | File Extensions             |
| ------------- | -------------------------- |
| PDF           | `.pdf`                     |
| Word          | `.docx`                    |
| Excel         | `.xlsx`, `.xls`            |
| PowerPoint    | `.pptx`, `.ppt`            |
| Images        | `.jpg`, `.jpeg`, `.png`    |

## Installation
1. Clone this repository:
git clone https://github.com/arc-github01/optiv.git
cd your-repo

2. Install dependencies:
pip install -r requirements.txt

3. (Windows) Install Tesseract OCR and configure its executable path if not already set.
4. Set your Google Gemini AI API key where indicated in the code.

## Usage
1. Run the main application:
2. Open the local URL displayed by the Gradio interface in your browser.
3. Enter the directory path containing files you want to analyze/redact, or upload individual files.
4. Click **Analyze / Redact All Files**.
5. View the analysis results, download redacted files from the UI.

## Contributing
Contributions are welcome! Please follow these steps:
- Fork the repository
- Create a new branch (`git checkout -b feature-branch`)
- Make your changes
- Test thoroughly
- Submit a pull request describing your enhancements or fixes

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions, issues, or suggestions, reach out via your GitHub profile or open an issue in this repository.

---

*This README follows a standard, clear GitHub-friendly structure to make the project accessible and engaging.* 
