"""
Directory-level file processing
"""
import os
from pathlib import Path
import pandas as pd
from .file_processor import process_single_file
from config.settings import SUPPORTED_EXTENSIONS, OUTPUT_DIR_NAME


def process_directory(directory_path):
    """Process all supported files in a directory"""
    if not directory_path or not os.path.exists(directory_path):
        return "Invalid directory path", None, None
    
    path_obj = Path(directory_path)
    
    # Collect all supported files
    file_set = set()
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in path_obj.glob(f'*{ext}'):
            file_set.add(file_path)
        for file_path in path_obj.glob(f'*{ext.upper()}'):
            file_set.add(file_path)
    
    files = sorted(file_set)
    
    if not files:
        return "No supported files found in directory", None, None
    
    results = []
    processed_images = []
    redacted_files_list = []
    
    for file_path in files:
        print(f"Processing: {file_path}")
        result = process_single_file(file_path)
        results.append(result)
        if result["Processed Image"] is not None:
            processed_images.append((result["File Name"], result["Processed Image"]))
        if result["Redacted File"] is not None:
            redacted_files_list.append(result["Redacted File"])
    
    df = pd.DataFrame(results)
    df = df[["File Name", "File Type", "File Description", "Key Findings"]]
    
    html_output = generate_html_report(df)
    
    image_gallery = None
    if processed_images:
        image_gallery = [img for _, img in processed_images]
    
    download_info = f"Redacted files saved in: {os.path.join(directory_path, OUTPUT_DIR_NAME)}" if redacted_files_list else "No files required redaction"
    
    return html_output, image_gallery, download_info


def generate_html_report(df):
    """Generate styled HTML report from DataFrame"""
    html_output = """
    <style>
        .analysis-table {
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin-top: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border-radius: 10px;
            overflow: hidden;
        }
        .analysis-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 15px;
            text-align: left;
            border: none;
            font-size: 1.1em;
        }
        .analysis-table td {
            padding: 15px;
            border: none;
            vertical-align: top;
            line-height: 1.4;
            color: #e0e0e0;
        }
        .analysis-table tr:nth-child(even) {
            background-color: #2d3748;
        }
        .analysis-table tr:nth-child(odd) {
            background-color: #1a202c;
        }
        .analysis-table tr:hover {
            background-color: #4a5568;
            transition: background-color 0.3s ease;
        }
        
        .file-type {
            font-weight: bold;
            color: #63b3ed;
        }
        .key-findings {
            white-space: pre-line;
            color: #cbd5e0;
        }
        .key-findings::before {
            content: "• ";
            color: #68d391;
        }
    </style>
    
    <table class="analysis-table">
        <thead>
            <tr>
                <th>File Name</th>
                <th>File Type</th>
                <th>File Description</th>
                <th>Key Findings</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df.iterrows():
        findings_formatted = row['Key Findings'].replace('\n', '\n• ')
        if findings_formatted and not findings_formatted.startswith('•'):
            findings_formatted = '• ' + findings_formatted
        
        html_output += f"""
            <tr>
                <td style="color: #ffffff; font-weight: 600;">{row['File Name']}</td>
                <td class="file-type">{row['File Type']}</td>
                <td style="color: #e2e8f0;">{row['File Description']}</td>
                <td class="key-findings">{findings_formatted}</td>
            </tr>
        """
    
    html_output += """
        </tbody>
    </table>
    """
    
    return html_output