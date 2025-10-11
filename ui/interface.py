"""
Gradio UI interface configuration with file upload
"""
import gradio as gr
from .styles import CUSTOM_CSS
from processors.file_processor import process_single_file
import os


def process_uploaded_file(file):
    """Process a single uploaded file and return results with download link"""
    if file is None:
        return None, "<div style='color: #f72585; padding: 20px; text-align: center;'>⚠️ Please upload a file first</div>", None
    
    try:
        # Process the file
        result = process_single_file(file.name)
        
        # Create HTML report
        html_output = f"""
        <div style='background: #1a1a2e; padding: 30px; border-radius: 15px; border: 1px solid #2d3748;'>
            <div style='background: linear-gradient(135deg, #4361ee, #3a0ca3); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: white; margin: 0;'>📄 {result['File Name']}</h2>
                <p style='color: #cbd5e0; margin: 10px 0 0 0;'>Type: {result['File Type']}</p>
            </div>
            
            <div style='background: #2d3748; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #4cc9f0;'>
                <h3 style='color: white; margin-top: 0;'>📝 File Description</h3>
                <p style='color: #e2e8f0; line-height: 1.6;'>{result['File Description']}</p>
            </div>
            
            <div style='background: #2d3748; padding: 20px; border-radius: 10px; border-left: 5px solid #7209b7;'>
                <h3 style='color: white; margin-top: 0;'>🔍 Key Findings</h3>
                <p style='color: #e2e8f0; line-height: 1.6; white-space: pre-wrap;'>{result['Key Findings']}</p>
            </div>
            
            <div style='background: linear-gradient(135deg, #4cc9f0, #4895ef); padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;'>
                <p style='color: white; margin: 0; font-weight: 600;'>✅ Processing Complete - Redacted file is ready for download</p>
            </div>
        </div>
        """
        
        # Return processed image or redacted file path
        processed_image = result.get('Processed Image')
        redacted_file = result.get('Redacted File')
        
        return processed_image, html_output, redacted_file
        
    except Exception as e:
        error_html = f"""
        <div style='background: #1a1a2e; padding: 30px; border-radius: 15px; border: 2px solid #f72585;'>
            <h2 style='color: #f72585;'>❌ Processing Error</h2>
            <p style='color: #e2e8f0;'>{str(e)}</p>
        </div>
        """
        return None, error_html, None


def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue")) as iface:
        with gr.Column(elem_classes="main-container"):
            with gr.Column(elem_classes="header"):
                gr.Markdown("""
                # 🔒 Advanced Document Redaction System
                ### Secure • Intelligent • Automated
                """)
            
            with gr.Row(elem_classes="feature-grid"):
                with gr.Column(elem_classes="feature-card"):
                    gr.Markdown("""
                    <div class='feature-icon'>🛡️</div>
                    <h3>Smart Detection</h3>
                    <p>Advanced pattern recognition for sensitive data including PII, credentials, and confidential information</p>
                    """)
                
                with gr.Column(elem_classes="feature-card"):
                    gr.Markdown("""
                    <div class='feature-icon'>🤖</div>
                    <h3>AI-Powered Analysis</h3>
                    <p>Gemini AI integration for intelligent content analysis and insights generation</p>
                    """)
                
                with gr.Column(elem_classes="feature-card"):
                    gr.Markdown("""
                    <div class='feature-icon'>🎯</div>
                    <h3>Multi-Format Support</h3>
                    <p>Process PDFs, Word documents, Excel files, PowerPoint, and images seamlessly</p>
                    """)
                
                with gr.Column(elem_classes="feature-card"):
                    gr.Markdown("""
                    <div class='feature-icon'>⚡</div>
                    <h3>Instant Processing</h3>
                    <p>Upload and analyze your document with AI-powered redaction in seconds</p>
                    """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📁 Supported File Formats")
                    with gr.Row(elem_classes="supported-formats"):
                        gr.Markdown("<div class='format-badge'>PDF</div>")
                        gr.Markdown("<div class='format-badge'>DOCX</div>")
                        gr.Markdown("<div class='format-badge'>XLSX</div>")
                        gr.Markdown("<div class='format-badge'>PPTX</div>")
                        gr.Markdown("<div class='format-badge'>JPG/PNG</div>")
            
            with gr.Column(elem_classes="input-section"):
                gr.Markdown("""
                ## 🚀 Get Started
                Upload your document to begin automated analysis and redaction
                """)
                
                with gr.Row():
                    file_input = gr.File(
                        label="📂 Upload Document",
                        file_types=[".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".ppt", ".jpg", ".jpeg", ".png"],
                        type="filepath",
                        elem_classes="file-input"
                    )
                
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze & Redact Document", 
                        variant="primary", 
                        size="lg",
                        elem_classes="analyze-btn"
                    )
                
                with gr.Row():
                    clear_btn = gr.Button(
                        "🔄 Clear & Upload New", 
                        variant="secondary",
                        size="lg",
                        elem_classes="clear-btn"
                    )
            
            with gr.Column(elem_classes="results-section"):
                gr.Markdown("## 📊 Analysis Results")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        processed_image_output = gr.Image(
                            label="🖼️ Processed Image",
                            type="pil",
                            visible=True,
                            show_label=True
                        )
                    
                    with gr.Column(scale=1):
                        output_html = gr.HTML(
                            label="📋 Document Analysis Report",
                            show_label=True
                        )
                
                with gr.Row():
                    download_file = gr.File(
                        label="📥 Download Redacted Document",
                        show_label=True,
                        elem_classes="download-section"
                    )
            
            with gr.Column(elem_classes="footer"):
                gr.Markdown("""
                ---
                **🔐 Security First** • **🤖 Powered by AI** • **⚡ Built for Performance**
                
                *Your documents are processed securely and never stored on our servers*
                """)
        
        # Event handlers
        analyze_btn.click(
            fn=process_uploaded_file,
            inputs=[file_input],
            outputs=[processed_image_output, output_html, download_file]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, None, None),
            inputs=[],
            outputs=[file_input, processed_image_output, output_html, download_file]
        )
    
    return iface