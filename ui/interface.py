"""
Gradio UI interface configuration
"""
import gradio as gr
from .styles import CUSTOM_CSS
from processors.directory_processor import process_directory


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
                    <h3>Batch Processing</h3>
                    <p>Automatically analyze and redact multiple files in directory with single click</p>
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
                Enter the directory path containing your documents to begin automated analysis and redaction
                """)
                
                with gr.Row():
                    directory_input = gr.Textbox(
                        label="📂 Directory Path",
                        placeholder="C:/Users/YourName/Documents/FilesToAnalyze",
                        lines=1,
                        info="Enter the full path to the directory containing your files",
                        elem_classes="directory-input"
                    )
                
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze & Redact All Files", 
                        variant="primary", 
                        size="lg",
                        elem_classes="analyze-btn"
                    )
            
            with gr.Column(elem_classes="results-section"):
                gr.Markdown("## 📊 Analysis Results")
                
                with gr.Row():
                    output_html = gr.HTML(
                        label="📋 Document Analysis Report",
                        show_label=True
                    )
            
            with gr.Column(elem_classes="footer"):
                gr.Markdown("""
                ---
                **🔐 Security First** • **🤖 Powered by AI** • **⚡ Built for Performance**
                
                *Your documents are processed securely and never stored on our servers*
                """)
        
        analyze_btn.click(
            fn=process_directory,
            inputs=[directory_input],
            outputs=[output_html]
        )
    
    return iface