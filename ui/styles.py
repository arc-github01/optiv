"""
CSS styles for the Gradio interface
"""

CUSTOM_CSS = """
:root {
    --primary: #4361ee;
    --secondary: #3a0ca3;
    --accent: #7209b7;
    --success: #4cc9f0;
    --warning: #f72585;
    --dark-bg: #0f0f23;
    --dark-card: #1a1a2e;
    --dark-text: #e2e8f0;
    --dark-border: #2d3748;
}

.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
    min-height: 100vh;
    color: var(--dark-text);
}

.main-container {
    background: rgba(26, 26, 46, 0.95) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    margin: 20px;
    padding: 30px;
    border: 1px solid var(--dark-border);
}

.header {
    text-align: center;
    margin-bottom: 30px;
    padding: 30px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 15px;
    color: white;
    box-shadow: 0 10px 20px rgba(67, 97, 238, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

.header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.header p {
    margin: 10px 0 0 0;
    font-size: 1.2em;
    opacity: 0.9;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.feature-card {
    background: var(--dark-card);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border-left: 5px solid var(--primary);
    border: 1px solid var(--dark-border);
    color: var(--dark-text);
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 30px rgba(0,0,0,0.4);
    border-left: 5px solid var(--accent);
}

.feature-icon {
    font-size: 2.5em;
    margin-bottom: 15px;
    color: var(--primary);
}

.feature-card h3 {
    margin: 0 0 10px 0;
    color: white;
    font-weight: 600;
}

.feature-card p {
    margin: 0;
    color: #cbd5e0;
    line-height: 1.5;
}

.input-section {
    background: var(--dark-card);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    margin: 20px 0;
    border: 1px solid var(--dark-border);
}

.input-section h2 {
    color: white;
    margin-bottom: 20px;
    font-weight: 600;
}

.file-input {
    background: #2d3748;
    border: 2px dashed #4a5568;
    border-radius: 10px;
    padding: 20px;
    transition: all 0.3s ease;
    color: white;
    min-height: 100px;
}

.file-input:hover {
    border-color: var(--primary);
    background: #374151;
}

.file-input label {
    color: white !important;
}

.analyze-btn {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border: none !important;
    padding: 15px 40px !important;
    border-radius: 50px !important;
    font-size: 1.1em !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(67, 97, 238, 0.4) !important;
    width: 100% !important;
}

.analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(67, 97, 238, 0.6) !important;
}

.clear-btn {
    background: linear-gradient(135deg, #6c757d, #495057) !important;
    color: white !important;
    border: none !important;
    padding: 12px 30px !important;
    border-radius: 50px !important;
    font-size: 1em !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(108, 117, 125, 0.3) !important;
    width: 100% !important;
}

.clear-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108, 117, 125, 0.5) !important;
}

.results-section {
    background: var(--dark-card);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    margin: 20px 0;
    border: 1px solid var(--dark-border);
}

.results-section h2 {
    color: white;
    margin-bottom: 20px;
    font-weight: 600;
    border-bottom: 3px solid var(--success);
    padding-bottom: 10px;
}

.download-section {
    background: linear-gradient(135deg, var(--success), #4895ef);
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    border: 2px solid rgba(76, 201, 240, 0.3);
}

.download-section label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
}

.file-type-badge {
    display: inline-block;
    background: var(--accent);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 600;
    margin: 5px;
}

.processing-status {
    background: linear-gradient(135deg, var(--success), #4895ef);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 20px 0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    color: #a0aec0;
    border-top: 1px solid var(--dark-border);
}

.supported-formats {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin: 20px 0;
}

.format-badge {
    background: #2d3748;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    color: var(--primary);
    border: 2px solid var(--primary);
    transition: all 0.3s ease;
}

.format-badge:hover {
    background: var(--primary);
    color: white;
    transform: scale(1.05);
}

label {
    color: white !important;
}

.gr-textbox, .gr-input {
    color: white !important;
}

.gr-markdown {
    color: var(--dark-text) !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: white !important;
}

.gr-file {
    background: #2d3748 !important;
    border: 2px dashed #4a5568 !important;
    border-radius: 10px !important;
}

.gr-file:hover {
    border-color: var(--primary) !important;
}

.gr-image {
    border-radius: 10px;
    border: 2px solid var(--dark-border);
    background: #2d3748;
}

.gr-html {
    background: transparent;
}

/* File upload area styling */
.wrap.svelte-1cl284s {
    background: #2d3748 !important;
    border: 2px dashed #4a5568 !important;
    border-radius: 10px !important;
}

.wrap.svelte-1cl284s:hover {
    border-color: var(--primary) !important;
    background: #374151 !important;
}
"""