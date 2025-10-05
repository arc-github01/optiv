"""
Main application entry point for Document Redaction System
"""
from ui.interface import create_interface
from config.settings import SERVER_HOST, SERVER_PORT


def main():
    """Initialize and launch the application"""
    print("=" * 60)
    print("🔒 Advanced Document Redaction System")
    print("=" * 60)
    print(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)
    
    iface = create_interface()
    
    iface.launch(
        share=False,
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        show_error=True
    )


if __name__ == "__main__":
    main()