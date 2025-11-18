"""Main entry point for CUA PDF Reader."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.gradio_app import create_ui

if __name__ == "__main__":
    print("="*50)
    print("Starting CUA PDF Reader")
    print("="*50)
    
    demo = create_ui()
    
    print("\n🚀 Launching Gradio UI...")
    print("📍 Open your browser to: http://localhost:7860")
    print("\nPress Ctrl+C to stop\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )