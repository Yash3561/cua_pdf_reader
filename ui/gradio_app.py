"""Gradio UI for CUA PDF Reader."""
import gradio as gr
import asyncio
import numpy as np
import cv2
from typing import Optional
import os
from dotenv import load_dotenv

from utils.mongodb_handler import MongoDBHandler
from utils.vlm_processor import VLMProcessor
from agent.cua_agent import CUAAgent
from webrtc_server.server import webrtc_server
import threading
import uvicorn
from webrtc_server.signaling import app as signaling_app

load_dotenv()


class CUAApp:
    """Main CUA Application."""
    
    def __init__(self):
        self.db = MongoDBHandler()
        self.vlm = VLMProcessor()
        self.agent = CUAAgent(vlm_processor=self.vlm)  # Pass VLM to agent
        self.current_frame = None
        self.extracted_content = None
        
        print("✅ CUA App initialized")
    
    def process_frame_from_upload(self, image, full_page_mode=False):
        """Process an uploaded image (for testing without WebRTC)."""
        if image is None:
            return "No image provided", None
        
        try:
            # Process with VLM
            print(f"🔍 Processing image (full_page={full_page_mode})...")
            self.extracted_content = self.vlm.process_frame(image, full_page=full_page_mode)
            self.current_frame = image
            
            # Format output
            output = "=== 📄 EXTRACTED CONTENT ===\n\n"
            
            # Text section
            output += "📝 **TEXT:**\n"
            if self.extracted_content.get('text'):
                text = self.extracted_content['text']
                output += f"{text}\n\n"
                output += f"[{len(text)} characters extracted]\n\n"
            else:
                output += "No text detected\n\n"
            
            # Tables
            tables = self.extracted_content.get('tables', [])
            output += f"📊 **TABLES:** {len(tables)}\n"
            if tables:
                for i, table in enumerate(tables):
                    output += f"  • Table {i+1} at position {table.get('bbox')}\n"
            output += "\n"
            
            # Figures
            figures = self.extracted_content.get('figures', [])
            output += f"🖼️ **FIGURES:** {len(figures)}\n"
            if figures:
                for i, fig in enumerate(figures):
                    output += f"  • Figure {i+1} (area: {fig.get('area')}px²)\n"
            output += "\n"
            
            # Highlights
            highlights = self.extracted_content.get('highlights', {})
            yellow_count = len(highlights.get('yellow', []))
            purple_count = len(highlights.get('purple', []))
            output += f"🎨 **HIGHLIGHTS:**\n"
            output += f"  • Yellow (user): {yellow_count}\n"
            output += f"  • Purple (auto): {purple_count}\n"
            
            # Auto-highlights with explanations
            auto_highlights = self.extracted_content.get('auto_highlights', [])
            if auto_highlights:
                output += f"\n💜 **AUTO-HIGHLIGHTED SECTIONS:**\n"
                for i, hl in enumerate(auto_highlights[:5], 1):  # Show first 5
                    output += f"  {i}. {hl.get('type', 'Unknown').title()}: {hl.get('reason', 'N/A')}\n"
            
            # Structured sections preview
            structured = self.extracted_content.get('structured_text', {})
            if structured.get('sections'):
                output += f"\n📑 **SECTIONS DETECTED:** {len(structured.get('sections', []))}\n"
                for section in structured.get('sections', [])[:3]:  # Show first 3
                    title = section.get('title', 'Unknown')
                    content_preview = section.get('content', '')[:100]
                    output += f"  • {title}: {content_preview}...\n"
            
            # Store in database
            self.db.store_extracted_content({
                "content": self.extracted_content,
                "full_page_mode": full_page_mode
            })
            
            print("✅ Processing complete!")
            return output, image
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, None
    
    async def ask_question(self, question: str, history):
        """Process a question using the agent."""
        if not question.strip():
            return history, ""
        
        try:
            # Initialize history if needed
            history = history or []
            
            # Add user message to history (messages format)
            history.append({"role": "user", "content": question})
            
            # Get response from agent (pass current image for diagram parsing)
            response = await self.agent.process_query(
                query=question,
                extracted_content=self.extracted_content,
                current_image=self.current_frame
            )
            
            # Add assistant response to history
            history.append({"role": "assistant", "content": response})
            
            # Store in database
            self.db.store_question_answer({
                "question": question,
                "answer": response,
                "extracted_content": self.extracted_content
            })
            
            return history, ""
        
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\nPlease ensure:\n1. Ollama is running (ollama serve)\n2. Model qwen2.5:3b is installed (ollama pull qwen2.5:3b)"
            print(f"❌ Chat error: {e}")
            print(traceback.format_exc())
            
            # Add error to history
            history = history or []
            if history and history[-1].get("role") == "user":
                history.append({"role": "assistant", "content": error_msg})
            else:
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": error_msg})
            
            return history, ""
    
    def set_paper_info(self, paper_id: str, paper_title: str):
        """Set the current paper being viewed."""
        if paper_id and paper_title:
            self.agent.set_current_paper(paper_id, paper_title)
            return f"✅ Set current paper: {paper_title}"
        return "⚠️ Please provide both paper ID and title"
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.agent.reset_conversation()
        return []  # Return empty list for messages format


def start_signaling_server():
    """Start the WebRTC signaling server in a background thread."""
    def run_server():
        uvicorn.run(signaling_app, host="0.0.0.0", port=8080, log_level="info")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("✅ WebRTC signaling server started on http://localhost:8080")
    return server_thread


def create_ui():
    """Create the Gradio interface."""
    # Start signaling server
    start_signaling_server()
    
    app = CUAApp()
    
    # Load custom CSS
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()
    
    with gr.Blocks(
        title="CUA PDF Reader", 
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet"),
        css=custom_css
    ) as demo:
        # Modern Header
        with gr.Row(elem_classes=["glass-panel", "app-header"]):
            gr.HTML("""
            <div style="text-align: center;">
                <h1 class="gradient-text app-title">📄 CUA PDF Reader</h1>
                <p class="app-subtitle">AI-Powered Academic Paper Analysis • Extract • Understand • Interact</p>
                <div style="margin-top: 1rem;">
                    <span class="status-badge success">✓ MongoDB</span>
                    <span class="status-badge success">✓ Ollama</span>
                    <span class="status-badge success">✓ WebRTC</span>
                </div>
            </div>
            """)
        
        with gr.Tabs():
            # Tab 1: Configuration
            with gr.Tab("⚙️ Configuration"):
                with gr.Row(elem_classes=["glass-panel"]):
                    gr.Markdown("### 📋 Paper Configuration")
                
                with gr.Row(elem_classes=["glass-panel"]):
                    with gr.Column():
                        paper_id_input = gr.Textbox(
                            label="Paper ID",
                            placeholder="e.g., ARXIV:2401.16889",
                            info="Semantic Scholar or ArXiv ID"
                        )
                        paper_title_input = gr.Textbox(
                            label="Paper Title",
                            placeholder="e.g., Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control"
                        )
                        set_paper_btn = gr.Button("Set Current Paper", variant="primary")
                        paper_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("""
                        **System Info:**
                        - VLM: EasyOCR (GPU-accelerated)
                        - LLM: Qwen2.5-3B via Ollama
                        - Database: MongoDB (Local)
                        - API: Semantic Scholar (Free tier)
                        
                        **Current Features:**
                        - ✅ Text extraction from PDFs
                        - ✅ Table & figure detection
                        - ✅ Highlight detection (yellow/purple)
                        - ✅ Auto-highlighting important sections
                        - ✅ Column-aware text extraction
                        - ✅ Q&A with context
                        """)
                
                set_paper_btn.click(
                    fn=app.set_paper_info,
                    inputs=[paper_id_input, paper_title_input],
                    outputs=paper_status
                )
            
            # Tab 2: Screen Capture (RECOMMENDED - PRIMARY METHOD)
            with gr.Tab("📸 Screen Capture"):
                with gr.Row(elem_classes=["glass-panel"]):
                    gr.Markdown("""
                    ### 📝 Upload PDF Screenshots
                    
                    **✨ Best method for accurate text extraction!**
                    
                    **Quick Start:** Open PDF → Screenshot (`Win+Shift+S` / `Cmd+Shift+4`) → Upload → Process
                    
                    **Tips:** Zoom to 150-200% • Enable Full Page Mode for multi-column papers
                    """)
                
                with gr.Row(elem_classes=["glass-panel"]):
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload PDF Screenshot",
                            type="numpy"
                        )
                        full_page_mode = gr.Checkbox(
                            label="Full Page Mode (better for multi-column papers)",
                            value=True,
                            info="Splits page into regions for column-aware extraction"
                        )
                        process_btn = gr.Button("Process Image", variant="primary", size="lg")
                    
                    with gr.Column():
                        extracted_output = gr.Textbox(
                            label="Extracted Content",
                            lines=20,
                            max_lines=25
                        )
                        processed_image = gr.Image(label="Processed Image")
                
                def process_with_mode(image, full_page):
                    return app.process_frame_from_upload(image, full_page)
                
                process_btn.click(
                    fn=process_with_mode,
                    inputs=[image_input, full_page_mode],
                    outputs=[extracted_output, processed_image]
                )
            
            # Tab 3: WebRTC Capture (Advanced)
            with gr.Tab("🔹 WebRTC Capture"):
                with gr.Row(elem_classes=["glass-panel"]):
                    gr.Markdown("""
                    ### 🔴 Real-time Screen Capture
                    
                    **⚠️ Note:** PDFs may appear blank due to hardware acceleration. Use Screenshot tab for best results.
                    
                    **Quick Start:**
                    1. Click the button below to open the WebRTC client
                    2. Click "Start Screen Share" in the new window
                    3. Select your screen/window to share
                    4. Return here and click "Capture Latest Frame"
                    """)
                
                with gr.Row(elem_classes=["glass-panel"]):
                    with gr.Column():
                        open_client_btn = gr.Button(
                            "🌐 Open WebRTC Client",
                            variant="secondary",
                            size="lg",
                            elem_id="open-client-btn"
                        )
                        gr.HTML("""
                        <script>
                        document.getElementById('open-client-btn').onclick = function() {
                            window.open('http://localhost:8080/client.html', '_blank');
                        };
                        </script>
                        """)
                
                with gr.Row(elem_classes=["glass-panel"]):
                    with gr.Column():
                        webrtc_status = gr.Textbox(
                            label="WebRTC Status",
                            value="Not connected",
                            interactive=False
                        )
                        capture_frame_btn = gr.Button("Capture Latest Frame", variant="primary")
                        process_webrtc_btn = gr.Button("Process Captured Frame", variant="secondary")
                    
                    with gr.Column():
                        webrtc_frame = gr.Image(label="Captured Frame", type="numpy")
                        webrtc_output = gr.Textbox(
                            label="Extracted Content",
                            lines=15
                        )
                
                def capture_webrtc_frame():
                    """Capture the latest frame from WebRTC."""
                    import requests
                    try:
                        # Check signaling server status
                        print("🔍 Checking WebRTC status...")
                        status_resp = requests.get("http://localhost:8080/status", timeout=2)
                        status_data = status_resp.json()
                        print(f"📊 Status response: {status_data}")
                        
                        is_capturing = status_data.get("is_capturing", False)
                        has_frame = status_data.get("has_frame", False)
                        
                        print(f"📹 is_capturing: {is_capturing}, has_frame: {has_frame}")
                        
                        if is_capturing:
                            frame = webrtc_server.get_latest_frame_sync()  # Use sync method
                            print(f"🖼️ Frame from server: {type(frame)}, {frame.shape if hasattr(frame, 'shape') else 'None'}")
                            
                            if frame is not None:
                                # Convert numpy array to PIL Image if needed
                                from PIL import Image
                                if isinstance(frame, np.ndarray):
                                    # Convert BGR to RGB for PIL
                                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    frame = Image.fromarray(frame)
                                app.current_frame = frame
                                print(f"✅ Frame converted to PIL: {frame.size}")
                                return frame, f"✅ Frame captured successfully! Size: {frame.size}"
                            return None, f"⚠️ Connected but no frame available yet. is_capturing={is_capturing}, has_frame={has_frame}"
                        else:
                            return None, "❌ WebRTC not connected. Please start screen sharing at http://localhost:8080/client.html"
                    except Exception as e:
                        import traceback
                        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)
                        return None, error_msg
                
                def process_webrtc_frame():
                    """Process the captured WebRTC frame."""
                    if app.current_frame is None:
                        return "⚠️ No frame captured. Please capture a frame first."
                    try:
                        result = app.process_frame_from_upload(app.current_frame, full_page_mode=True)
                        return result[0]  # Return just the text output
                    except Exception as e:
                        return f"❌ Error: {str(e)}"
                
                capture_frame_btn.click(
                    fn=capture_webrtc_frame,
                    outputs=[webrtc_frame, webrtc_status]
                )
                
                process_webrtc_btn.click(
                    fn=process_webrtc_frame,
                    outputs=webrtc_output
                )
            
            # Tab 4: Chat Interface
            with gr.Tab("💬 Ask Questions"):
                with gr.Row(elem_classes=["glass-panel"]):
                    gr.Markdown("""
                    ### 🤖 AI Assistant
                    
                    Ask questions about your paper! The agent can explain highlights, analyze tables, identify key findings, and search for related papers.
                    
                    **💡 Tip:** Process a PDF screenshot first in the Screen Capture tab!
                    """)
                
                with gr.Row(elem_classes=["glass-panel"]):
                    chatbot = gr.Chatbot(
                        height=500,
                        label="Conversation",
                        type="messages",
                        elem_classes=["chatbot-container"]
                    )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about the paper...",
                        scale=4
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation")
                    
                    # Example questions from project requirements
                    gr.Examples(
                        examples=[
                            "Explain the yellow highlighted text in simple terms",
                            "What are the auto-highlighted important sections and why?",
                            "What is table 1 showing?",
                            "What is the most detrimental ablation study in this paper and why?",
                            "Search for related papers on attention mechanisms"
                        ],
                        inputs=question_input,
                        label="Example Questions (from Project Requirements)"
                    )
                
                async def handle_question(question, history):
                    return await app.ask_question(question, history)
                
                submit_btn.click(
                    fn=handle_question,
                    inputs=[question_input, chatbot],
                    outputs=[chatbot, question_input]
                )
                
                question_input.submit(
                    fn=handle_question,
                    inputs=[question_input, chatbot],
                    outputs=[chatbot, question_input]
                )
                
                clear_btn.click(
                    fn=app.reset_conversation,
                    outputs=chatbot
                )
        
        # Footer - Quick Guide
        with gr.Row(elem_classes=["glass-panel"]):
            with gr.Column():
                gr.Markdown("""
                ### 📖 Quick Start Guide
                
                **1. Configure** → Set paper ID and title  
                **2. Extract** → Upload PDF screenshot and process  
                **3. Ask** → Chat with the AI about your paper
                
                ### ✨ Key Features
                - 📝 Column-aware text extraction • 📊 Table & figure detection • 🎨 Highlight detection  
                - 💜 Auto-highlighting • 🔍 Semantic Scholar search • 💾 MongoDB storage • 🤖 Context-aware Q&A
                """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )