"""Gradio UI for CUA PDF Reader."""
import gradio as gr
import asyncio
import numpy as np
import cv2
from typing import Optional
import os
from dotenv import load_dotenv
from PIL import Image
import time
import requests

from utils.mongodb_handler import MongoDBHandler
from utils.vlm_processor import VLMProcessor
from utils.frame_history import FrameHistory
from agent.cua_agent import CUAAgent
import json
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
        self.frame_history = FrameHistory(max_frames=50)  # Store last 50 frames
        self.auto_process_frames = False  # Auto-process frames as they arrive
        self.auto_capture_enabled = False  # Auto-capture frames periodically
        self.auto_capture_interval = 3.0  # Capture every 3 seconds
        self.auto_capture_thread = None  # Background thread for auto-capture
        self.auto_capture_running = False
        
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
            
            # Add to frame history
            if isinstance(image, np.ndarray):
                self.frame_history.add_frame(
                    frame=image,
                    extracted_content=self.extracted_content
                )
            
            print("✅ Processing complete!")
            return output, image
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, None
    
    async def ask_question(self, question: str, history):
        """Process a question using the agent with frame history context."""
        if not question.strip():
            return history, ""
        
        try:
            # Initialize history if needed
            history = history or []
            
            # Add user message to history (messages format)
            history.append({"role": "user", "content": question})
            
            # Search frame history for relevant content
            relevant_frames = self.frame_history.search_content(question, max_results=3)
            
            # Combine current content with historical content
            combined_content = self.extracted_content.copy() if self.extracted_content else {}
            
            if relevant_frames:
                # Add context from relevant historical frames
                historical_texts = []
                for frame_data, score in relevant_frames:
                    frame_content = frame_data.get("extracted_content", {})
                    frame_text = frame_content.get("text", "") or frame_data.get("structured_text", {}).get("full_text", "")
                    if frame_text:
                        historical_texts.append(f"[From earlier frame, relevance: {score:.1f}]\n{frame_text[:500]}")
                
                if historical_texts:
                    combined_content["historical_context"] = "\n\n".join(historical_texts)
                    combined_content["frame_history_count"] = self.frame_history.get_frame_count()
            
            # Get response from agent (pass combined content including history)
            response = await self.agent.process_query(
                query=question,
                extracted_content=combined_content,
                current_image=self.current_frame,
                frame_history=self.frame_history  # Pass history for advanced queries
            )
            
            # Add assistant response to history
            history.append({"role": "assistant", "content": response})
            
            # Store in database
            self.db.store_question_answer({
                "question": question,
                "answer": response,
                "extracted_content": combined_content,
                "frames_searched": len(relevant_frames)
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
    
    with gr.Blocks(title="CUA PDF Reader", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📄 Computer Using Agent - PDF Reader
        
        AI agent that helps you understand academic papers through screen capture and natural language interaction.
        """)
        
        with gr.Tabs():
            # Tab 1: Configuration
            with gr.Tab("⚙️ Configuration"):
                gr.Markdown("### System Configuration")
                
                with gr.Row():
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
            with gr.Tab("📸 Screen Capture (Recommended)"):
                gr.Markdown("""
                ### Upload PDF Screenshots
                
                **✨ Best method for accurate text extraction!**
                
                **Quick Start:**
                1. Open your PDF in any viewer (Adobe, Chrome, Firefox, etc.)
                2. Take screenshot: 
                   - **Windows**: `Win + Shift + S` or Snipping Tool
                   - **Mac**: `Cmd + Shift + 4`
                3. Upload screenshot below
                4. Click "Process Image"
                
                **Tips for Best Results:**
                - 📏 Zoom PDF to **150-200%** for crisp text
                - 🎯 For **single-column text**: Use normal mode
                - 📰 For **multi-column papers**: Enable "Full Page Mode"
                - ✂️ For **best column accuracy**: Crop each column separately
                
                **What gets detected:**
                - 📝 All text content (column-aware)
                - 📊 Tables and their positions
                - 🖼️ Figures and diagrams  
                - 🎨 Yellow highlights (user-added)
                - 💜 Auto-highlighted important sections
                """)
                
                with gr.Row():
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
            with gr.Tab("🔹 WebRTC Capture (Advanced)"):
                gr.Markdown("""
                ### Real-time Screen Capture via WebRTC
                
                **⚠️ IMPORTANT: PDF Capture Issues**
                
                PDFs often show as blank in WebRTC due to hardware acceleration. **Solutions:**
                
                **Option 1: Disable Hardware Acceleration**
                - **Chrome PDF**: Go to `chrome://flags` → search "hardware" → disable "Hardware-accelerated video decode"
                - **Adobe Reader**: Edit → Preferences → General → uncheck "Use hardware acceleration"  
                - **Firefox**: Better PDF capture by default
                
                **Option 2: Workaround (Easier)**
                1. Take a screenshot of PDF (Win + Shift + S)
                2. Paste screenshot into a new browser tab
                3. Share that tab via WebRTC instead
                
                **Quick Start:**
                1. Click link below to open screen share client
                2. Click "Start Screen Share" in new window
                3. Select your PDF tab/window or entire screen
                4. Come back here and click "Capture Latest Frame"
                5. Process the captured frame
                
                **Screen Share Client:** [Open in New Tab](http://localhost:8080/client.html){target="_blank"}
                
                **Troubleshooting:**
                - ❌ Blank capture? → Share entire screen instead of tab
                - ❌ Still blank? → Use Firefox or the Screenshot method above
                - ✅ Works best with: Image files, web pages, non-accelerated content
                """)
                
                with gr.Row():
                    with gr.Column():
                        webrtc_status = gr.Textbox(
                            label="WebRTC Status",
                            value="Not connected",
                            interactive=False
                        )
                        auto_capture_toggle = gr.Checkbox(
                            label="🤖 Auto-capture frames",
                            value=False,
                            info="Automatically capture frames every 3 seconds (no manual clicking needed!)"
                        )
                        auto_process_toggle = gr.Checkbox(
                            label="🔄 Auto-process frames",
                            value=False,
                            info="Automatically process captured frames and add to history"
                        )
                        capture_frame_btn = gr.Button("Capture Latest Frame (Manual)", variant="primary")
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
                                frame_rgb = frame.copy()
                                if isinstance(frame, np.ndarray):
                                    # Convert BGR to RGB for PIL
                                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    frame_pil = Image.fromarray(frame_rgb)
                                else:
                                    frame_pil = frame
                                
                                app.current_frame = frame_pil
                                
                                # Auto-process frame if enabled (but don't block UI)
                                if app.auto_process_frames:
                                    # Process in background thread to avoid blocking
                                    def auto_process():
                                        try:
                                            # Process using VLM (process_frame expects numpy array)
                                            # frame is already numpy array from WebRTC
                                            processed = app.vlm.process_frame(frame, full_page=True)
                                            app.frame_history.add_frame(
                                                frame=frame,
                                                extracted_content=processed
                                            )
                                            print(f"✅ Frame auto-processed and added to history (total: {app.frame_history.get_frame_count()})")
                                        except Exception as e:
                                            print(f"⚠️ Auto-processing failed: {e}")
                                            import traceback
                                            traceback.print_exc()
                                    
                                    # Start background processing
                                    threading.Thread(target=auto_process, daemon=True).start()
                                
                                print(f"✅ Frame converted to PIL: {frame_pil.size}")
                                history_info = f" | History: {app.frame_history.get_frame_count()} frames"
                                return frame_pil, f"✅ Frame captured! Size: {frame_pil.size}{history_info}"
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
                        # Convert PIL back to numpy if needed
                        if isinstance(app.current_frame, Image.Image):
                            frame_np = np.array(app.current_frame)
                        else:
                            frame_np = app.current_frame
                        
                        result = app.process_frame_from_upload(frame_np, full_page_mode=True)
                        output = result[0]
                        
                        # Add frame history info
                        history_count = app.frame_history.get_frame_count()
                        output += f"\n\n📚 **Frame History:** {history_count} frames stored"
                        output += f"\n💡 **Context Awareness:** You can ask questions about content from previous frames!"
                        output += f"\n   Example: 'What did the abstract say?' or 'Explain the method from 5 frames ago'"
                        
                        # If we have multiple frames, show stitching status
                        if history_count > 1:
                            try:
                                stitched_info = app.frame_history.stitch_all_frames()
                                if stitched_info.get("method") == "stitched":
                                    output += f"\n\n🔗 **Document Stitching:** {stitched_info.get('frames_used', 0)} frames stitched together"
                                    output += f"\n   Full document reconstruction available! Click 'Stitch All Frames' in Ask Questions tab to view."
                            except Exception as e:
                                output += f"\n\n⚠️ Stitching check failed: {str(e)}"
                        
                        return output
                    except Exception as e:
                        return f"❌ Error: {str(e)}"
                
                def auto_capture_loop():
                    """Background thread that automatically captures frames periodically."""
                    last_frame_hash = None
                    frame_count = 0
                    
                    while app.auto_capture_running:
                        try:
                            # Check if WebRTC is connected (non-blocking)
                            try:
                                status_resp = requests.get("http://localhost:8080/status", timeout=0.5)
                                status_data = status_resp.json()
                                is_capturing = status_data.get("is_capturing", False)
                                has_frame = status_data.get("has_frame", False)
                                
                                if is_capturing and has_frame:
                                    # Capture frame
                                    frame = webrtc_server.get_latest_frame_sync()
                                    if frame is not None:
                                        # Simple duplicate detection (hash-based)
                                        import hashlib
                                        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
                                        
                                        # Skip if same frame
                                        if frame_hash == last_frame_hash:
                                            time.sleep(app.auto_capture_interval)
                                            continue
                                        
                                        last_frame_hash = frame_hash
                                        frame_count += 1
                                        
                                        # Convert to PIL (in background to avoid blocking)
                                        def process_frame():
                                            try:
                                                frame_rgb = frame.copy()
                                                if isinstance(frame, np.ndarray):
                                                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                                                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                                    frame_pil = Image.fromarray(frame_rgb)
                                                else:
                                                    frame_pil = frame
                                                
                                                app.current_frame = frame_pil
                                                
                                                # Auto-process if enabled
                                                if app.auto_process_frames:
                                                    try:
                                                        processed = app.vlm.process_frame(frame, full_page=True)
                                                        app.frame_history.add_frame(
                                                            frame=frame,
                                                            extracted_content=processed
                                                        )
                                                        print(f"✅ Auto-captured & processed frame #{frame_count} (total: {app.frame_history.get_frame_count()})")
                                                    except Exception as e:
                                                        print(f"⚠️ Auto-processing failed: {e}")
                                                        # Store frame without processing on error
                                                        app.frame_history.add_frame(frame=frame, extracted_content=None)
                                                else:
                                                    # Just store frame without processing
                                                    app.frame_history.add_frame(frame=frame, extracted_content=None)
                                                    print(f"✅ Auto-captured frame #{frame_count} (total: {app.frame_history.get_frame_count()})")
                                            except Exception as e:
                                                print(f"⚠️ Frame processing error: {e}")
                                        
                                        # Process in separate thread to avoid blocking
                                        threading.Thread(target=process_frame, daemon=True).start()
                                        
                            except requests.exceptions.RequestException:
                                pass  # WebRTC not connected, silently continue
                            except Exception as e:
                                print(f"⚠️ Auto-capture check error: {e}")
                            
                            # Wait for next capture
                            time.sleep(app.auto_capture_interval)
                            
                        except KeyboardInterrupt:
                            break
                        except Exception as e:
                            print(f"⚠️ Auto-capture loop error: {e}")
                            time.sleep(app.auto_capture_interval)
                
                def toggle_auto_capture(enabled):
                    """Toggle automatic frame capture."""
                    app.auto_capture_enabled = enabled
                    
                    if enabled:
                        if not app.auto_capture_running:
                            app.auto_capture_running = True
                            app.auto_capture_thread = threading.Thread(target=auto_capture_loop, daemon=True)
                            app.auto_capture_thread.start()
                            status_msg = f"✅ Auto-capture ENABLED - Capturing frames every {app.auto_capture_interval}s"
                            status_msg += f"\n📚 Current history: {app.frame_history.get_frame_count()} frames"
                            status_msg += "\n💡 Just scroll through your PDF - frames will be captured automatically!"
                        else:
                            status_msg = "✅ Auto-capture already running"
                    else:
                        app.auto_capture_running = False
                        status_msg = "⏸️ Auto-capture DISABLED"
                    
                    return status_msg
                
                def toggle_auto_process(enabled):
                    """Toggle automatic frame processing."""
                    app.auto_process_frames = enabled
                    status_msg = f"✅ Auto-processing {'ENABLED' if enabled else 'DISABLED'}"
                    if enabled:
                        status_msg += f"\n📚 Current history: {app.frame_history.get_frame_count()} frames"
                        status_msg += "\n💡 Frames will be processed automatically when captured"
                    return status_msg
                
                auto_capture_toggle.change(
                    fn=toggle_auto_capture,
                    inputs=auto_capture_toggle,
                    outputs=webrtc_status
                )
                
                auto_process_toggle.change(
                    fn=toggle_auto_process,
                    inputs=auto_process_toggle,
                    outputs=webrtc_status
                )
                
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
                gr.Markdown("""
                ### Ask questions about the paper
                
                The agent can:
                - 📖 Explain highlighted text and technical concepts
                - 📊 Analyze tables and figures
                - 🔍 Search for related papers and references
                - 💡 Identify key findings and ablation studies
                - 🎨 Explain auto-highlighted important sections
                - 📚 **Answer questions about content from previous frames** (context-aware!)
                - 🔗 **Access full stitched document** (reconstructed from all frames!)
                
                **Frame History & Stitching:** 
                - The system remembers the last 50 frames you've processed
                - Frames are automatically stitched together to recreate the full PDF
                - You can ask about content from earlier frames, even if you've scrolled away!
                - The full document is reconstructed by detecting overlaps and removing duplicates
                
                **Make sure to process PDF screenshots first in the "Screen Capture" tab!**
                """)
                
                # Frame history status
                history_status = gr.Markdown(
                    value=f"📚 **Frame History:** {app.frame_history.get_frame_count()} frames stored",
                    visible=True
                )
                
                # Stitch document button
                def stitch_document():
                    """Stitch all frames together to reconstruct full document."""
                    history_count = app.frame_history.get_frame_count()
                    if history_count < 2:
                        return "⚠️ Need at least 2 frames to stitch. Process more frames first!"
                    
                    try:
                        stitched = app.frame_history.stitch_all_frames()
                        method = stitched.get("method", "unknown")
                        frames_used = stitched.get("frames_used", 0)
                        total_frames = stitched.get("total_frames", 0)
                        unique_frames = stitched.get("unique_frames", 0)
                        text = stitched.get("stitched_text", "")
                        
                        output = f"🔗 **Document Stitching Complete!**\n\n"
                        output += f"**Method:** {method}\n"
                        output += f"**Frames Used:** {frames_used} / {total_frames} (unique: {unique_frames})\n\n"
                        output += f"**Stitched Document Preview:**\n"
                        output += f"---\n{text[:2000]}...\n---\n\n"
                        output += f"💡 The full stitched document is now available for questions!"
                        
                        # Store stitched document
                        app.extracted_content = app.extracted_content or {}
                        app.extracted_content["stitched_document"] = stitched
                        
                        return output
                    except Exception as e:
                        import traceback
                        return f"❌ Error stitching: {str(e)}\n{traceback.format_exc()}"
                
                with gr.Row():
                    stitch_btn = gr.Button("🔗 Stitch All Frames → Reconstruct Full Document", variant="secondary")
                    stitch_output = gr.Textbox(label="Stitching Result", lines=10, interactive=False)
                
                stitch_btn.click(fn=stitch_document, outputs=stitch_output)
                
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation",
                    type="messages"
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
                    result_history, _ = await app.ask_question(question, history)
                    # Update history status
                    history_count = app.frame_history.get_frame_count()
                    history_status_value = f"📚 **Frame History:** {history_count} frames stored"
                    if history_count > 0:
                        history_status_value += f"\n💡 You can ask about content from any of these {history_count} frames!"
                    return result_history, "", history_status_value
                
                submit_btn.click(
                    fn=handle_question,
                    inputs=[question_input, chatbot],
                    outputs=[chatbot, question_input, history_status]
                )
                
                question_input.submit(
                    fn=handle_question,
                    inputs=[question_input, chatbot],
                    outputs=[chatbot, question_input, history_status]
                )
                
                clear_btn.click(
                    fn=app.reset_conversation,
                    outputs=chatbot
                )
        
        gr.Markdown("""
        ---
        ### 📖 Quick Guide
        
        **1. Setup (Configuration Tab)**
        - Set your paper ID and title for context
        
        **2. Extract Content (Screen Capture Tab)**  
        - Upload PDF screenshot
        - Enable "Full Page Mode" for multi-column papers
        - Process the image
        
        **3. Ask Questions (Chat Tab)**
        - Ask about highlighted sections, tables, figures
        - Get explanations of complex concepts
        - Search for related papers
        
        ### ✨ Features
        - 📝 Column-aware text extraction
        - 📊 Automatic table & figure detection
        - 🎨 Yellow highlight detection (user annotations)
        - 💜 Auto-highlighting of important sections (Abstract, Methods, Results, etc.)
        - 🔍 Semantic Scholar integration for paper search
        - 💾 MongoDB storage of all interactions
        - 🤖 Context-aware Q&A with Qwen2.5-3B
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )