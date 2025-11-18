"""Gradio UI for CUA PDF Reader."""
import gradio as gr
import asyncio
import numpy as np
from typing import Optional
import os
from dotenv import load_dotenv

from utils.mongodb_handler import MongoDBHandler
from utils.vlm_processor import VLMProcessor
from agent.cua_agent import CUAAgent
from webrtc_server.server import webrtc_server

load_dotenv()


class CUAApp:
    """Main CUA Application."""
    
    def __init__(self):
        self.db = MongoDBHandler()
        self.vlm = VLMProcessor()
        self.agent = CUAAgent()
        self.current_frame = None
        self.extracted_content = None
        
        print("✅ CUA App initialized")
    
    def process_frame_from_upload(self, image):
        """Process an uploaded image (for testing without WebRTC)."""
        if image is None:
            return "No image provided", None
        
        try:
            # Process with VLM
            self.extracted_content = self.vlm.process_frame(image)
            self.current_frame = image
            
            # Format output
            output = "=== Extracted Content ===\n\n"
            
            if self.extracted_content.get('text'):
                output += f"Text:\n{self.extracted_content['text']}\n\n"
            else:
                output += "No text detected\n\n"
            
            if self.extracted_content.get('tables'):
                output += f"Tables detected: {len(self.extracted_content['tables'])}\n"
            
            if self.extracted_content.get('figures'):
                output += f"Figures detected: {len(self.extracted_content['figures'])}\n"
            
            # Store in database
            self.db.store_extracted_content({
                "content": self.extracted_content,
                "image_size": self.extracted_content.get('image_size')
            })
            
            return output, image
        
        except Exception as e:
            return f"Error processing frame: {str(e)}", None
    
    async def ask_question(self, question: str, history):
        """Process a question using the agent."""
        if not question.strip():
            return history, ""
        
        try:
            # Add user message to history
            history = history or []
            history.append([question, None])
            
            # Get response from agent
            response = await self.agent.process_query(
                query=question,
                extracted_content=self.extracted_content
            )
            
            # Update history with response
            history[-1][1] = response
            
            # Store in database
            self.db.store_question_answer({
                "question": question,
                "answer": response,
                "extracted_content": self.extracted_content
            })
            
            return history, ""
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history[-1][1] = error_msg
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
        return []


def create_ui():
    """Create the Gradio interface."""
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
                            placeholder="e.g., ARXIV:2205.14756",
                            info="Semantic Scholar or ArXiv ID"
                        )
                        paper_title_input = gr.Textbox(
                            label="Paper Title",
                            placeholder="e.g., EfficientViT: Multi-Scale Linear Attention..."
                        )
                        set_paper_btn = gr.Button("Set Current Paper", variant="primary")
                        paper_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("""
                        **System Info:**
                        - VLM: EasyOCR (GPU-accelerated)
                        - LLM: Qwen2.5-3B
                        - Database: MongoDB (Local)
                        - API: Semantic Scholar (No key needed)
                        """)
                
                set_paper_btn.click(
                    fn=app.set_paper_info,
                    inputs=[paper_id_input, paper_title_input],
                    outputs=paper_status
                )
            
            # Tab 2: Screen Capture (Test Mode)
            with gr.Tab("📸 Screen Capture (Test Mode)"):
                gr.Markdown("""
                ### Upload a PDF screenshot for testing
                In production, this will capture from WebRTC screen sharing.
                """)
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload PDF Screenshot",
                            type="numpy"
                        )
                        process_btn = gr.Button("Process Image", variant="primary")
                    
                    with gr.Column():
                        extracted_output = gr.Textbox(
                            label="Extracted Content",
                            lines=15,
                            max_lines=20
                        )
                        processed_image = gr.Image(label="Processed Image")
                
                process_btn.click(
                    fn=app.process_frame_from_upload,
                    inputs=image_input,
                    outputs=[extracted_output, processed_image]
                )
            
            # Tab 3: Chat Interface
            with gr.Tab("💬 Ask Questions"):
                gr.Markdown("""
                ### Ask questions about the paper
                The agent can explain text, analyze tables, discuss figures, and search for references.
                """)
                
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation"
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
                    
                    # Example questions
                    gr.Examples(
                        examples=[
                            "Explain the highlighted text in simple terms",
                            "What is this table showing?",
                            "What are the key findings of this paper?",
                            "What is the most important ablation study?",
                            "Search for related papers on attention mechanisms"
                        ],
                        inputs=question_input
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
        
        gr.Markdown("""
        ---
        **Tips:**
        1. Set your current paper in Configuration tab
        2. Upload a PDF screenshot in Screen Capture tab
        3. Ask questions in the Chat tab
        
        **Features:**
        - Extracts text, tables, and figures from PDFs
        - Searches academic paper databases
        - Provides tutorial-style explanations
        - Maintains conversation context
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )