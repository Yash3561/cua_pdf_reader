"""Computer Using Agent using Ollama directly."""
import os
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import ollama
import logging
from .tools import (
    search_semantic_scholar,
    get_paper_details,
    get_paper_citations,
    get_paper_references
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CUADependencies(BaseModel):
    """Dependencies for the CUA agent."""
    current_paper_id: Optional[str] = None
    current_paper_title: Optional[str] = None
    extracted_content: Dict = Field(default_factory=dict)
    conversation_history: List[Dict] = Field(default_factory=list)
    current_image: Optional[object] = None  # Store PIL Image for diagram parsing


SYSTEM_PROMPT = """You are a Computer Using Agent (CUA) specialized in helping users understand academic papers in AI.

Your capabilities:
1. Extract and explain text, tables, figures, and equations from papers displayed on screen
2. Search for paper metadata and references using Semantic Scholar
3. Provide tutorial-style explanations for complex concepts
4. Identify and explain important sections of papers
5. Analyze ablation studies and experimental results

When answering:
- Be clear, concise, and educational
- Use the extracted content from the screen when relevant
- Cite sources when referencing external papers
- Explain technical concepts in an accessible way
- Highlight key insights and findings

If the user asks you to search for papers or get paper information, let them know you can help with that.
"""


class CUAAgent:
    """Main CUA Agent using Ollama."""
    
    def __init__(self, model_name: str = "qwen2.5:3b", vlm_processor=None):
        self.model_name = model_name
        self.deps = CUADependencies()
        self.system_prompt = SYSTEM_PROMPT
        self.vlm = vlm_processor  # Store VLM processor for diagram parsing
    
    async def process_query(
        self,
        query: str,
        extracted_content: Optional[Dict] = None,
        current_image: Optional[object] = None,
        frame_history: Optional[object] = None
    ) -> str:
        """
        Process a user query using the agent.
        
        Args:
            query: User's question
            extracted_content: Content extracted from screen by VLM
            current_image: PIL Image for diagram parsing (optional)
        
        Returns:
            Agent's response
        """
        # Update extracted content if provided
        if extracted_content:
            self.deps.extracted_content = extracted_content
        if current_image is not None:
            self.deps.current_image = current_image
        
        # Check if query involves tool usage
        response = await self._handle_query_with_tools(query)
        if response:
            return response
        
        # Build context message for regular query
        context_parts = []
        
        if self.deps.current_paper_title:
            context_parts.append(f"Current paper: {self.deps.current_paper_title}")
        
        # Add historical context if available
        if extracted_content and extracted_content.get('historical_context'):
            context_parts.append(f"Historical context from previous frames:\n{extracted_content['historical_context']}")
            if extracted_content.get('frame_history_count'):
                context_parts.append(f"Total frames in history: {extracted_content['frame_history_count']}")
        
        # Get text from current frame
        current_text = self.deps.extracted_content.get('text', '')
        if not current_text:
            structured = self.deps.extracted_content.get('structured_text', {})
            current_text = structured.get('full_text', '')
        
        if current_text:
            # Truncate if too long
            if len(current_text) > 2000:
                current_text = current_text[:2000] + "..."
            context_parts.append(f"Current frame extracted text:\n{current_text}")
        
        # If frame history is available, add summary
        if frame_history and hasattr(frame_history, 'get_frame_count'):
            history_count = frame_history.get_frame_count()
            if history_count > 0:
                # Check if stitched document is available
                stitched_doc = self.deps.extracted_content.get("stitched_document")
                if stitched_doc and stitched_doc.get("stitched_text"):
                    # Use stitched document (full reconstructed PDF)
                    stitched_text = stitched_doc.get("stitched_text", "")
                    if len(stitched_text) > 5000:
                        stitched_text = stitched_text[:5000] + "..."
                    context_parts.append(f"\n📄 **Full Stitched Document** (reconstructed from {stitched_doc.get('frames_used', 0)} frames):\n{stitched_text}")
                else:
                    # Fallback: Get combined text from all frames (limited)
                    all_text = frame_history.get_all_text(max_chars=3000)
                    if all_text:
                        context_parts.append(f"\nCombined text from {history_count} frames in history:\n{all_text}")
        
        if self.deps.extracted_content.get('tables'):
            context_parts.append(f"Found {len(self.deps.extracted_content['tables'])} tables on screen")
        
        if self.deps.extracted_content.get('figures'):
            context_parts.append(f"Found {len(self.deps.extracted_content['figures'])} figures on screen")
        
        # Prepare messages for Ollama
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
        # Add conversation history (last 3 exchanges)
        for msg in self.deps.conversation_history[-6:]:
            messages.append(msg)
        
        # Add current query with context
        user_message = query
        if context_parts:
            user_message = "\n\n".join(context_parts) + f"\n\nUser query: {query}"
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response from Ollama
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            
            assistant_response = response['message']['content']
            
            # Update conversation history
            self.deps.conversation_history.append({
                "role": "user",
                "content": query
            })
            self.deps.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            # Keep only last 10 exchanges
            if len(self.deps.conversation_history) > 20:
                self.deps.conversation_history = self.deps.conversation_history[-20:]
            
            return assistant_response
        
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _extract_yellow_highlighted_text(self) -> Optional[str]:
        """Extract text from yellow highlighted regions."""
        highlights = self.deps.extracted_content.get('highlights', {}).get('yellow', [])
        if not highlights:
            return None
        
        # Get structured text blocks
        structured = self.deps.extracted_content.get('structured_text', {})
        raw_blocks = structured.get('raw_blocks', [])
        
        # Find text blocks that intersect with yellow highlights
        highlighted_texts = []
        for highlight in highlights:
            hl_bbox = highlight.get('bbox', [])
            if len(hl_bbox) != 4:
                continue
            
            # Handle both [x, y, w, h] and [x1, y1, x2, y2] formats
            if len(hl_bbox) == 4:
                # Check if it's [x, y, w, h] or [x1, y1, x2, y2]
                # If w or h is very large, it's likely [x, y, w, h]
                if hl_bbox[2] > 1000 or hl_bbox[3] > 1000:
                    # Format: [x, y, w, h]
                    hl_x1, hl_y1, hl_w, hl_h = hl_bbox
                    hl_x2 = hl_x1 + hl_w
                    hl_y2 = hl_y1 + hl_h
                else:
                    # Format: [x1, y1, x2, y2]
                    hl_x1, hl_y1, hl_x2, hl_y2 = hl_bbox
            
            for block in raw_blocks:
                block_bbox = block.get('bbox', [])
                if len(block_bbox) != 4:
                    continue
                
                # Block bbox is always [x1, y1, x2, y2] from OCR
                bx1, by1, bx2, by2 = block_bbox
                
                # Check intersection (with some margin for better matching)
                margin = 5
                if not (bx2 + margin < hl_x1 or bx1 - margin > hl_x2 or by2 + margin < hl_y1 or by1 - margin > hl_y2):
                    highlighted_texts.append(block.get('text', ''))
        
        return ' '.join(highlighted_texts) if highlighted_texts else None
    
    def _get_auto_highlight_explanations(self) -> List[Dict]:
        """Get explanations for auto-highlighted (purple) sections."""
        auto_highlights = self.deps.extracted_content.get('auto_highlights', [])
        return auto_highlights
    
    async def _handle_query_with_tools(self, query: str) -> Optional[str]:
        """Check if query requires tool usage and handle it."""
        query_lower = query.lower()
        
        # 1. Explain yellow highlighted text
        if any(keyword in query_lower for keyword in ['yellow', 'highlighted text', 'highlighted in yellow']):
            yellow_text = self._extract_yellow_highlighted_text()
            if yellow_text:
                context = f"User highlighted this text (in yellow):\n{yellow_text}\n\n"
                context += "Please provide a tutorial-style explanation of this highlighted text in simple, educational terms."
                # Use LLM to explain
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ]
                try:
                    response = ollama.chat(model=self.model_name, messages=messages)
                    return response['message']['content']
                except Exception as e:
                    logger.error(f"Error explaining yellow text: {e}")
                    return f"Found highlighted text: {yellow_text[:200]}...\n\nPlease ask me to explain it and I'll provide a tutorial-style explanation."
            else:
                return "No yellow highlighted text detected on the current screen. Please highlight some text first."
        
        # 2. Explain auto-highlighted (purple) sections
        if any(keyword in query_lower for keyword in ['purple', 'auto-highlight', 'important sections', 'why important']):
            auto_highlights = self._get_auto_highlight_explanations()
            if auto_highlights:
                output = "**Auto-highlighted Important Sections (Purple):**\n\n"
                for i, hl in enumerate(auto_highlights, 1):
                    output += f"{i}. **{hl.get('type', 'Unknown').title()}**\n"
                    output += f"   - Reason: {hl.get('reason', 'N/A')}\n"
                    if hl.get('title'):
                        output += f"   - Title: {hl.get('title')}\n"
                    output += "\n"
                return output
            else:
                return "No auto-highlighted sections found. Important sections, figures, and tables are automatically highlighted in purple as you scroll."
        
        # 3. Table analysis
        if any(keyword in query_lower for keyword in ['table', 'what is this table', 'table showing']):
            tables = self.deps.extracted_content.get('tables', [])
            if not tables:
                return "No tables detected on the current screen."
            
            # Extract table number if specified
            table_num = None
            for word in query_lower.split():
                if word.isdigit():
                    table_num = int(word) - 1
                    break
            
            if table_num is not None and 0 <= table_num < len(tables):
                table = tables[table_num]
                table_content = table.get('content', '')
                context = f"Analyze this table (Table {table_num + 1}):\n\n"
                if table_content:
                    context += f"Table Content:\n{table_content}\n\n"
                else:
                    context += f"Table detected at position {table.get('bbox')}.\n"
                context += "What is this table showing? Explain the data, structure, and key insights."
            else:
                context = f"Analyze the {len(tables)} table(s) detected on screen:\n\n"
                for i, table in enumerate(tables, 1):
                    table_content = table.get('content', '')
                    if table_content:
                        context += f"Table {i} Content:\n{table_content}\n\n"
                    else:
                        context += f"Table {i} at position {table.get('bbox')}\n\n"
                context += "What are these tables showing? Explain the data, structure, and key insights."
            
            # Add structured text context
            structured = self.deps.extracted_content.get('structured_text', {})
            if structured.get('sections'):
                context += f"\n\nContext from paper sections: {structured.get('full_text', '')[:1000]}"
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
            try:
                response = ollama.chat(model=self.model_name, messages=messages)
                return response['message']['content']
            except Exception as e:
                logger.error(f"Error analyzing table: {e}")
                return f"Found {len(tables)} table(s). Please ask me to analyze a specific table number."
        
        # 4. Ablation study analysis
        if any(keyword in query_lower for keyword in ['ablation', 'most detrimental', 'worst ablation']):
            structured = self.deps.extracted_content.get('structured_text', {})
            full_text = structured.get('full_text', '') or self.deps.extracted_content.get('text', '')
            
            if not full_text:
                return "No text extracted. Please process the paper first."
            
            # Search for ablation-related sections
            ablation_keywords = ['ablation', 'without', 'removing', 'removal', 'w/o', 'baseline', 'component']
            sections = structured.get('sections', [])
            ablation_sections = []
            
            for section in sections:
                content_lower = section.get('content', '').lower()
                title_lower = section.get('title', '').lower()
                if any(kw in content_lower or kw in title_lower for kw in ablation_keywords):
                    ablation_sections.append(section)
            
            # Build context with ablation-focused content
            context = "Analyze this paper to identify ablation studies and their impact:\n\n"
            
            if ablation_sections:
                context += "Ablation-related sections found:\n"
                for section in ablation_sections:
                    context += f"\n{section.get('title', 'Unknown')}:\n{section.get('content', '')[:500]}\n"
                context += "\n"
            else:
                # Use full text but focus on results/experiments sections
                context += f"Paper content:\n{full_text[:4000]}\n\n"
            
            context += "Instructions:\n"
            context += "1. Find all ablation studies mentioned (experiments where components are removed)\n"
            context += "2. Identify which ablation study removal causes the MOST DETRIMENTAL impact on performance\n"
            context += "3. Look for metrics like accuracy, F1 score, or other performance measures\n"
            context += "4. Explain why this particular component is critical to the system\n"
            context += "5. Compare the performance drop with other ablation studies"
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
            try:
                response = ollama.chat(model=self.model_name, messages=messages)
                return response['message']['content']
            except Exception as e:
                logger.error(f"Error analyzing ablation: {e}")
                return "Error analyzing ablation studies. Please ensure the paper content has been extracted."
        
        # 5. Diagram replication (Mermaid/Excalidraw) - Extra credit
        if any(keyword in query_lower for keyword in ['diagram', 'block diagram', 'mermaid', 'excalidraw', 'replicate']):
            from utils.diagram_parser import DiagramParser
            
            figures = self.deps.extracted_content.get('figures', [])
            if not figures:
                return "No figures/diagrams detected. Please ensure a diagram is visible on screen."
            
            # Try to parse the largest figure (likely to be a diagram)
            largest_figure = max(figures, key=lambda f: f.get('area', 0))
            
            # Try to use diagram parser if image is available
            if self.deps.current_image is not None and self.vlm is not None:
                try:
                    parser = DiagramParser(self.vlm)
                    diagram_data = parser.detect_diagram_structure(
                        self.deps.current_image,
                        largest_figure.get('bbox', [0, 0, 100, 100])
                    )
                    
                    # Generate Mermaid
                    mermaid_code = parser.to_mermaid(diagram_data)
                    
                    # Generate Excalidraw
                    image_size = self.deps.current_image.size if hasattr(self.deps.current_image, 'size') else (1920, 1080)
                    excalidraw_json = parser.to_excalidraw(diagram_data, image_size)
                    
                    output = "## 📊 Diagram Replication (Extra Credit)\n\n"
                    output += f"**Detected {len(diagram_data.get('nodes', []))} nodes and {len(diagram_data.get('edges', []))} connections**\n\n"
                    output += "### Mermaid Diagram:\n\n"
                    output += mermaid_code + "\n\n"
                    output += "### Excalidraw JSON:\n\n"
                    output += f"```json\n{excalidraw_json[:2000]}...\n```\n"
                    output += "\n*(Full JSON available in response data)*"
                    
                    return output
                except Exception as e:
                    logger.error(f"Diagram parsing error: {e}")
                    # Fall through to LLM-based generation
            
            # Fallback: Use LLM to generate based on description
            structured = self.deps.extracted_content.get('structured_text', {})
            context = f"Analyze the figures/diagrams in this paper:\n\n"
            context += f"Found {len(figures)} figure(s). Largest figure at position {largest_figure.get('bbox')}.\n\n"
            context += f"Paper content context:\n{structured.get('full_text', '')[:2000]}\n\n"
            context += "Identify any block diagrams or system architectures. Generate:\n"
            context += "1. A Mermaid diagram code (```mermaid ... ```)\n"
            context += "2. An Excalidraw JSON representation\n"
            context += "Replicate the bounded block diagram structure with nodes and connections."
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
            try:
                response = ollama.chat(model=self.model_name, messages=messages)
                return response['message']['content']
            except Exception as e:
                logger.error(f"Error replicating diagram: {e}")
                return "Error generating diagram. Please ensure a clear block diagram is visible."
        
        # Search for papers
        if any(keyword in query_lower for keyword in ['search for', 'find papers', 'look up papers']):
            # Extract search terms (simple approach)
            search_terms = query.replace('search for', '').replace('find papers', '').replace('look up papers', '').strip()
            if search_terms:
                results = search_semantic_scholar(search_terms, limit=5)
                
                if not results:
                    return "No papers found for this query."
                
                output = f"Found {len(results)} papers:\n\n"
                for i, paper in enumerate(results, 1):
                    title = paper.get('title', 'Unknown')
                    authors = paper.get('authors', [])
                    author_names = ', '.join([a.get('name', '') for a in authors[:3]])
                    if len(authors) > 3:
                        author_names += ' et al.'
                    year = paper.get('year', 'N/A')
                    citations = paper.get('citationCount', 0)
                    
                    output += f"{i}. **{title}**\n"
                    output += f"   - Authors: {author_names}\n"
                    output += f"   - Year: {year} | Citations: {citations}\n\n"
                
                return output
        
        # Get paper details
        if 'paper details' in query_lower or 'paper info' in query_lower:
            if self.deps.current_paper_id:
                details = get_paper_details(self.deps.current_paper_id)
                
                if details:
                    output = f"**{details.get('title', 'Unknown')}**\n\n"
                    
                    authors = details.get('authors', [])
                    if authors:
                        author_names = ', '.join([a.get('name', '') for a in authors])
                        output += f"**Authors:** {author_names}\n\n"
                    
                    output += f"**Year:** {details.get('year', 'N/A')}\n"
                    output += f"**Venue:** {details.get('venue', 'N/A')}\n"
                    output += f"**Citations:** {details.get('citationCount', 0)}\n"
                    output += f"**References:** {details.get('referenceCount', 0)}\n\n"
                    
                    if details.get('abstract'):
                        output += f"**Abstract:**\n{details['abstract']}\n"
                    
                    return output
        
        # Get citations
        if 'citations' in query_lower or 'citing papers' in query_lower:
            if self.deps.current_paper_id:
                citations = get_paper_citations(self.deps.current_paper_id, limit=10)
                
                if citations:
                    output = f"**Papers citing this work** (showing {len(citations)}):\n\n"
                    for i, citation in enumerate(citations, 1):
                        citing_paper = citation.get('citingPaper', {})
                        title = citing_paper.get('title', 'Unknown')
                        authors = citing_paper.get('authors', [])
                        author_names = ', '.join([a.get('name', '') for a in authors[:2]])
                        if len(authors) > 2:
                            author_names += ' et al.'
                        year = citing_paper.get('year', 'N/A')
                        
                        output += f"{i}. {title}\n"
                        output += f"   - {author_names} ({year})\n\n"
                    
                    return output
        
        # Get references
        if 'references' in query_lower or 'cited papers' in query_lower:
            if self.deps.current_paper_id:
                references = get_paper_references(self.deps.current_paper_id, limit=10)
                
                if references:
                    output = f"**Papers referenced by this work** (showing {len(references)}):\n\n"
                    for i, reference in enumerate(references, 1):
                        cited_paper = reference.get('citedPaper', {})
                        title = cited_paper.get('title', 'Unknown')
                        authors = cited_paper.get('authors', [])
                        author_names = ', '.join([a.get('name', '') for a in authors[:2]])
                        if len(authors) > 2:
                            author_names += ' et al.'
                        year = cited_paper.get('year', 'N/A')
                        
                        output += f"{i}. {title}\n"
                        output += f"   - {author_names} ({year})\n\n"
                    
                    return output
        
        return None
    
    def set_current_paper(self, paper_id: str, title: str):
        """Set the current paper being viewed."""
        self.deps.current_paper_id = paper_id
        self.deps.current_paper_title = title
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.deps.conversation_history = []


# Test the agent
async def test_agent():
    """Test the CUA agent."""
    agent = CUAAgent()
    
    # Test 1: Simple query
    print("Test 1: Simple query")
    response = await agent.process_query(
        "What is attention mechanism in neural networks?"
    )
    print(f"Response: {response[:200]}...\n")
    
    # Test 2: Search functionality
    print("\nTest 2: Search for papers")
    response = await agent.process_query(
        "Search for papers about transformers"
    )
    print(f"Response:\n{response}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())