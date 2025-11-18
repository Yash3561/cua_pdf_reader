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
    
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.deps = CUADependencies()
        self.system_prompt = SYSTEM_PROMPT
    
    async def process_query(
        self,
        query: str,
        extracted_content: Optional[Dict] = None
    ) -> str:
        """
        Process a user query using the agent.
        
        Args:
            query: User's question
            extracted_content: Content extracted from screen by VLM
        
        Returns:
            Agent's response
        """
        # Update extracted content if provided
        if extracted_content:
            self.deps.extracted_content = extracted_content
        
        # Check if query involves tool usage
        response = await self._handle_query_with_tools(query)
        if response:
            return response
        
        # Build context message for regular query
        context_parts = []
        
        if self.deps.current_paper_title:
            context_parts.append(f"Current paper: {self.deps.current_paper_title}")
        
        if self.deps.extracted_content.get('text'):
            text = self.deps.extracted_content['text']
            # Truncate if too long
            if len(text) > 2000:
                text = text[:2000] + "..."
            context_parts.append(f"Extracted text from screen:\n{text}")
        
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
    
    async def _handle_query_with_tools(self, query: str) -> Optional[str]:
        """Check if query requires tool usage and handle it."""
        query_lower = query.lower()
        
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