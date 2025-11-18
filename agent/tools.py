"""Tools for the CUA agent."""
import requests
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def search_semantic_scholar(query: str, limit: int = 5) -> List[Dict]:
    """
    Search for papers on Semantic Scholar.
    
    Args:
        query: Search query string
        limit: Maximum number of results
    
    Returns:
        List of paper metadata
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,citationCount,url,openAccessPdf"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", [])
    
    except Exception as e:
        logger.error(f"Semantic Scholar search error: {e}")
        return []


def get_paper_details(paper_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific paper.
    
    Args:
        paper_id: Semantic Scholar paper ID or ArXiv ID
    
    Returns:
        Paper details
    """
    try:
        # Handle ArXiv IDs
        if not paper_id.startswith("ARXIV:") and "arxiv" in paper_id.lower():
            # Extract arxiv ID from URL or format it
            arxiv_id = paper_id.split("/")[-1].replace(".pdf", "")
            paper_id = f"ARXIV:{arxiv_id}"
        
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {
            "fields": "title,authors,year,abstract,citationCount,referenceCount,venue,citations,references"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Paper details error: {e}")
        return None


def get_paper_citations(paper_id: str, limit: int = 10) -> List[Dict]:
    """
    Get citations for a paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum number of citations
    
    Returns:
        List of citing papers
    """
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        params = {
            "fields": "title,authors,year",
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", [])
    
    except Exception as e:
        logger.error(f"Citations error: {e}")
        return []


def get_paper_references(paper_id: str, limit: int = 10) -> List[Dict]:
    """
    Get references from a paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum number of references
    
    Returns:
        List of referenced papers
    """
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        params = {
            "fields": "title,authors,year",
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", [])
    
    except Exception as e:
        logger.error(f"References error: {e}")
        return []


# Test
if __name__ == "__main__":
    # Test search
    results = search_semantic_scholar("attention is all you need", limit=3)
    print(f"Found {len(results)} papers")
    if results:
        print(f"First paper: {results[0].get('title')}")
    
    # Test paper details
    details = get_paper_details("ARXIV:2205.14756")
    if details:
        print(f"\nPaper details: {details.get('title')}")