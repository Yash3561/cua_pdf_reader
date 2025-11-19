"""Frame history manager for storing and retrieving past frames and their content."""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
from collections import deque
import numpy as np
from PIL import Image
from .frame_stitcher import FrameStitcher


class FrameHistory:
    """Manages history of captured frames and their extracted content."""
    
    def __init__(self, max_frames: int = 50):
        """
        Initialize frame history.
        
        Args:
            max_frames: Maximum number of frames to keep in history
        """
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)  # (timestamp, frame, extracted_content)
        self.frame_lock = None  # Will be set if needed for threading
        self.stitcher = FrameStitcher(overlap_threshold=0.2)  # Initialize stitcher
    
    def add_frame(
        self,
        frame: np.ndarray,
        extracted_content: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Add a frame to history.
        
        Args:
            frame: Frame as numpy array
            extracted_content: Extracted content from VLM (optional)
            timestamp: Timestamp (defaults to current time)
        
        Returns:
            Frame index in history
        """
        if timestamp is None:
            timestamp = time.time()
        
        frame_data = {
            "timestamp": timestamp,
            "frame": frame.copy() if isinstance(frame, np.ndarray) else frame,
            "extracted_content": extracted_content or {},
            "frame_index": len(self.frames)
        }
        
        self.frames.append(frame_data)
        return len(self.frames) - 1
    
    def get_frame(self, index: int) -> Optional[Dict]:
        """Get frame by index (0 = oldest, -1 = newest)."""
        if not self.frames:
            return None
        if index < 0:
            index = len(self.frames) + index
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None
    
    def get_latest_frame(self) -> Optional[Dict]:
        """Get the most recent frame."""
        return self.frames[-1] if self.frames else None
    
    def get_recent_frames(self, count: int = 10) -> List[Dict]:
        """Get the most recent N frames."""
        return list(self.frames)[-count:] if self.frames else []
    
    def search_content(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Search through frame history for content matching query.
        
        Args:
            query: Search query (keywords)
            max_results: Maximum number of results
        
        Returns:
            List of (frame_data, relevance_score) tuples
        """
        if not self.frames:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for frame_data in self.frames:
            content = frame_data.get("extracted_content", {})
            text = content.get("text", "") or content.get("structured_text", {}).get("full_text", "")
            text_lower = text.lower()
            
            # Simple relevance scoring
            score = 0
            for word in query_words:
                if word in text_lower:
                    score += text_lower.count(word)
            
            if score > 0:
                results.append((frame_data, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def get_all_text(self, max_chars: int = 10000) -> str:
        """
        Get all extracted text from frame history.
        
        Args:
            max_chars: Maximum characters to return
        
        Returns:
            Combined text from all frames
        """
        all_texts = []
        total_chars = 0
        
        for frame_data in self.frames:
            content = frame_data.get("extracted_content", {})
            text = content.get("text", "") or content.get("structured_text", {}).get("full_text", "")
            if text:
                if total_chars + len(text) > max_chars:
                    remaining = max_chars - total_chars
                    all_texts.append(text[:remaining])
                    break
                all_texts.append(text)
                total_chars += len(text)
        
        return "\n\n--- Frame Break ---\n\n".join(all_texts)
    
    def get_frame_count(self) -> int:
        """Get number of frames in history."""
        return len(self.frames)
    
    def clear(self):
        """Clear all frame history."""
        self.frames.clear()
        self.stitcher = FrameStitcher(overlap_threshold=0.2)  # Reset stitcher
    
    def stitch_all_frames(self) -> Dict:
        """
        Stitch all frames together to reconstruct the full document.
        
        Returns:
            Dict with stitched text and metadata
        """
        if not self.frames:
            return {"text": "", "frames_used": 0, "stitched_text": ""}
        
        # Prepare frames and contents for stitching
        frames_list = []
        contents_list = []
        
        for frame_data in self.frames:
            frames_list.append({"frame": frame_data.get("frame")})
            contents_list.append(frame_data.get("extracted_content", {}))
        
        # Stitch frames
        result = self.stitcher.stitch_frames(frames_list, contents_list)
        return result
    
    def get_stitched_document(self, max_chars: int = 50000) -> str:
        """
        Get the full stitched document text.
        
        Args:
            max_chars: Maximum characters to return
        
        Returns:
            Full stitched document text
        """
        stitched = self.stitch_all_frames()
        text = stitched.get("stitched_text", "")
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

