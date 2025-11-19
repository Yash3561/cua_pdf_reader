"""Frame stitching to reconstruct full PDF from multiple frames."""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import time
from collections import deque


class FrameStitcher:
    """Stitches multiple frames together to reconstruct a continuous document."""
    
    def __init__(self, overlap_threshold: float = 0.3):
        """
        Initialize frame stitcher.
        
        Args:
            overlap_threshold: Minimum overlap ratio to consider frames as consecutive (0.0-1.0)
        """
        self.overlap_threshold = overlap_threshold
        self.stitched_frames = []  # List of stitched frame data
        self.last_frame_hash = None  # Hash of last frame to detect duplicates
    
    def detect_overlap(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, int]:
        """
        Detect overlap between two frames using feature matching.
        
        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)
        
        Returns:
            (overlap_ratio, vertical_offset) where:
            - overlap_ratio: 0.0-1.0, how much frames overlap
            - vertical_offset: pixels frame2 is offset from frame1 (positive = scrolled down)
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
            
            # Use ORB detector for feature matching
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                # Fallback: simple vertical alignment check
                return self._simple_overlap_check(gray1, gray2)
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 10:
                return self._simple_overlap_check(gray1, gray2)
            
            # Calculate vertical offset from matches
            offsets = []
            for match in matches[:50]:  # Use top 50 matches
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                offset = pt2[1] - pt1[1]  # Y difference
                offsets.append(offset)
            
            # Use median offset (more robust than mean)
            vertical_offset = int(np.median(offsets))
            
            # Calculate overlap ratio
            h1, h2 = gray1.shape[0], gray2.shape[0]
            if vertical_offset > 0:  # Scrolled down
                overlap_height = max(0, h1 - vertical_offset)
            else:  # Scrolled up (unlikely but possible)
                overlap_height = max(0, h2 + vertical_offset)
            
            overlap_ratio = overlap_height / min(h1, h2) if min(h1, h2) > 0 else 0.0
            
            return overlap_ratio, vertical_offset
        
        except Exception as e:
            print(f"⚠️ Overlap detection error: {e}")
            return self._simple_overlap_check(gray1, gray2)
    
    def _simple_overlap_check(self, gray1: np.ndarray, gray2: np.ndarray) -> Tuple[float, int]:
        """Simple overlap check using template matching."""
        try:
            h1, w1 = gray1.shape
            h2, w2 = gray2.shape
            
            # Check if frames are similar size (likely same document)
            if abs(h1 - h2) > h1 * 0.2:  # More than 20% height difference
                return 0.0, 0
            
            # Try template matching at bottom of frame1 with top of frame2
            template_height = min(200, h1 // 4, h2 // 4)  # Use 25% of height or 200px
            template = gray1[-template_height:, :]  # Bottom of frame1
            search_area = gray2[:template_height * 2, :]  # Top of frame2
            
            if template.shape[0] > search_area.shape[0] or template.shape[1] > search_area.shape[1]:
                return 0.0, 0
            
            result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.5:  # Good match
                # Calculate offset
                vertical_offset = h1 - (max_loc[1] + template_height)
                overlap_ratio = template_height / min(h1, h2)
                return overlap_ratio, vertical_offset
            
            return 0.0, 0
        
        except Exception:
            return 0.0, 0
    
    def is_duplicate(self, frame: np.ndarray, threshold: float = 0.95) -> bool:
        """
        Check if frame is duplicate of last frame.
        
        Args:
            frame: Frame to check
            threshold: Similarity threshold (0.0-1.0)
        
        Returns:
            True if frame is likely a duplicate
        """
        if self.last_frame_hash is None:
            self.last_frame_hash = self._hash_frame(frame)
            return False
        
        current_hash = self._hash_frame(frame)
        similarity = self._hash_similarity(self.last_frame_hash, current_hash)
        
        if similarity > threshold:
            return True
        
        self.last_frame_hash = current_hash
        return False
    
    def _hash_frame(self, frame: np.ndarray) -> np.ndarray:
        """Create a hash of frame for duplicate detection."""
        # Resize to small size for hashing
        small = cv2.resize(frame, (64, 64)) if len(frame.shape) == 3 else cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        return gray.flatten()
    
    def _hash_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Calculate similarity between two hashes."""
        if hash1.shape != hash2.shape:
            return 0.0
        diff = np.abs(hash1.astype(float) - hash2.astype(float))
        similarity = 1.0 - (np.mean(diff) / 255.0)
        return similarity
    
    def stitch_frames(
        self,
        frames: List[Dict],
        extracted_contents: List[Dict]
    ) -> Dict:
        """
        Stitch multiple frames together into a continuous document.
        
        Args:
            frames: List of frame data dicts with 'frame' key
            extracted_contents: List of extracted content dicts
        
        Returns:
            Stitched document with combined text and metadata
        """
        if not frames or len(frames) != len(extracted_contents):
            return {"text": "", "frames_used": 0, "stitched_text": ""}
        
        # Remove duplicates
        unique_frames = []
        unique_contents = []
        for i, frame_data in enumerate(frames):
            frame = frame_data.get("frame")
            if frame is None:
                continue
            
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            
            if not self.is_duplicate(frame):
                unique_frames.append(frame_data)
                unique_contents.append(extracted_contents[i])
        
        if len(unique_frames) < 2:
            # Not enough frames to stitch, just combine text
            all_texts = [content.get("text", "") or content.get("structured_text", {}).get("full_text", "")
                        for content in unique_contents]
            combined_text = "\n\n".join([t for t in all_texts if t])
            return {
                "text": combined_text,
                "frames_used": len(unique_frames),
                "stitched_text": combined_text,
                "method": "text_only"
            }
        
        # Detect overlaps and order frames
        ordered_frames = []
        ordered_contents = []
        
        # Start with first frame
        ordered_frames.append(unique_frames[0])
        ordered_contents.append(unique_contents[0])
        
        # Build chain of overlapping frames
        used_indices = {0}
        current_frame = unique_frames[0]["frame"]
        if isinstance(current_frame, Image.Image):
            current_frame = np.array(current_frame)
        
        while len(ordered_frames) < len(unique_frames):
            best_match_idx = None
            best_overlap = 0.0
            best_offset = 0
            
            for i, frame_data in enumerate(unique_frames):
                if i in used_indices:
                    continue
                
                frame = frame_data.get("frame")
                if frame is None:
                    continue
                
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                
                overlap, offset = self.detect_overlap(current_frame, frame)
                
                if overlap > best_overlap and overlap > self.overlap_threshold:
                    best_overlap = overlap
                    best_match_idx = i
                    best_offset = offset
            
            if best_match_idx is None:
                # No good match found, add remaining frames in order
                for i, frame_data in enumerate(unique_frames):
                    if i not in used_indices:
                        ordered_frames.append(frame_data)
                        ordered_contents.append(unique_contents[i])
                        used_indices.add(i)
                break
            
            # Add best match
            ordered_frames.append(unique_frames[best_match_idx])
            ordered_contents.append(unique_contents[best_match_idx])
            used_indices.add(best_match_idx)
            current_frame = unique_frames[best_match_idx]["frame"]
            if isinstance(current_frame, Image.Image):
                current_frame = np.array(current_frame)
        
        # Combine text from ordered frames, removing duplicates
        stitched_texts = []
        seen_text_chunks = set()
        
        for content in ordered_contents:
            text = content.get("text", "") or content.get("structured_text", {}).get("full_text", "")
            if not text:
                continue
            
            # Simple deduplication: check if text chunk is too similar to previous
            text_chunk = text[:200]  # First 200 chars as signature
            if text_chunk not in seen_text_chunks:
                stitched_texts.append(text)
                seen_text_chunks.add(text_chunk)
        
        stitched_text = "\n\n".join(stitched_texts)
        
        return {
            "text": stitched_text,
            "frames_used": len(ordered_frames),
            "stitched_text": stitched_text,
            "method": "stitched",
            "total_frames": len(frames),
            "unique_frames": len(unique_frames)
        }

