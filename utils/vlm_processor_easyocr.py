"""VLM Processor using EasyOCR - More stable alternative."""
import torch
from PIL import Image
import numpy as np
from typing import Dict, List
import easyocr
import gc

class VLMProcessor:
    """Processes video frames using EasyOCR."""
    
    def __init__(self):
        """Initialize VLM processor with EasyOCR."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Initializing EasyOCR on {self.device}...")
        
        # Initialize EasyOCR reader
        gpu = True if self.device == "cuda" else False
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        
        print(f"✅ EasyOCR loaded successfully")
        
        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    def extract_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Perform OCR
            results = self.reader.readtext(img_array)
            
            # Combine text results
            text = " ".join([result[1] for result in results])
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return text
        
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return ""
    
    def detect_tables(self, image: Image.Image) -> List[Dict]:
        """Detect potential table regions (simplified)."""
        # For now, return empty - will enhance later
        return []
    
    def detect_figures(self, image: Image.Image) -> List[Dict]:
        """Detect figures (simplified)."""
        # For now, return empty - will enhance later
        return []
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single video frame."""
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        
        # Resize if too large (save GPU memory)
        max_size = 1280
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        # Extract content
        text = self.extract_text(image)
        tables = self.detect_tables(image)
        figures = self.detect_figures(image)
        
        return {
            "text": text,
            "tables": tables,
            "figures": figures,
            "image_size": image.size
        }
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

# Test
if __name__ == "__main__":
    processor = VLMProcessor()
    print("EasyOCR Processor initialized successfully!")
    
    # Test with a blank image
    test_image = Image.new('RGB', (800, 600), color='white')
    result = processor.process_frame(test_image)
    print(f"Test result: {result}")
    processor.cleanup()