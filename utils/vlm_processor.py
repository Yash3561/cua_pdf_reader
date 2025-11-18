"""VLM Processor using EasyOCR - Simple and effective."""
import torch
from PIL import Image
import numpy as np
from typing import Dict, List
import easyocr
import cv2
import gc

class VLMProcessor:
    """Processes video frames using EasyOCR."""
    
    def __init__(self):
        """Initialize VLM processor with EasyOCR."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Initializing EasyOCR on {self.device}...")
        
        # Initialize EasyOCR reader
        gpu = True if self.device == "cuda" else False
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
        
        print(f"✅ EasyOCR loaded successfully")
        
        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    def extract_text(self, image: Image.Image, paragraph: bool = True) -> str:
        """Extract text from image using OCR."""
        try:
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Perform OCR with paragraph mode
            results = self.reader.readtext(
                img_array,
                detail=1,
                paragraph=paragraph,
                batch_size=4
            )
            
            if not results:
                return "No text detected in this region"
            
            image_width = image.width if isinstance(image, Image.Image) else img_array.shape[1]
            column_threshold = max(60, image_width * 0.035)
            columns = []
            line_heights = []
            
            for detection in results:
                try:
                    if len(detection) == 3:
                        bbox, text, confidence = detection
                    elif len(detection) == 2:
                        bbox, text = detection
                        confidence = 1.0
                    else:
                        print(f"⚠️ Unexpected detection format: {detection}")
                        continue
                except ValueError:
                    print(f"⚠️ Failed to unpack detection result: {detection}")
                    continue
                
                # Skip low confidence
                if confidence < 0.3:
                    continue
                
                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]
                center_x = sum(xs) / len(xs)
                top_y = min(ys)
                height = max(1.0, max(ys) - min(ys))
                line_heights.append(height)
                
                placed = False
                for column in columns:
                    if abs(center_x - column["center"]) <= column_threshold:
                        column["items"].append({"y": top_y, "text": text, "height": height})
                        column["center"] = (column["center"] * column["count"] + center_x) / (column["count"] + 1)
                        column["count"] += 1
                        placed = True
                        break
                if not placed:
                    columns.append({
                        "center": center_x,
                        "count": 1,
                        "items": [{"y": top_y, "text": text, "height": height}]
                    })
            
            if not columns:
                return "No text detected"
            
            # Determine average line height for spacing heuristics
            avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 20
            line_gap_threshold = max(0.8 * avg_line_height, 15)
            
            # Sort columns from left to right and build text column-wise
            columns = sorted(columns, key=lambda col: col["center"])
            column_texts = []
            for column in columns:
                items = sorted(column["items"], key=lambda item: item["y"])
                prev_y = -1e6
                lines = []
                for item in items:
                    if item["y"] - prev_y > line_gap_threshold and lines:
                        lines.append('\n')
                    lines.append(item["text"])
                    prev_y = item["y"]
                column_texts.append(' '.join(lines))
            
            joined_columns = '\n\n'.join(column_texts)
                
            full_text = joined_columns.strip()
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return full_text if full_text else "No text detected"
        
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return f"Error during text extraction: {str(e)}"
    
    def extract_text_regions(self, image: Image.Image, regions: List[tuple]) -> List[str]:
        """Extract text from specific regions of the image."""
        texts = []
        img_array = np.array(image)
        
        for region in regions:
            x, y, w, h = region
            roi = img_array[y:y+h, x:x+w]
            roi_img = Image.fromarray(roi)
            text = self.extract_text(roi_img, paragraph=True)
            texts.append(text)
        
        return texts
    
    def detect_tables(self, image: Image.Image) -> List[Dict]:
        """Detect potential table regions using line detection."""
        try:
            img = np.array(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            
            # Threshold
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine
            table_mask = cv2.add(detect_horizontal, detect_vertical)
            
            # Find contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 5000:  # Minimum table size
                    x, y, w, h = cv2.boundingRect(contour)
                    tables.append({
                        "index": i,
                        "bbox": [x, y, w, h],
                        "area": int(area)
                    })
            
            return tables
        
        except Exception as e:
            print(f"⚠️ Table detection error: {e}")
            return []
    
    def detect_figures(self, image: Image.Image) -> List[Dict]:
        """Detect figures/images using simple heuristics."""
        try:
            img = np.array(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            
            # Edge detection
            edges = cv2.Canny(gray, 30, 100)
            
            # Dilate to connect nearby edges
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            figures = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                # Filter by area (figures are usually medium to large)
                if 10000 < area < 1000000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    # Reasonable aspect ratios for figures
                    if 0.3 < aspect_ratio < 4.0:
                        figures.append({
                            "index": i,
                            "bbox": [x, y, w, h],
                            "area": int(area),
                            "aspect_ratio": round(aspect_ratio, 2)
                        })
            
            return figures
        
        except Exception as e:
            print(f"⚠️ Figure detection error: {e}")
            return []
    
    def detect_highlights(self, image: Image.Image) -> Dict[str, List]:
        """Detect yellow and purple highlights in the image."""
        try:
            img = np.array(image)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Yellow highlight detection (broader range)
            yellow_lower = np.array([15, 50, 100])
            yellow_upper = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            
            # Purple/Magenta highlight detection
            purple_lower = np.array([125, 50, 50])
            purple_upper = np.array([165, 255, 255])
            purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
            
            # Find contours
            yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            yellow_highlights = []
            for contour in yellow_contours:
                if cv2.contourArea(contour) > 200:  # Minimum highlight size
                    x, y, w, h = cv2.boundingRect(contour)
                    yellow_highlights.append({"bbox": [x, y, w, h]})
            
            purple_highlights = []
            for contour in purple_contours:
                if cv2.contourArea(contour) > 200:
                    x, y, w, h = cv2.boundingRect(contour)
                    purple_highlights.append({"bbox": [x, y, w, h]})
            
            return {
                "yellow": yellow_highlights,
                "purple": purple_highlights
            }
        
        except Exception as e:
            print(f"⚠️ Highlight detection error: {e}")
            return {"yellow": [], "purple": []}
    
    def split_into_regions(self, image: Image.Image, overlap: int = 100) -> List[tuple]:
        """
        Split large image into overlapping regions for better OCR.
        Returns list of (x, y, w, h) tuples.
        """
        width, height = image.size
        region_height = 800  # Process in chunks
        regions = []
        
        y = 0
        while y < height:
            h = min(region_height, height - y)
            regions.append((0, y, width, h))
            y += region_height - overlap  # Overlap to avoid cutting text
        
        return regions
    
    def process_frame(self, frame: np.ndarray, full_page: bool = False) -> Dict:
        """
        Process a single video frame.
        
        Args:
            frame: Input image
            full_page: If True, split into regions for better OCR
        """
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        
        # Resize if too large (save GPU memory)
        max_size = 1920
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        # Extract text
        if full_page and image.size[1] > 1200:
            # Split into regions for large images
            print("📄 Splitting large image into regions...")
            regions = self.split_into_regions(image)
            region_texts = self.extract_text_regions(image, regions)
            text = '\n\n'.join(region_texts)
        else:
            # Process whole image
            text = self.extract_text(image)
        
        # Detect other elements
        tables = self.detect_tables(image)
        figures = self.detect_figures(image)
        highlights = self.detect_highlights(image)
        
        return {
            "text": text,
            "tables": tables,
            "figures": figures,
            "highlights": highlights,
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
    print("VLM Processor initialized successfully!")
    processor.cleanup()