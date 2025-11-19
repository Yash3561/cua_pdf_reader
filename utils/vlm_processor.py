"""VLM Processor using EasyOCR - Simple and effective."""
import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import easyocr
import cv2
import gc
import re

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
        """Extract text and return a formatted string (legacy helper)."""
        structured = self.extract_structured_text(image, paragraph=paragraph)
        full_text = structured.get("full_text")
        if full_text:
            return full_text
        return "No text detected"

    def extract_structured_text(
        self,
        image: Image.Image,
        paragraph: bool = True,
        figures: Optional[List[Dict]] = None,
        tables: Optional[List[Dict]] = None
    ) -> Dict:
        """Extract text with simple layout understanding (title, abstract, sections)."""
        try:
            img_array = np.array(image)
            results = self.reader.readtext(
                img_array,
                detail=1,
                paragraph=paragraph,
                batch_size=4
            )
            
            if not results:
                return {
                    "full_text": "No text detected",
                    "header": "",
                    "abstract": "",
                    "sections": [],
                    "captions": [],
                    "raw_blocks": []
                }
            
            structured = self._structure_ocr_results(
                results=results,
                image_size=image.size,
                figures=figures or [],
                tables=tables or []
            )
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return structured
        
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return {
                "full_text": f"Error during text extraction: {str(e)}",
                "header": "",
                "abstract": "",
                "sections": [],
                "captions": [],
                "raw_blocks": []
            }

    def _structure_ocr_results(
        self,
        results: List,
        image_size: Tuple[int, int],
        figures: List[Dict],
        tables: List[Dict]
    ) -> Dict:
        """Convert EasyOCR results into structured text sections."""
        width, height = image_size
        header_threshold = height * 0.18
        footer_threshold = height * 0.92
        figure_regions = self._expand_regions(figures)
        table_regions = self._expand_regions(tables)
        
        blocks = []
        confidences = []
        
        # Browser UI patterns to filter out
        browser_ui_patterns = [
            r'^\d+\s*/\s*\d+$',  # Page numbers like "3 / 43"
            r'^\d+%$',  # Zoom percentages like "100%"
            r'^[A-Z]:\\',  # Windows file paths like "C:\Users\..."
            r'^[A-Z]:/',  # Windows paths with forward slash
            r'^File\s+\d+',  # "File 6"
            r'^Verify',  # "Verify it's you"
            r'^Gemini',  # Browser names
            r'^\d+\s*\(',  # Numbers with parentheses at start
            r'^pdf\s*$',  # Just "pdf"
            r'^\s*[|~:]\s*$',  # Just symbols
            r'^\s*[0-9]+\s*$',  # Just numbers
        ]
        
        for detection in results:
            try:
                if len(detection) == 3:
                    bbox, text, confidence = detection
                elif len(detection) == 2:
                    bbox, text = detection
                    confidence = 1.0
                else:
                    continue
            except ValueError:
                continue
            
            if confidence < 0.3:
                continue
            
            clean_text = text.strip()
            if not clean_text:
                continue
            
            # Filter out browser UI elements
            import re
            is_browser_ui = False
            for pattern in browser_ui_patterns:
                if re.match(pattern, clean_text, re.IGNORECASE):
                    is_browser_ui = True
                    break
            
            # Also filter if text is very short and contains mostly symbols
            if len(clean_text) <= 3 and not clean_text.isalnum():
                is_browser_ui = True
            
            # Calculate bbox coordinates
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Filter if text is in top/bottom 5% of image (likely browser UI)
            if y_max < height * 0.05 or y_min > height * 0.95:
                is_browser_ui = True
            
            if is_browser_ui:
                continue
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            block_height = max(1.0, y_max - y_min)
            
            block = {
                "text": clean_text,
                "bbox": [x_min, y_min, x_max, y_max],
                "center": (center_x, center_y),
                "height": block_height,
                "width": max(1.0, x_max - x_min)
            }
            
            block_type = self._classify_block(
                block=block,
                header_threshold=header_threshold,
                footer_threshold=footer_threshold,
                figure_regions=figure_regions,
                table_regions=table_regions
            )
            block["type"] = block_type
            blocks.append(block)
            confidences.append(confidence)
        
        if not blocks:
            return {
                "full_text": "No text detected",
                "header": "",
                "abstract": "",
                "sections": [],
                "captions": [],
                "raw_blocks": []
            }
        
        body_blocks = [b for b in blocks if b["type"] in ("body", "table_text")]
        if not body_blocks:
            body_blocks = [b for b in blocks if b["type"] not in ("header", "footer")]
        
        avg_line_height = np.median([b["height"] for b in body_blocks]) if body_blocks else 18.0
        line_gap_threshold = max(0.9 * avg_line_height, 15)
        
        # IMPROVED COLUMN DETECTION: Direct gap-based approach
        # Step 1: Sort all blocks by X-coordinate
        sorted_by_x = sorted(body_blocks, key=lambda b: b["center"][0])
        
        if not sorted_by_x:
            ordered_body = []
        else:
            # Step 2: Find the largest gap in X positions (column separator)
            # Calculate all gaps between consecutive blocks
            gaps = []
            for i in range(len(sorted_by_x) - 1):
                left_block = sorted_by_x[i]
                right_block = sorted_by_x[i + 1]
                # Gap = start of right block - end of left block
                gap = right_block["bbox"][0] - left_block["bbox"][2]
                gap_center = (left_block["bbox"][2] + right_block["bbox"][0]) / 2
                gaps.append((gap, gap_center, i))
            
            # Sort gaps by size (largest first)
            gaps.sort(key=lambda g: g[0], reverse=True)
            
            # Find the largest gap that's likely a column separator
            # For 2-column papers, this should be near the middle of the page
            gap_threshold = max(80, width * 0.03)  # Minimum gap size
            column_separator = None
            
            for gap_size, gap_center, gap_idx in gaps:
                if gap_size > gap_threshold:
                    # Check if this gap is in a reasonable position (not at edges)
                    # For 2-column papers, separator should be around 40-60% of width
                    if width * 0.35 < gap_center < width * 0.65:
                        column_separator = gap_center
                        break
            
            # If no good separator found, try the largest gap regardless of position
            if column_separator is None and gaps:
                largest_gap_size, largest_gap_center, _ = gaps[0]
                if largest_gap_size > gap_threshold:
                    column_separator = largest_gap_center
            
            # Step 3: Split blocks into columns
            if column_separator is not None:
                # Two-column layout
                left_blocks = [b for b in sorted_by_x if b["center"][0] < column_separator]
                right_blocks = [b for b in sorted_by_x if b["center"][0] >= column_separator]
                
                # Verify both columns have reasonable content
                if left_blocks and right_blocks and len(left_blocks) > 2 and len(right_blocks) > 2:
                    # Calculate actual column bounds
                    left_x_min = min(b["bbox"][0] for b in left_blocks)
                    left_x_max = max(b["bbox"][2] for b in left_blocks)
                    right_x_min = min(b["bbox"][0] for b in right_blocks)
                    right_x_max = max(b["bbox"][2] for b in right_blocks)
                    
                    # Verify gap is still significant
                    actual_gap = right_x_min - left_x_max
                    if actual_gap > gap_threshold:
                        columns = [
                            {
                                "x_min": left_x_min,
                                "x_max": left_x_max,
                                "items": left_blocks
                            },
                            {
                                "x_min": right_x_min,
                                "x_max": right_x_max,
                                "items": right_blocks
                            }
                        ]
                    else:
                        # Gap too small, treat as single column
                        columns = [{"x_min": 0, "x_max": width, "items": sorted_by_x}]
                else:
                    # Not enough blocks in one column, treat as single column
                    columns = [{"x_min": 0, "x_max": width, "items": sorted_by_x}]
            else:
                # No significant gap found, treat as single column
                columns = [{"x_min": 0, "x_max": width, "items": sorted_by_x}]
            
            # Step 4: Sort items within each column by Y-coordinate (top to bottom)
            ordered_body = []
            for column in sorted(columns, key=lambda c: c["x_min"]):  # Process columns left to right
                sorted_items = sorted(column["items"], key=lambda item: item["bbox"][1])  # Sort by Y
                
                prev_y = None
                for item in sorted_items:
                    if prev_y is None:
                        item["new_paragraph"] = True
                    else:
                        item["new_paragraph"] = (item["bbox"][1] - prev_y) > line_gap_threshold
                    prev_y = item["bbox"][3]  # Use bottom of bbox
                    ordered_body.append(item)
        
        abstract_lines, body_sequence = self._split_abstract(ordered_body, height)
        sections = self._build_sections(body_sequence)
        
        header_text = self._join_blocks([b for b in blocks if b["type"] == "header"])
        footer_text = self._join_blocks([b for b in blocks if b["type"] == "footer"])
        caption_texts = self._group_captions(blocks, figure_regions)
        
        full_parts = []
        if header_text:
            full_parts.append(header_text)
        if abstract_lines:
            full_parts.append(f"Abstract\n{self._tokens_to_paragraph(abstract_lines)}")
        for section in sections:
            content = section.get("content", "")
            if content:
                full_parts.append(f"{section['title']}\n{content}")
        if caption_texts:
            full_parts.append("Figure Captions\n" + "\n".join(caption_texts))
        if footer_text:
            full_parts.append(footer_text)
        
        full_text = "\n\n".join([part.strip() for part in full_parts if part.strip()])
        
        return {
            "full_text": full_text if full_text else "No text detected",
            "header": header_text,
            "footer": footer_text,
            "abstract": self._tokens_to_paragraph(abstract_lines),
            "sections": sections,
            "captions": caption_texts,
            "raw_blocks": ordered_body
        }

    def _expand_regions(self, items: List[Dict], padding: int = 12) -> List[Tuple[float, float, float, float]]:
        regions = []
        for item in items:
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            regions.append((
                x - padding,
                y - padding,
                x + w + padding,
                y + h + padding
            ))
        return regions

    def _classify_block(
        self,
        block: Dict,
        header_threshold: float,
        footer_threshold: float,
        figure_regions: List[Tuple[float, float, float, float]],
        table_regions: List[Tuple[float, float, float, float]]
    ) -> str:
        center_y = block["center"][1]
        bbox = block["bbox"]
        if center_y < header_threshold:
            return "header"
        if center_y > footer_threshold:
            return "footer"
        # Only classify as caption if text is BELOW the figure (captions are usually below)
        # Use smaller margin to avoid over-matching
        for region in figure_regions:
            rx1, ry1, rx2, ry2 = region
            # Check if block is below the figure and horizontally aligned
            block_y = bbox[1]  # Top of block
            if block_y > ry2 and block_y < ry2 + 100:  # Within 100px below figure
                if self._intersects_regions(bbox, [region], extra_margin=5):
                    return "caption"
        if self._intersects_regions(bbox, table_regions, extra_margin=6):
            return "table_text"
        return "body"

    def _intersects_regions(
        self,
        bbox: List[float],
        regions: List[Tuple[float, float, float, float]],
        extra_margin: int = 0
    ) -> bool:
        x1, y1, x2, y2 = bbox
        for rx1, ry1, rx2, ry2 in regions:
            if x2 + extra_margin < rx1 or x1 - extra_margin > rx2:
                continue
            if y2 + extra_margin < ry1 or y1 - extra_margin > ry2:
                continue
            return True
        return False

    def _split_abstract(self, ordered_blocks: List[Dict], image_height: int):
        abstract_lines = []
        body_sequence = []
        abstract_started = False
        abstract_done = False
        for block in ordered_blocks:
            text_lower = block["text"].lower()
            if not abstract_started and "abstract" in text_lower and block["bbox"][1] < image_height * 0.45:
                abstract_started = True
            
            if abstract_started and not abstract_done:
                if self._is_section_heading(block["text"]) and not text_lower.startswith("abstract"):
                    abstract_done = True
                    body_sequence.append(block)
                    continue
                abstract_lines.append(block["text"])
                continue
            
            body_sequence.append(block)
        return abstract_lines, body_sequence

    def _build_sections(self, body_blocks: List[Dict]) -> List[Dict]:
        sections = []
        current_section = None
        for block in body_blocks:
            text = block["text"].strip()
            if not text:
                continue
            if self._is_section_heading(text):
                heading = self._normalize_heading(text)
                current_section = {
                    "title": heading,
                    "content_tokens": []
                }
                sections.append(current_section)
                continue
            if current_section is None:
                current_section = {"title": "Main Content", "content_tokens": []}
                sections.append(current_section)
            if block.get("new_paragraph") and current_section["content_tokens"]:
                current_section["content_tokens"].append("\n")
            current_section["content_tokens"].append(text)
        
        for section in sections:
            section["content"] = self._tokens_to_paragraph(section.pop("content_tokens", []))
        
        return sections

    def _is_section_heading(self, text: str) -> bool:
        stripped = text.strip(" :-—")
        if len(stripped) < 3:
            return False
        if stripped.lower().startswith("fig"):
            return False
        word_count = len(stripped.split())
        uppercase_ratio = self._uppercase_ratio(stripped)
        has_number_prefix = bool(re.match(r"^(\(?\d+(\.\d+)*\)?|[IVXLC]+\.)\s+", stripped))
        heading_keywords = [
            "abstract",
            "introduction",
            "related work",
            "background",
            "method",
            "methods",
            "approach",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "appendix"
        ]
        normalized = stripped.lower()
        if normalized in heading_keywords:
            return True
        if has_number_prefix and word_count <= 12:
            return True
        if word_count <= 8 and uppercase_ratio > 0.6:
            return True
        return False

    def _normalize_heading(self, text: str) -> str:
        stripped = text.strip()
        stripped = re.sub(r"^[\(\s]*(\d+(\.\d+)*|[IVXLC]+\.)\s+", "", stripped)
        stripped = stripped.replace("—", "-").strip(" :-")
        return stripped if stripped else text.strip()

    def _uppercase_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        uppercase = [c for c in letters if c.isupper()]
        return len(uppercase) / len(letters)

    def _join_blocks(self, blocks: List[Dict]) -> str:
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
        texts = [block["text"] for block in sorted_blocks]
        return " ".join(texts).strip()

    def _tokens_to_paragraph(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        paragraphs = []
        current = []
        for token in tokens:
            if token == "\n":
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            current.append(token)
        if current:
            paragraphs.append(" ".join(current))
        return "\n".join([p.strip() for p in paragraphs if p.strip()])

    def _group_captions(
        self,
        blocks: List[Dict],
        figure_regions: List[Tuple[float, float, float, float]]
    ) -> List[str]:
        caption_blocks = [b for b in blocks if b["type"] == "caption"]
        if not caption_blocks:
            return []
        caption_blocks = sorted(caption_blocks, key=lambda b: b["bbox"][1])
        grouped = []
        
        # Group nearby caption blocks together (they might be split across lines)
        caption_groups = []
        current_group = []
        prev_y = None
        
        for block in caption_blocks:
            if prev_y is None:
                current_group = [block]
            else:
                # If blocks are close vertically (within 2x line height), group them
                y_gap = block["bbox"][1] - prev_y
                if y_gap < 50:  # Close vertically, likely same caption
                    current_group.append(block)
                else:
                    if current_group:
                        caption_groups.append(current_group)
                    current_group = [block]
            prev_y = block["bbox"][3]
        
        if current_group:
            caption_groups.append(current_group)
        
        # Match caption groups to figures
        for group in caption_groups:
            # Combine text from all blocks in group
            group_text = " ".join([b["text"] for b in group])
            
            # Find the best matching figure (closest vertically)
            best_match = None
            best_distance = float('inf')
            
            # Get the center Y of the caption group
            group_y = np.mean([b["center"][1] for b in group])
            
            for idx, region in enumerate(figure_regions, start=1):
                # Check if caption is below the figure (captions are usually below)
                rx1, ry1, rx2, ry2 = region
                figure_bottom = ry2
                
                # Caption should be below figure, within reasonable distance
                if group_y > figure_bottom:
                    distance = group_y - figure_bottom
                    # Also check horizontal overlap
                    group_x_min = min(b["bbox"][0] for b in group)
                    group_x_max = max(b["bbox"][2] for b in group)
                    
                    if (group_x_max >= rx1 and group_x_min <= rx2) and distance < best_distance:
                        best_match = idx
                        best_distance = distance
            
            if best_match and best_distance < 200:  # Only match if reasonably close
                grouped.append(f"Figure {best_match}: {group_text}")
            else:
                # No match found, just add the text
                grouped.append(group_text)
        
        return grouped
    
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
                    # Extract table content
                    table_roi = image.crop((x, y, x + w, y + h))
                    table_text = self.extract_text(table_roi, paragraph=False)
                    
                    tables.append({
                        "index": i,
                        "bbox": [x, y, w, h],
                        "area": int(area),
                        "content": table_text  # Add extracted text content
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
    
    def auto_highlight_important_sections(
        self,
        structured_text: Dict,
        figures: List[Dict],
        tables: List[Dict]
    ) -> List[Dict]:
        """
        Automatically identify important sections/figures for purple highlighting.
        Returns list of bboxes to highlight.
        """
        highlights = []
        
        # Highlight section headings (Introduction, Methods, Results, etc.)
        important_sections = [
            "introduction", "abstract", "method", "methods", "approach",
            "results", "experiments", "discussion", "conclusion"
        ]
        
        for section in structured_text.get("sections", []):
            title_lower = section.get("title", "").lower()
            if any(keyword in title_lower for keyword in important_sections):
                # Find the bbox for this section from raw blocks
                section_blocks = [
                    b for b in structured_text.get("raw_blocks", [])
                    if b.get("text", "").lower().startswith(title_lower[:10])
                ]
                if section_blocks:
                    first_block = section_blocks[0]
                    highlights.append({
                        "bbox": first_block.get("bbox"),
                        "type": "section",
                        "title": section.get("title"),
                        "reason": f"Important section: {section.get('title')}"
                    })
        
        # Highlight large figures (likely important)
        for fig in figures:
            if fig.get("area", 0) > 50000:  # Large figures are often key
                highlights.append({
                    "bbox": fig.get("bbox"),
                    "type": "figure",
                    "index": fig.get("index"),
                    "reason": f"Large figure (area: {fig.get('area')}px²) - likely important"
                })
        
        # Highlight tables (usually contain key data)
        for table in tables:
            if table.get("area", 0) > 10000:
                highlights.append({
                    "bbox": table.get("bbox"),
                    "type": "table",
                    "index": table.get("index"),
                    "reason": "Table detected - contains structured data"
                })
        
        return highlights
    
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
        
        # Detect other elements first for layout awareness
        tables = self.detect_tables(image)
        figures = self.detect_figures(image)
        highlights = self.detect_highlights(image)
        
        text_data = self.extract_structured_text(
            image=image,
            paragraph=not full_page,
            figures=figures,
            tables=tables
        )
        text = text_data.get("full_text", "No text detected")
        
        # Auto-highlight important sections
        auto_highlights = self.auto_highlight_important_sections(
            structured_text=text_data,
            figures=figures,
            tables=tables
        )
        
        # Add auto-highlights to purple highlights
        for auto_hl in auto_highlights:
            highlights["purple"].append({
                "bbox": auto_hl.get("bbox"),
                "reason": auto_hl.get("reason"),
                "type": auto_hl.get("type")
            })
        
        return {
            "text": text,
            "structured_text": text_data,
            "tables": tables,
            "figures": figures,
            "highlights": highlights,
            "auto_highlights": auto_highlights,
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