"""Diagram parser for converting detected diagrams to Mermaid and Excalidraw formats."""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json
import re


class DiagramParser:
    """Parse diagrams and convert to Mermaid/Excalidraw formats."""
    
    def __init__(self, ocr_processor):
        """
        Initialize diagram parser.
        
        Args:
            ocr_processor: VLMProcessor instance for text extraction
        """
        self.ocr = ocr_processor
    
    def detect_diagram_structure(self, image: Image.Image, bbox: List[int]) -> Dict:
        """
        Detect structure of a diagram (boxes, arrows, text).
        
        Args:
            image: Full image
            bbox: Bounding box [x, y, w, h] of the diagram region
            
        Returns:
            Dictionary with detected nodes, edges, and text
        """
        try:
            # Crop to diagram region
            x, y, w, h = bbox
            diagram_img = image.crop((x, y, x + w, y + h))
            img_array = np.array(diagram_img)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect boxes/rectangles
            boxes = self._detect_boxes(gray)
            
            # Detect arrows/lines
            arrows = self._detect_arrows(gray)
            
            # Extract text from boxes
            nodes = []
            for i, box in enumerate(boxes):
                box_x, box_y, box_w, box_h = box
                # Extract text from box region
                box_roi = diagram_img.crop((box_x, box_y, box_x + box_w, box_y + box_h))
                text = self.ocr.extract_text(box_roi, paragraph=False)
                text = text.strip()[:50]  # Limit text length
                
                nodes.append({
                    "id": f"node_{i}",
                    "text": text or f"Node {i+1}",
                    "bbox": [box_x, box_y, box_w, box_h],
                    "center": [box_x + box_w // 2, box_y + box_h // 2]
                })
            
            # Detect connections between boxes
            edges = self._detect_connections(boxes, arrows, nodes)
            
            return {
                "nodes": nodes,
                "edges": edges,
                "bbox": bbox
            }
        
        except Exception as e:
            print(f"⚠️ Diagram parsing error: {e}")
            return {"nodes": [], "edges": [], "bbox": bbox}
    
    def _detect_boxes(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular boxes in the image."""
        # Threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small noise
                continue
            
            # Approximate as rectangle
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4:  # At least 4 points for a rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h if h > 0 else 0
                # Reasonable aspect ratios for boxes
                if 0.2 < aspect_ratio < 5.0:
                    boxes.append((x, y, w, h))
        
        return boxes
    
    def _detect_arrows(self, gray: np.ndarray) -> List[Dict]:
        """Detect arrows and lines in the image."""
        # Use HoughLines to detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        arrows = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                arrows.append({
                    "start": [int(x1), int(y1)],
                    "end": [int(x2), int(y2)],
                    "angle": angle
                })
        
        return arrows
    
    def _detect_connections(
        self,
        boxes: List[Tuple[int, int, int, int]],
        arrows: List[Dict],
        nodes: List[Dict]
    ) -> List[Dict]:
        """Detect which boxes are connected by arrows."""
        edges = []
        
        for arrow in arrows:
            start = arrow["start"]
            end = arrow["end"]
            
            # Find which boxes the arrow connects
            from_node = None
            to_node = None
            
            for node in nodes:
                node_bbox = node["bbox"]
                node_x, node_y, node_w, node_h = node_bbox
                node_center = node["center"]
                
                # Check if arrow starts near this box
                if self._point_near_box(start, node_bbox, threshold=20):
                    from_node = node["id"]
                
                # Check if arrow ends near this box
                if self._point_near_box(end, node_bbox, threshold=20):
                    to_node = node["id"]
            
            if from_node and to_node and from_node != to_node:
                edges.append({
                    "from": from_node,
                    "to": to_node
                })
        
        # Remove duplicates
        unique_edges = []
        seen = set()
        for edge in edges:
            key = (edge["from"], edge["to"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)
        
        return unique_edges
    
    def _point_near_box(self, point: List[int], bbox: List[int], threshold: int = 20) -> bool:
        """Check if a point is near a bounding box."""
        px, py = point
        bx, by, bw, bh = bbox
        
        # Check if point is within threshold of box edges
        if (bx - threshold <= px <= bx + bw + threshold and
            by - threshold <= py <= by + bh + threshold):
            return True
        return False
    
    def to_mermaid(self, diagram_data: Dict) -> str:
        """
        Convert diagram data to Mermaid flowchart syntax.
        
        Args:
            diagram_data: Dictionary with nodes and edges
            
        Returns:
            Mermaid code string
        """
        nodes = diagram_data.get("nodes", [])
        edges = diagram_data.get("edges", [])
        
        if not nodes:
            return "```mermaid\ngraph TD\n  No diagram structure detected\n```"
        
        mermaid = "```mermaid\ngraph TD\n"
        
        # Add nodes
        for node in nodes:
            node_id = node["id"]
            text = node.get("text", "Node")
            # Clean text for Mermaid (remove special chars)
            text = re.sub(r'[^\w\s-]', '', text)
            text = text.strip()[:30]  # Limit length
            if not text:
                text = node_id.replace("_", " ").title()
            
            mermaid += f"  {node_id}[\"{text}\"]\n"
        
        # Add edges
        for edge in edges:
            from_id = edge["from"]
            to_id = edge["to"]
            mermaid += f"  {from_id} --> {to_id}\n"
        
        mermaid += "```"
        
        return mermaid
    
    def to_excalidraw(self, diagram_data: Dict, image_size: Tuple[int, int]) -> str:
        """
        Convert diagram data to Excalidraw JSON format.
        
        Args:
            diagram_data: Dictionary with nodes and edges
            image_size: (width, height) of the original image
            
        Returns:
            Excalidraw JSON string
        """
        nodes = diagram_data.get("nodes", [])
        edges = diagram_data.get("edges", [])
        bbox = diagram_data.get("bbox", [0, 0, 100, 100])
        
        if not nodes:
            return json.dumps({
                "type": "excalidraw",
                "version": 2,
                "source": "cua-pdf-reader",
                "elements": []
            }, indent=2)
        
        elements = []
        element_id = 0
        
        # Convert nodes to rectangles with text
        for node in nodes:
            node_bbox = node["bbox"]
            x, y, w, h = node_bbox
            
            # Rectangle element
            rect_id = f"rect_{element_id}"
            elements.append({
                "type": "rectangle",
                "version": 1,
                "versionNonce": element_id,
                "isDeleted": False,
                "id": rect_id,
                "fillStyle": "solid",
                "strokeWidth": 2,
                "strokeStyle": "solid",
                "roughness": 1,
                "opacity": 100,
                "angle": 0,
                "x": float(x + bbox[0]),
                "y": float(y + bbox[1]),
                "strokeColor": "#000000",
                "backgroundColor": "transparent",
                "width": float(w),
                "height": float(h),
                "seed": element_id,
                "groupIds": [],
                "frameId": None,
                "roundness": None,
                "boundElements": [],
                "updated": 1,
                "link": None,
                "locked": False
            })
            element_id += 1
            
            # Text element
            text = node.get("text", "")
            if text:
                text_id = f"text_{element_id}"
                elements.append({
                    "type": "text",
                    "version": 1,
                    "versionNonce": element_id,
                    "isDeleted": False,
                    "id": text_id,
                    "fillStyle": "solid",
                    "strokeWidth": 2,
                    "strokeStyle": "solid",
                    "roughness": 1,
                    "opacity": 100,
                    "angle": 0,
                    "x": float(x + bbox[0] + 5),
                    "y": float(y + bbox[1] + h // 2 - 10),
                    "strokeColor": "#000000",
                    "backgroundColor": "transparent",
                    "width": float(w - 10),
                    "height": float(20),
                    "seed": element_id,
                    "groupIds": [],
                    "frameId": None,
                    "roundness": None,
                    "boundElements": [],
                    "updated": 1,
                    "link": None,
                    "locked": False,
                    "fontSize": 16,
                    "fontFamily": 1,
                    "text": text[:50],
                    "textAlign": "center",
                    "verticalAlign": "middle",
                    "baseline": 15,
                    "containerId": None,
                    "originalText": text[:50]
                })
                element_id += 1
        
        # Convert edges to arrows
        for edge in edges:
            from_node = next((n for n in nodes if n["id"] == edge["from"]), None)
            to_node = next((n for n in nodes if n["id"] == edge["to"]), None)
            
            if from_node and to_node:
                from_center = from_node["center"]
                to_center = to_node["center"]
                
                arrow_id = f"arrow_{element_id}"
                elements.append({
                    "type": "arrow",
                    "version": 1,
                    "versionNonce": element_id,
                    "isDeleted": False,
                    "id": arrow_id,
                    "fillStyle": "solid",
                    "strokeWidth": 2,
                    "strokeStyle": "solid",
                    "roughness": 1,
                    "opacity": 100,
                    "angle": 0,
                    "x": float(from_center[0] + bbox[0]),
                    "y": float(from_center[1] + bbox[1]),
                    "strokeColor": "#000000",
                    "backgroundColor": "transparent",
                    "width": float(to_center[0] - from_center[0]),
                    "height": float(to_center[1] - from_center[1]),
                    "seed": element_id,
                    "groupIds": [],
                    "frameId": None,
                    "roundness": {
                        "type": 2
                    },
                    "boundElements": [],
                    "updated": 1,
                    "link": None,
                    "locked": False,
                    "points": [
                        [0, 0],
                        [float(to_center[0] - from_center[0]), float(to_center[1] - from_center[1])]
                    ],
                    "lastCommittedPoint": None,
                    "startBinding": None,
                    "endBinding": None,
                    "startArrowhead": None,
                    "endArrowhead": "arrow"
                })
                element_id += 1
        
        excalidraw_data = {
            "type": "excalidraw",
            "version": 2,
            "source": "cua-pdf-reader",
            "elements": elements,
            "appState": {
                "gridSize": None,
                "viewBackgroundColor": "#ffffff"
            },
            "files": {}
        }
        
        return json.dumps(excalidraw_data, indent=2)

