"""
Visual Detector — from silent-video-segmentation project (exact copy).
YOLOv8 + heuristic detection for charts, tables, diagrams, slides.
"""

import os
import shutil
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from loguru import logger


YOLO_VISUAL_TYPES = {
    "chart": "chart", "bar chart": "chart", "pie chart": "chart",
    "line chart": "graph", "table": "table", "diagram": "diagram",
    "flowchart": "diagram", "graph": "graph", "plot": "graph",
    "slide": "slide", "infographic": "infographic", "figure": "diagram",
}


class VisualDetector:
    def __init__(self, confidence_threshold: float = 0.65, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.yolo_model = self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8x.pt")
            logger.info("✅ YOLOv8x loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"YOLOv8 not available: {e} — using heuristic detection")
            return None

    def detect(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Visual detection on {len(frames)} frames")
        informative = []
        for frame in frames:
            result = self._detect_frame(frame)
            if result:
                informative.append(result)
        logger.info(f"Informative: {len(informative)}/{len(frames)} frames")
        return informative

    def _detect_frame(self, frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = frame["frame_path"]
        img = cv2.imread(path)
        if img is None:
            return None
        if self.yolo_model:
            yolo_result = self._yolo_detect(img, path)
            if yolo_result:
                frame.update(yolo_result)
                frame["detection_method"] = "yolo"
                self._save_detected_frame(frame, img)
                return frame
        heuristic_result = self._heuristic_detect(img)
        if heuristic_result:
            frame.update(heuristic_result)
            frame["detection_method"] = "heuristic"
            self._save_detected_frame(frame, img)
            return frame
        return None

    def _yolo_detect(self, img: np.ndarray, path: str) -> Optional[Dict]:
        try:
            results = self.yolo_model(img, verbose=False, conf=self.confidence_threshold)
            if not results or len(results[0].boxes) == 0:
                return None
            best_conf, best_class, best_bbox = 0.0, None, None
            for box in results[0].boxes:
                conf = float(box.conf[0])
                class_name = self.yolo_model.names[int(box.cls[0])].lower()
                visual_type = self._map_class_to_type(class_name)
                if visual_type and conf > best_conf:
                    best_conf = conf
                    best_class = visual_type
                    best_bbox = [int(x) for x in box.xyxy[0].tolist()]
            if best_class and best_conf >= self.confidence_threshold:
                return {"visual_type": best_class, "confidence": round(best_conf, 3), "bbox": best_bbox}
        except Exception as e:
            logger.debug(f"YOLOv8 error: {e}")
        return None

    def _heuristic_detect(self, img: np.ndarray) -> Optional[Dict]:
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (h * w)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            unique_hues = len(np.unique(hsv[:, :, 0]))
            color_variety = unique_hues / 180.0
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
            v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
            line_density = (np.count_nonzero(h_lines) + np.count_nonzero(v_lines)) / (h * w)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            score = 0.0
            visual_type = "unknown"
            if line_density > 0.01:
                score += 0.4
                visual_type = "table" if line_density > 0.02 else "chart"
            if edge_density > 0.05:
                score += 0.3
            if laplacian_var > 500:
                score += 0.2
            if color_variety < 0.4:
                score += 0.1
            if self._is_natural_scene(img):
                return None
            if score >= 0.5:
                return {"visual_type": visual_type, "confidence": round(min(score, 0.95), 3), "bbox": None}
        except Exception as e:
            logger.debug(f"Heuristic detection error: {e}")
        return None

    def _is_natural_scene(self, img: np.ndarray) -> bool:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        high_sat = np.count_nonzero((sat > 80) & (val > 60) & (val < 220))
        return high_sat / (img.shape[0] * img.shape[1]) > 0.45

    def _map_class_to_type(self, class_name: str) -> Optional[str]:
        for key, vtype in YOLO_VISUAL_TYPES.items():
            if key in class_name:
                return vtype
        return None

    def _save_detected_frame(self, frame: Dict, img: np.ndarray):
        out_dir = os.path.dirname(os.path.dirname(frame["frame_path"]))
        detected_dir = os.path.join(out_dir, "detected")
        os.makedirs(detected_dir, exist_ok=True)
        fname = os.path.basename(frame["frame_path"])
        detected_path = os.path.join(detected_dir, fname)
        if not os.path.exists(detected_path):
            cv2.imwrite(detected_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame["frame_path"] = detected_path
