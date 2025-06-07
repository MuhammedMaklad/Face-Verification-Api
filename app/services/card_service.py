import os
import cv2
import easyocr
from ultralytics import YOLO
import pytesseract
from PIL import Image
import numpy as np


class IDCardProcessor:
    def __init__(self, tesseract_path="r'C:\Program Files\Tesseract-OCR\tesseract.exe'", yolo_model_path='app/services/best (1).pt'):
        """
        Initialize the ID card processing service

        Args:
            tesseract_path (str): Path to Tesseract OCR executable
            yolo_model_path (str): Path to YOLO model weights
        """
        # Configure Tesseract
        # if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Initialize OCR readers
        self.reader = easyocr.Reader(['ar'])
        self.model = YOLO(yolo_model_path)

        # Class ID mappings (adjust based on your model)
        self.class_ids = {
            'name': [5, 11],
            'address': [0, 1],
            'id': [8],
            'job': [9, 10]
        }

    def load_image(self, file):
        """
        Load image from file object or bytes

        Args:
            file: File object or bytes

        Returns:
            np.ndarray: Loaded image as numpy array

        Raises:
            ValueError: If image cannot be loaded
        """
        if isinstance(file, bytes):
            nparr = np.frombuffer(file, np.uint8)
        else:
            nparr = np.frombuffer(file.read(), np.uint8)

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not load image from input")
        return frame

    def process_image(self, frame):
        """
        Main processing pipeline for ID card images

        Args:
            frame (np.ndarray): Input image

        Returns:
            dict: Extracted information (name, address, id, job)
        """
        # Detect and extract text
        names, addresses, info = self._detect_and_extract_text(frame)

        # Process detected info
        processed_info = self._process_detected_info(names, addresses, info)

        return processed_info

    def _detect_and_extract_text(self, frame):
        """
        Internal method for text detection and extraction

        Args:
            frame (np.ndarray): Input image

        Returns:
            tuple: (names_with_boxes, addresses_with_boxes, detected_info)
        """
        results = self.model.track(
            frame,
            persist=True,
            iou=0.4,
            show=False,
            tracker="bytetrack.yaml",
            verbose=False
        )

        names_with_boxes = []
        addresses_with_boxes = []
        detected_info = {"name": [], "address": [], "id": []}

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls = results[0].boxes.cls.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu()
            conf = results[0].boxes.conf.tolist()

            for box, track_id, cof, c in zip(boxes, track_ids, conf, cls):
                box = box.cpu().int().tolist()
                roi = frame[box[1] - 15:box[3] + 15, box[0] - 15:box[2] + 15]

                if roi.size == 0:
                    continue

                # Process based on class ID
                if c in self.class_ids['name']:
                    self._process_name(roi, box, names_with_boxes, detected_info, c)
                elif c in self.class_ids['address']:
                    self._process_address(roi, box, addresses_with_boxes)
                elif c in self.class_ids['id']:
                    self._process_id(roi, detected_info)

        return names_with_boxes, addresses_with_boxes, detected_info

    def _process_name(self, roi, box, names_with_boxes, detected_info, class_id):
        """Process name fields"""
        for result in self.reader.readtext(roi, detail=1):
            text = result[1]
            bbox = result[0]
            word_x = box[0] + int((bbox[0][0] + bbox[2][0]) / 2)

            if class_id == 11:
                names_with_boxes.append({"text": text, "x": word_x})
            elif class_id == 5:
                detected_info["name"].append(text)

    def _process_address(self, roi, box, addresses_with_boxes):
        """Process address fields"""
        for result in self.reader.readtext(roi, detail=1):
            text = result[1]
            bbox = result[0]
            word_x = box[0] + int((bbox[0][0] + bbox[2][0]) / 2)
            addresses_with_boxes.append({"text": text, "x": word_x})

    def _process_id(self, roi, detected_info):
        """Process ID fields using Tesseract"""
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_pil = roi_pil.convert('L')
        roi_pil = roi_pil.point(lambda x: 0 if x < 128 else 255, '1')
        boxes = pytesseract.image_to_boxes(roi_pil, lang='ara_num_test', config='--psm 6')
        detected_digits = [char for box_line in boxes.splitlines()
                           if (char := box_line.split(' ')[0]).isdigit()]
        detected_info["id"].append(''.join(detected_digits))

    def _process_detected_info(self, names_with_boxes, addresses_with_boxes, detected_info):
        """Post-process detected information"""
        detected_info["name"] = (
                (detected_info["name"][0] + ' ' if detected_info["name"] else '') +
                self._sort_and_join_text(names_with_boxes, reverse=True)
        )
        detected_info["address"] = self._sort_and_join_text(addresses_with_boxes, reverse=True)
        return detected_info

    def _sort_and_join_text(self, text_with_boxes, reverse=True):
        """Sort text by x-coordinate and join"""
        text_with_boxes.sort(key=lambda item: item['x'], reverse=reverse)
        return " ".join(item['text'] for item in text_with_boxes)

    def process_job(self, frame):
        """Process job information from back of ID card"""
        results = self.model.track(
            frame,
            persist=True,
            iou=0.4,
            show=False,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detected_jobs = []

        if results[0].boxes.id is not None:
            for box, track_id, cof, c in zip(results[0].boxes.xyxy.cpu(),
                                             results[0].boxes.id.int().cpu().tolist(),
                                             results[0].boxes.conf.tolist(),
                                             results[0].boxes.cls.int().cpu().tolist()):
                if c not in self.class_ids['job']:
                    continue

                box = box.cpu().int().tolist()
                roi = frame[box[1] - 15:box[3] + 15, box[0] - 15:box[2] + 15]

                if roi.size == 0:
                    continue

                for result in self.reader.readtext(roi, detail=1):
                    detected_jobs.append({
                        "text": result[1],
                        "x": box[0] + int((result[0][0][0] + result[0][2][0]) / 2)
                    })

        return self._sort_and_join_text(detected_jobs, reverse=False)