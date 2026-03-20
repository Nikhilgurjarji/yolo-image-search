from ultralytics import YOLO
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.config import load_config , save_config


class YOLOv11Inference:
    def __init__(self,model_name , device = 'cpu'):
        self.model = YOLO(model_name)
        self.model.to(device) 
        self.device = device

        config = load_config()
        self.conf_threshold = config["model"]["conf_threshold"]
        self.extension = config["data"]["image_extensions"]

    def process_image(self,image_path):
        results = self.model.predict(
            source=image_path,
            conf = self.conf_threshold,
            device = self.device
        )

        detection = []
        class_counts = {}

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()
            
                detection.append({
                    "class" : cls ,
                    "conf" : conf ,
                    "bbox" :bbox ,
                    "count" : 1
                })

                class_counts[cls] = class_counts.get(cls,0) + 1
        
        for det in detection:
            det['count'] = class_counts[det['class']]
        
        return {
            "image_path" : str(image_path),
            "detection" : detection ,
            "total_objects" : len(detection),
            "unique_objects" : list(class_counts.keys()),
            "class_counts" : class_counts
        }

    def process_directory(self,dir_path):
        metadata = []

        patterns = [f"*{ext}" for ext in self.extension]

        image_path = []
        for pattern in patterns:
            image_path.extend(Path(dir_path).glob(pattern))
        
        for img_path in image_path:
            try:
                metadata.append(self.process_image(img_path))
            except Exception as e:
                print("error processing Directory" ,{str(e)})
                continue
        return metadata