from dataclasses import dataclass
import cv2

@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple
    center: tuple

class YOLOv8Detector:
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, weights="yolov8n.pt", confidence=0.5, device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.names = self.model.names

    def detect(self, frame):
        kwargs = {"conf": self.confidence, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(frame, **kwargs)
        detections = []
        if not results:
            return detections
        result = results[0]
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            name = str(self.names[cls_id]).lower()
            if name not in self.VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append(Detection(-1, name, conf,
                                        (int(x1), int(y1), int(x2), int(y2)),
                                        (float(cx), float(cy))))
        return detections

    @staticmethod
    def draw(frame, detections):
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            if d.track_id >= 0:
                label += f" ID:{d.track_id}"
            cv2.putText(out, label, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out
