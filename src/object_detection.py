
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.40):

        self.model = YOLO(model_path)
        self.confidence = confidence

        # Clases relevantes para navegación
        self.target_classes = [
            "person",
            "chair",
            "table",
            "backpack",
            "bottle"
        ]

    def detect(self, frame):

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=320,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                if class_name not in self.target_classes:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                track_id = -1

                if box.id is not None:
                    track_id = int(box.id[0])

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections
