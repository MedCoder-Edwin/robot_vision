
import cv2


class Visualizer:

    def draw_detections(self, frame, detections):

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            class_name = detection["class"]
            confidence = detection["confidence"]
            track_id = detection["track_id"]

            label = f"{class_name} ID:{track_id} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return frame

    def draw_free_space(self, frame, free_space):

        y_start = free_space["y_start"]
        sectors = free_space["sectors"]

        height, width = frame.shape[:2]

        # Delimitar ROI
        cv2.line(
            frame,
            (0, y_start),
            (width, y_start),
            (255, 255, 0),
            2
        )

        # Delimitar sectores
        cv2.line(
            frame,
            (width // 3, y_start),
            (width // 3, height),
            (255, 0, 255),
            1
        )

        cv2.line(
            frame,
            (2 * width // 3, y_start),
            (2 * width // 3, height),
            (255, 0, 255),
            1
        )

        # Mostrar puntuaciones
        labels = [
            ("LEFT", sectors["left"]),
            ("CENTER", sectors["center"]),
            ("RIGHT", sectors["right"])
        ]

        for i, (name, score) in enumerate(labels):

            x = 20 + i * (width // 3)

            cv2.putText(
                frame,
                f"{name}: {score * 100:.1f}%",
                (x, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        return frame
