
import cv2
import time

from src.camera import Camera
from src.object_detection import ObjectDetector
from src.free_space import FreeSpaceDetector
from src.visualization import Visualizer
from src.navigation import NavigationPolicy


def main():

    camera = Camera(camera_index=0)

    detector = ObjectDetector(
        model_path="yolov8n.pt",
        confidence=0.40
    )

    free_space_detector = FreeSpaceDetector()

    policy = NavigationPolicy()

    visualizer = Visualizer()

    previous_time = time.perf_counter()

    try:

        while True:

            frame = camera.read()

            if frame is None:
                continue

            # 1. Deteccion de objetos
            detections = detector.detect(frame)

            # 2. Deteccion de espacio libre
            free_space = free_space_detector.detect(frame)

            # 3. Politica de navegacion
            action = policy.decide(free_space, detections, frame.shape)
            

            # 3. Visualizacion
            output = frame.copy()

            output = visualizer.draw_detections(
                output,
                detections
            )

            output = visualizer.draw_free_space(
                output,
                free_space
            )
            cv2.putText(output, f"ACTION: {action}", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0) if action == "FORWARD" else (0, 0, 255), 2)
            # 4. FPS
            current_time = time.perf_counter()

            elapsed = current_time - previous_time

            fps = 1.0 / elapsed if elapsed > 0 else 0

            previous_time = current_time

            cv2.putText(
                output,
                f"FPS: {fps:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Mostrar imagen original
            cv2.imshow(
                "Robot Vision",
                output
            )

            # Mostrar mascara
            cv2.imshow(
                "Free Space Mask",
                free_space["mask"]
            )

            cv2.imshow(
                            free_space_detector.window_name,
                            free_space["calibration"]
                        )
            

            key = cv2.waitKey(1) & 0xFF
            free_space_detector.handle_key(key)
            if key == ord("q") or key == 27:
                break

    finally:

        camera.release()


if __name__ == "__main__":
    main()
