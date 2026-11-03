import argparse
from pathlib import Path
import time

import cv2

from src.camera import Camera
from src.free_space import FreeSpaceDetector
from src.visualization import Visualizer
from src.navigation import NavigationPolicy


def select_video():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Selecciona un video del piso y los obstaculos",
            filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                       ("Todos los archivos", "*.*")],
        )
    finally:
        root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Prueba de vision con camara o video.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Ruta de un video grabado.")
    source.add_argument("--select-video", action="store_true", help="Abrir selector de video.")
    source.add_argument("--camera", type=int, default=0, help="Indice de camara (0 por defecto).")
    return parser.parse_args()


def show(name, view):
    # Ajusta solo la vista; el analisis conserva la imagen original.
    h, w = view.shape[:2]
    scale = min(1.0, 960 / w, 640 / h)
    if scale < 1:
        view = cv2.resize(view, (round(w * scale), round(h * scale)))
    cv2.imshow(name, view)


def main():
    args = parse_args()
    if args.select_video:
        selected = select_video()
        if not selected:
            return
        args.video = Path(selected)
    if args.video is not None and not args.video.is_file():
        raise SystemExit(f"No existe el video: {args.video}")

    camera = Camera(camera_index=args.camera, video_path=args.video)
    try:
        from src.object_detection import ObjectDetector

        model = Path(__file__).resolve().parent / "yolov8n.pt"
        detector = ObjectDetector(model_path=str(model), confidence=0.40)
        free_space_detector = FreeSpaceDetector()
        policy = NavigationPolicy()
        visualizer = Visualizer()

        paused = camera.is_video
        ended = False
        read_next = True
        frame = None
        detections = []
        frame_number = 0
        fps = 0.0
        print("C: muestra de piso | R: borrar calibracion | Q/Esc: salir")
        if camera.is_video:
            print("Espacio: reproducir/pausar | N: siguiente cuadro")

        while True:
            started = time.perf_counter()
            if read_next:
                next_frame = camera.read()
                if next_frame is None:
                    if not camera.is_video or frame is None:
                        print("No se reciben imagenes. Prueba finalizada.")
                        break
                    ended = True
                    paused = True
                else:
                    frame = next_frame
                    frame_number += 1
                    # No alimentar el seguimiento repetidamente con el cuadro pausado.
                    detections = detector.detect(frame)
                    fps = 1.0 / max(time.perf_counter() - started, 1e-6)

            free_space = free_space_detector.detect(frame)
            action = policy.decide(free_space, detections, frame.shape)
            output = visualizer.draw_detections(frame.copy(), detections)
            output = visualizer.draw_free_space(output, free_space)
            cv2.putText(output, f"ACTION: {action}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if action == "FORWARD" else (0, 0, 255), 2)
            cv2.putText(output, f"Procesamiento: {fps:.1f} FPS", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if camera.is_video:
                state = "FIN" if ended else "PAUSA" if paused else "VIDEO"
                cv2.putText(output, f"{state} | Cuadro {frame_number}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            show("Robot Vision", output)
            show("Free Space Mask", free_space["mask"])
            show(free_space_detector.window_name, free_space["calibration"])

            # No avanzar mas rapido que el archivo ni saltar cuadros si tarda.
            elapsed_ms = (time.perf_counter() - started) * 1000
            delay = (max(1, round(1000 / camera.fps - elapsed_ms))
                     if camera.is_video and not paused else 20)
            key = cv2.waitKey(delay) & 0xFF
            free_space_detector.handle_key(key)
            if key in (ord("q"), 27):
                break
            if camera.is_video:
                if key == ord(" ") and not ended:
                    paused = not paused
                step = key == ord("n") and not ended
                if step:
                    paused = True
                read_next = not ended and (not paused or step)
            else:
                read_next = True
    finally:
        camera.release()


if __name__ == "__main__":
    main()
