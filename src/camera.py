
import cv2


class Camera:

    def __init__(self, camera_index=0, width=640, height=480, video_path=None):

        self.camera_index = camera_index

        self.is_video = video_path is not None
        self.cap = cv2.VideoCapture(str(video_path) if self.is_video else camera_index)

        if not self.is_video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"No se pudo abrir: {video_path if self.is_video else 'cámara'}")

    @property
    def fps(self):
        value = self.cap.get(cv2.CAP_PROP_FPS)
        return value if 0 < value <= 240 else 30.0

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        self.cap.release()
        cv2.destroyAllWindows()
