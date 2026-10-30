
import cv2


class Camera:

    def __init__(self, camera_index=0, width=640, height=480):

        self.camera_index = camera_index

        self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        self.cap.release()
        cv2.destroyAllWindows()
