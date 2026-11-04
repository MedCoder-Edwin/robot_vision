import cv2
import numpy as np
import os 

# Raíz del proyecto para guardar calibraciones = carpeta padre de src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(PROJECT_ROOT, "config", "floor_calibration.npz")

class ColorModel:
    """Modelo de color del piso en Lab (media + covarianza)."""

    REG = 10.0
    MAX_PIXELS = 2000   # píxeles que se conservan por muestra (mantiene el archivo pequeño)

    def __init__(self):
        self.samples = []
        self.mean = None
        self.icov = None

    @property
    def ready(self):
        return self.mean is not None

    def _fit(self):
        data = np.vstack(self.samples)
        self.mean = data.mean(axis=0)
        cov = np.cov(data.T) + np.eye(3, dtype=np.float32) * self.REG
        self.icov = np.linalg.inv(cov).astype(np.float32)

    def add_samples(self, bgr_patch):
        lab = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)
        lab = lab.reshape(-1, 3).astype(np.float32)
        if len(lab) > self.MAX_PIXELS:
            idx = np.random.choice(len(lab), self.MAX_PIXELS, replace=False)
            lab = lab[idx]
        self.samples.append(lab)
        self._fit()

    def set_samples(self, samples):
        self.samples = list(samples)
        self._fit()

    def reset(self):
        self.samples, self.mean, self.icov = [], None, None

    def distance(self, lab):
        d = lab - self.mean
        m2 = np.einsum("...i,ij,...j->...", d, self.icov, d)
        return np.sqrt(np.maximum(m2, 0))


class FreeSpaceDetector:

    PROC_WIDTH = 160
    ROW_FREE_MIN = 0.6  # Fracción mínima de fila libre para considerarla libre

    def __init__(self, roi_start=0.45):
        self.roi_start = roi_start
        self.floor = ColorModel()
        self.current_roi = None
        self.load_calibration()

        self.window_name = "Floor Calibration"
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("Umbral x10", self.window_name, 50, 200, lambda x: None)

    def _guide_box(self, roi):
        h, w = roi.shape[:2]
        return int(h * 0.60), int(h * 0.95), int(w * 0.35), int(w * 0.65)

    def save_calibration(self):
        if not self.floor.ready:
            print("No hay calibración que guardar")
            return
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        np.savez(
            CALIB_PATH,
            samples=np.vstack(self.floor.samples),
            counts=np.array([len(s) for s in self.floor.samples]),
            threshold=cv2.getTrackbarPos("Umbral x10", self.window_name),
        )
        print(f"Calibración guardada en {CALIB_PATH}")

    def load_calibration(self):
        if not os.path.exists(CALIB_PATH):
            print("No hay calibración guardada; calibra con 'c' y guarda con 's'")
            return
        try:
            data = np.load(CALIB_PATH)
            chunks = np.split(data["samples"], np.cumsum(data["counts"])[:-1])
            self.floor.set_samples(chunks)
            cv2.setTrackbarPos("Umbral x10", self.window_name, int(data["threshold"]))
            print(f"Calibración cargada: {len(chunks)} muestras")
        except Exception as e:
            print(f"No se pudo cargar la calibración: {e}")

    def handle_key(self, key):
        """c = añadir muestra | s = guardar | l = recargar | r = reiniciar."""
        if self.current_roi is None:
            return
        if key == ord("c"):
            y1, y2, x1, x2 = self._guide_box(self.current_roi)
            self.floor.add_samples(self.current_roi[y1:y2, x1:x2])
            print(f"Muestras de piso: {len(self.floor.samples)}")
        elif key == ord("s"):
            self.save_calibration()
        elif key == ord("l"):
            self.load_calibration()
        elif key == ord("r"):
            self.floor.reset()
            print("Modelo reiniciado (el archivo guardado no se borra)")

    def extract_roi(self, frame):
        y_start = int(frame.shape[0] * self.roi_start)
        return frame[y_start:, :], y_start

    def segment_floor(self, roi):
        scale = self.PROC_WIDTH / roi.shape[1]
        small = cv2.resize(roi, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

        thr = cv2.getTrackbarPos("Umbral x10", self.window_name) / 10.0
        mask = (self.floor.distance(lab) < thr).astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return cv2.resize(mask, (roi.shape[1], roi.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    
    def _depth(self, sector_mask):
        """Fracción (0-1) de la altura del ROI que está libre, contando desde abajo."""
        row_free = sector_mask.mean(axis=1) / 255.0
        blocked = np.where(row_free[::-1] < self.ROW_FREE_MIN)[0]
        n = len(row_free)
        return (blocked[0] if len(blocked) else n) / n
    
    def analyze_sectors(self, mask):
        w = mask.shape[1]
        parts = {
            "left": mask[:, : w // 3],
            "center": mask[:, w // 3: 2 * w // 3],
            "right": mask[:, 2 * w // 3:],
        }
        ratio = {k: np.count_nonzero(v) / v.size for k, v in parts.items()}
        depth = {k: self._depth(v) for k, v in parts.items()}
        return ratio, depth

    def detect(self, frame):
        roi, y_start = self.extract_roi(frame)
        self.current_roi = roi.copy()

        view = roi.copy()
        y1, y2, x1, x2 = self._guide_box(roi)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if self.floor.ready:
            mask = self.segment_floor(roi)
            msg = f"Muestras de piso: {len(self.floor.samples)}"
        else:
            mask = np.zeros(roi.shape[:2], np.uint8)
            msg = "Pon piso libre en el recuadro y pulsa 'c'"
        cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)

        ratio, depth = self.analyze_sectors(mask)
        return {"roi": roi, "ready": self.floor.ready, "mask": mask, "y_start": y_start,
                "sectors": ratio, "depth": depth, "calibration": view}