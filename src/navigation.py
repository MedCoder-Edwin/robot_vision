class NavigationPolicy:
    """Decide FORWARD / LEFT / RIGHT / STOP con histéresis."""

    def __init__(self, stop_depth=0.30, resume_depth=0.45, bbox_danger=0.75):
        self.stop_depth = stop_depth        # por debajo: el sector está bloqueado
        self.resume_depth = resume_depth    # para volver a avanzar se exige más margen
        self.bbox_danger = bbox_danger
        self.action = "STOP"

    def _yolo_blocks(self, detections, frame_shape):
        h, w = frame_shape[:2]
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            in_center = x2 > w / 3 and x1 < 2 * w / 3
            if in_center and y2 / h > self.bbox_danger:   # la base del bbox está cerca
                return True
        return False

    def decide(self, free_space, detections, frame_shape):
        if not free_space["ready"]:
            return "STOP"                   # sin calibrar = modo seguro

        d = free_space["depth"]
        yolo = self._yolo_blocks(detections, frame_shape)
        limit = self.stop_depth if self.action == "FORWARD" else self.resume_depth

        if d["center"] >= limit and not yolo:
            self.action = "FORWARD"
        else:
            side = "left" if d["left"] > d["right"] else "right"
            if d[side] >= self.resume_depth:
                self.action = side.upper()
            else:
                self.action = "STOP"
        return self.action