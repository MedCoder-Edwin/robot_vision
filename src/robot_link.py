import time

# (v [m/s], w [rad/s]). Convenio ROS: w positivo = giro a la izquierda.
ACTION_TO_VEL = {
    "FORWARD": (0.15, 0.0),
    "LEFT":    (0.0,  0.8),
    "RIGHT":   (0.0, -0.8),
    "STOP":    (0.0,  0.0),
}

# Solo para el modo serial (comandos de tu firmware)
ACTION_TO_CHAR = {"FORWARD": b"w", "LEFT": b"a", "RIGHT": b"d", "STOP": b"x"}


class RobotLink:
    """Envía la acción de navegación al robot. Backends: dry | serial | ros2."""

    def __init__(self, backend="dry", port="COM3", baud=115200, rate_hz=10.0):
        self.backend = backend
        self.period = 1.0 / rate_hz
        self.last_sent = 0.0
        self.last_action = None
        self.current_cmd = ("STOP" , 0.0, 0.0)  # (acción, v, w)

        if backend == "serial":
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2.0)  # la ESP32 suele reiniciarse al abrir el puerto
        elif backend == "ros2":
            import rclpy
            from geometry_msgs.msg import Twist
            self._Twist = Twist
            rclpy.init()
            self.node = rclpy.create_node("vision_link")
            self.pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        elif backend != "dry":
            raise ValueError(f"Backend desconocido: {backend}")

    def send(self, action):
        """Llamar en cada iteración del loop. Limita la tasa de envío a rate_hz,
        pero un cambio de acción se manda de inmediato."""
        if action not in ACTION_TO_VEL:
            action = "STOP"                      # lo desconocido = modo seguro

        now = time.monotonic()
        changed = action != self.last_action
        if not changed and now - self.last_sent < self.period:
            return

        v, w = ACTION_TO_VEL[action]
        self.current_cmd = (action, v, w)
        self._write(action, v, w)
        self.last_action = action
        self.last_sent = now

    def _write(self, action, v, w):
        if self.backend == "dry":
            if action != self.last_action:       # imprime solo los cambios
                print(f"[LINK dry] {action}: v={v:+.2f} m/s, w={w:+.2f} rad/s")
        elif self.backend == "serial":
            if action != self.last_action:       # el firmware mantiene el último comando
                self.ser.write(ACTION_TO_CHAR[action])
        elif self.backend == "ros2":
            msg = self._Twist()
            msg.linear.x = v
            msg.angular.z = w
            self.pub.publish(msg)

    def close(self):
        """Frena el robot antes de salir."""
        for _ in range(3):
            self._write("STOP", 0.0, 0.0)
            time.sleep(0.05)
        if self.backend == "serial":
            self.ser.close()
        elif self.backend == "ros2":
            import rclpy
            self.node.destroy_node()
            rclpy.shutdown()