from ultralytics import YOLO
import cv2
import time
import threading

# ----------------------------------------------------------------------
# Modulo de video en vivo: analiza los puntos (keypoints) con YOLO-Pose
#
# Optimizado para CPU (esta maquina no tiene CUDA):
#   - Modelo "nano" por defecto  -> mucho mas rapido que el "medium"
#   - imgsz reducido             -> menos pixeles que procesar
#   - Captura en un hilo aparte  -> siempre se infiere el frame mas nuevo
#     (elimina el retraso por acumulacion del buffer de la camara)
# ----------------------------------------------------------------------

# Indice de la camara (0 = webcam por defecto) o URL de un stream RTSP/IP
CAMERA_INDEX = 0

# Modelo YOLO de pose. Menor = mas FPS:
#   yolo11n-pose.pt  (nano,   ~15-30 fps CPU)
#   yolo11s-pose.pt  (small)
#   yolo11m-pose.pt  (medium, ~3 fps CPU)
MODEL_PATH = "yolo11n-pose.pt"

# Tamano al que se redimensiona el frame para la inferencia (multiplo de 32).
# Bajarlo sube los FPS a costa de precision: 640 / 480 / 416 / 320
IMGSZ = 416

# Resolucion pedida a la camara
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Umbral de confianza para dibujar un landmark
KP_CONF = 0.5

# Nombres de los 17 keypoints del formato COCO que usa YOLO-Pose
KEYPOINT_NAMES = [
    "nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der",
    "muneca_izq", "muneca_der", "cadera_izq", "cadera_der",
    "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der",
]


class CamaraHilo:
    """Lee la camara en un hilo y entrega siempre el ultimo frame disponible."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.parar = False
        self.lock = threading.Lock()
        self.hilo = threading.Thread(target=self._actualizar, daemon=True)
        self.hilo.start()

    def _actualizar(self):
        while not self.parar:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def leer(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            return self.ret, frame

    def abierta(self):
        return self.cap.isOpened()

    def liberar(self):
        self.parar = True
        self.hilo.join(timeout=1)
        self.cap.release()


def main():
    # Cargamos el modelo YOLO
    model = YOLO(MODEL_PATH)

    # Abrimos la camara en vivo (con hilo de captura)
    cam = CamaraHilo(CAMERA_INDEX)
    if not cam.abierta():
        print(f"No se pudo abrir la camara {CAMERA_INDEX}")
        return

    prev_time = time.time()
    fps_suave = 0.0

    while cam.abierta():
        # Leemos el frame mas reciente de la camara
        ret, frame = cam.leer()
        if not ret or frame is None:
            continue

        # Realizamos la inferencia de YOLO sobre el frame
        results = model(frame, conf=0.7, imgsz=IMGSZ, verbose=False)

        # Frame anotado por YOLO (esqueleto completo)
        annotated_frame = results[0].plot()

        # Analizamos los puntos de cada persona detectada
        keypoints = results[0].keypoints
        if keypoints is not None and keypoints.data.numel() > 0:
            for person_id, person in enumerate(keypoints.data):
                for i, (x, y, conf) in enumerate(person):
                    if conf > KP_CONF:
                        px, py = int(x), int(y)

                        # Resaltamos el punto
                        cv2.circle(annotated_frame, (px, py), 4, (0, 255, 0), -1)

                        # Etiquetamos el punto con su nombre
                        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
                        cv2.putText(
                            annotated_frame, name, (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                            cv2.LINE_AA,
                        )

                # Mostramos en consola los puntos validos de la persona
                validos = int((person[:, 2] > KP_CONF).sum())
                print(f"Persona {person_id}: {validos}/{len(person)} puntos detectados")

        # Calculamos y mostramos los FPS (suavizados)
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_suave = 0.9 * fps_suave + 0.1 * (1.0 / dt)
        cv2.putText(
            annotated_frame, f"FPS: {fps_suave:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )

        # Visualizamos los resultados
        cv2.imshow("YOLO Pose - Video en vivo", annotated_frame)

        # El ciclo se rompe al presionar "Esc"
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.liberar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
