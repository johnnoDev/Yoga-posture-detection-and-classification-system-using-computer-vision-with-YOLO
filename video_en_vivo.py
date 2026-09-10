from pathlib import Path
import time
import threading

import cv2
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Modulo de video en vivo: identifica la postura de yoga del usuario.
#
# Usa DOS modelos a la vez sobre la webcam:
#   1. YOLO-Pose  -> keypoints (esqueleto) de la persona
#   2. clasificador/modelo.pt (YOLO11-cls entrenado) -> nombre de la postura
#
# Optimizado para CPU (esta maquina no tiene CUDA):
#   - Pose "nano" + imgsz reducido
#   - La clasificacion se corre 1 de cada N frames (es la parte lenta)
#   - Se clasifica solo el recorte de la persona (mas rapido y preciso)
#   - Se suavizan las probabilidades con una media movil (EMA) para que
#     la etiqueta no parpadee entre frames
#   - Captura en un hilo aparte -> siempre se infiere el frame mas nuevo
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Indice de la camara (0 = webcam por defecto) o URL de un stream RTSP/IP
CAMERA_INDEX = 0

# Modelo YOLO de pose. Menor = mas FPS:
#   yolo11n-pose.pt (nano)  yolo11s-pose.pt (small)  yolo11m-pose.pt (medium)
POSE_PATH = "yolo11n-pose.pt"

# Clasificador de posturas. Se prefiere la copia estable clasificador/modelo.pt;
# si no existe, se usa el best.pt recien entrenado.
_CLS_PUBLICADO = BASE_DIR / "clasificador" / "modelo.pt"
_CLS_ENTRENADO = BASE_DIR / "clasificador" / "runs" / "classify" / "train" / "weights" / "best.pt"
CLS_PATH = _CLS_PUBLICADO if _CLS_PUBLICADO.exists() else _CLS_ENTRENADO

# Tamano al que se redimensiona el frame para la inferencia de pose (multiplo de 32)
IMGSZ = 416

# Resolucion pedida a la camara
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Umbral de confianza para dibujar un landmark
KP_CONF = 0.5

# ---- Parametros de la clasificacion ----------------------------------
# Cada cuantos frames se vuelve a clasificar la postura (subelo si va lento)
CLASIFICAR_CADA_N = 4
# Peso del frame nuevo en la media movil (0-1). Menor = mas estable, mas lento en reaccionar
EMA_ALPHA = 0.4
# Confianza minima para dar por buena una postura
UMBRAL_CONFIANZA = 0.60
# Nombre de la clase "no es yoga" (tal cual esta en el dataset)
CLASE_NO_YOGA = "no_yoga"

# Traduccion de los nombres de clase a algo legible en espanol
NOMBRES_ES = {
    "bridge pose": "postura del puente",
    "cobra pose": "postura de la cobra",
    "downward dog pose": "perro boca abajo",
    "mountain pose": "postura de la montana",
    "tree pose": "postura del arbol",
    "triangle pose": "postura del triangulo",
    "warrior pose": "postura del guerrero",
    "no_yoga": "no es yoga",
}

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


def recorte_persona(pose_result, shape):
    """Devuelve (x1, y1, x2, y2) de la persona mas grande, con margen; o None."""
    boxes = pose_result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[int(areas.argmax())]

    h, w = shape[:2]
    mx = (x2 - x1) * 0.12
    my = (y2 - y1) * 0.12
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return x1, y1, x2, y2


def veredicto(clase, conf):
    """Devuelve (texto, color_bgr, es_yoga) a partir de la clase y confianza."""
    nombre = NOMBRES_ES.get(clase, clase)
    if clase == CLASE_NO_YOGA:
        return "No se detecta una postura de yoga", (0, 0, 200), False
    if conf < UMBRAL_CONFIANZA:
        return f"Postura poco clara (parece: {nombre} {conf:.0%})", (0, 140, 220), False
    return f"{nombre}  ({conf:.0%})", (0, 170, 0), True


def dibujar_banda(frame, texto, color, fps):
    """Banda superior con el veredicto de la postura y los FPS."""
    ancho = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (ancho, 40), color, -1)
    cv2.putText(
        frame, texto, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"{fps:.0f} FPS", (ancho - 90, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def main():
    # Cargamos el modelo de pose
    pose_model = YOLO(POSE_PATH)

    # Cargamos el clasificador de posturas (obligatorio para este modulo)
    if not CLS_PATH.exists():
        print(f"No se encontro el clasificador en:\n  {CLS_PATH}")
        print("Entrena primero:  python clasificador/entrenar.py")
        return
    cls_model = YOLO(str(CLS_PATH))

    # Abrimos la camara en vivo (con hilo de captura)
    cam = CamaraHilo(CAMERA_INDEX)
    if not cam.abierta():
        print(f"No se pudo abrir la camara {CAMERA_INDEX}")
        return

    prev_time = time.time()
    fps_suave = 0.0

    frame_id = 0
    probs_ema = None                       # media movil de las probabilidades
    texto_postura = "Analizando..."
    color_banda = (128, 128, 128)

    while cam.abierta():
        # Leemos el frame mas reciente de la camara
        ret, frame = cam.leer()
        if not ret or frame is None:
            continue
        frame_id += 1

        # 1. Pose: keypoints de la persona
        pose_result = pose_model(frame, conf=0.7, imgsz=IMGSZ, verbose=False)[0]
        annotated_frame = pose_result.plot()

        # 2. Clasificacion de la postura (1 de cada N frames)
        if frame_id % CLASIFICAR_CADA_N == 0:
            bbox = recorte_persona(pose_result, frame.shape)
            roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame

            rc = cls_model(roi, verbose=False)[0]
            probs = rc.probs.data.cpu().numpy()

            if probs_ema is None:
                probs_ema = probs
            else:
                probs_ema = EMA_ALPHA * probs + (1.0 - EMA_ALPHA) * probs_ema

            idx = int(probs_ema.argmax())
            clase = cls_model.names[idx]
            conf = float(probs_ema[idx])
            texto_postura, color_banda, _ = veredicto(clase, conf)

        # 3. Keypoints resaltados y etiquetados
        keypoints = pose_result.keypoints
        if keypoints is not None and keypoints.data.numel() > 0:
            for person in keypoints.data:
                for i, (x, y, conf) in enumerate(person):
                    if conf > KP_CONF:
                        px, py = int(x), int(y)
                        cv2.circle(annotated_frame, (px, py), 4, (0, 255, 0), -1)
                        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
                        cv2.putText(
                            annotated_frame, name, (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                            cv2.LINE_AA,
                        )

        # 4. FPS (suavizados) + banda con la postura detectada
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_suave = 0.9 * fps_suave + 0.1 * (1.0 / dt)
        dibujar_banda(annotated_frame, texto_postura, color_banda, fps_suave)

        # Visualizamos los resultados
        cv2.imshow("YOLO Pose + Clasificador de yoga - Video en vivo", annotated_frame)

        # El ciclo se rompe al presionar "Esc"
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.liberar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
