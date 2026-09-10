"""
Analiza un video de yoga: dibuja el esqueleto (YOLO-Pose) y ademas predice
QUE postura de yoga se esta realizando (clasificador entrenado de 8 clases).

Uso:
    python video.py                     # video por defecto (videos/yoga.mp4)
    python video.py ruta/a/video.mp4

Esc para salir.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Modelo de pose (esqueleto / keypoints)
POSE_PATH = "yolo11m-pose.pt"

# Clasificador de posturas de yoga: se prefiere la copia estable
# clasificador/modelo.pt; si no existe, el best.pt recien entrenado.
_MODELO_PUBLICADO = BASE_DIR / "clasificador" / "modelo.pt"
_MODELO_ENTRENADO = BASE_DIR / "clasificador" / "runs" / "classify" / "train" / "weights" / "best.pt"
CLS_PATH = _MODELO_PUBLICADO if _MODELO_PUBLICADO.exists() else _MODELO_ENTRENADO

# Video de entrada por defecto
VIDEO_POR_DEFECTO = BASE_DIR / "videos" / "yoga.mp4"

# Clasificar 1 de cada N frames (ahorra CPU; entre medias se reutiliza
# la ultima prediccion). Subelo si va lento, bajalo si quieres mas reaccion.
CLASIFICAR_CADA = 5

# Confianza minima para dar por buena una postura de yoga
UMBRAL_CONFIANZA = 0.60
CLASE_NO_YOGA = "no_yoga"

# Traduccion de los nombres de clase del modelo a espanol
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


def recorte_persona(frame, pose_result, margen=0.15):
    """Recorta la caja de la persona mas grande detectada por el modelo de pose.

    Clasificar solo a la persona (y no todo el fotograma con su fondo) hace
    la prediccion mucho mas fiable, porque se parece a las imagenes con las
    que se entreno el clasificador. Devuelve None si no hay persona.
    """
    cajas = pose_result.boxes
    if cajas is None or len(cajas) == 0:
        return None

    xyxy = cajas.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[int(np.argmax(areas))]

    h, w = frame.shape[:2]
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def veredicto(cls_result):
    """(es_yoga, etiqueta_a_mostrar, confianza) a partir de la prediccion."""
    idx = int(cls_result.probs.top1)
    clase = cls_result.names[idx]
    conf = float(cls_result.probs.top1conf)

    if clase == CLASE_NO_YOGA or conf < UMBRAL_CONFIANZA:
        return False, "No es una postura de yoga", conf
    return True, NOMBRES_ES.get(clase, clase), conf


def main():
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else VIDEO_POR_DEFECTO
    if not entrada.exists():
        print(f"No existe el video: {entrada}")
        return
    if not Path(CLS_PATH).exists():
        print(f"No se encontro el clasificador en:\n  {CLS_PATH}")
        print("Entrena primero:  python clasificador/entrenar.py")
        return

    # Cargamos los dos modelos una sola vez
    pose_model = YOLO(POSE_PATH)
    cls_model = YOLO(str(CLS_PATH))

    cap = cv2.VideoCapture(str(entrada))
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {entrada}")
        return

    i = 0
    es_yoga, etiqueta, conf = False, "...", 0.0
    ultima_impresa = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pose: esqueleto + cajas de personas
        pose_result = pose_model(frame, verbose=False)[0]
        annotated = pose_result.plot()

        # 2. Clasificacion de la postura (1 de cada CLASIFICAR_CADA frames)
        if i % CLASIFICAR_CADA == 0:
            crop = recorte_persona(frame, pose_result)
            if crop is None:
                es_yoga, etiqueta, conf = False, "No se detecta persona", 0.0
            else:
                cls_result = cls_model(crop, verbose=False)[0]
                es_yoga, etiqueta, conf = veredicto(cls_result)

            if etiqueta != ultima_impresa:
                extra = f" ({conf:.0%})" if es_yoga else ""
                print(f"frame {i}: {etiqueta}{extra}")
                ultima_impresa = etiqueta
        i += 1

        # 3. Banda superior con el veredicto (verde = yoga, rojo = no)
        color = (0, 170, 0) if es_yoga else (0, 0, 200)
        texto = f"{etiqueta} ({conf:.0%})" if es_yoga else etiqueta
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), color, -1)
        cv2.putText(
            annotated, texto, (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        cv2.imshow("Yoga - esqueleto + postura", annotated)

        # El ciclo se rompe al presionar "Esc"
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
