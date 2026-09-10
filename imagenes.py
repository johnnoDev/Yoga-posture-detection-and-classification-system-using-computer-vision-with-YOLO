"""
Detector de posturas de yoga sobre una imagen.

Uso:
    python imagenes.py ruta/a/foto.jpg     # analiza esa imagen
    python imagenes.py ruta/a/carpeta/     # analiza todas las imagenes de la carpeta

Que hace:
    - Pasa la imagen por la IA entrenada (clasificador/modelo.pt) y dice que
      postura de yoga es, o avisa de que la imagen NO es una postura de yoga
      (clase "no_yoga" o confianza baja).
    - Halla los keypoints (puntos clave del cuerpo) de la persona con
      YOLO-Pose y dibuja el esqueleto + cada punto etiquetado en espanol.
    - Guarda una copia de la imagen anotada en resultados/.
"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Clasificador de posturas. Se prefiere la copia estable clasificador/modelo.pt
# (la que viaja con el repo); si no existe, se usa el best.pt recien entrenado.
_MODELO_PUBLICADO = BASE_DIR / "clasificador" / "modelo.pt"
_MODELO_ENTRENADO = BASE_DIR / "clasificador" / "runs" / "classify" / "train" / "weights" / "best.pt"
MODEL_PATH = _MODELO_PUBLICADO if _MODELO_PUBLICADO.exists() else _MODELO_ENTRENADO

# Modelo de pose para los keypoints. Mayor = mas preciso (mas lento):
#   yolo11n-pose.pt (nano)  yolo11m-pose.pt (medium)  yolo11x-pose.pt (extra)
POSE_PATH = "yolo11m-pose.pt"

# Carpeta donde se guardan las imagenes anotadas
DIR_SALIDA = BASE_DIR / "resultados"

# Confianza minima para dar por buena una postura de yoga.
UMBRAL_CONFIANZA = 0.60

# Confianza minima para dibujar un keypoint
KP_CONF = 0.5

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

EXTS_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clasificar(model, imagen):
    """Devuelve (es_yoga, clase, confianza, texto_veredicto)."""
    r = model(imagen, verbose=False)[0]

    idx = int(r.probs.top1)
    clase = r.names[idx]
    conf = float(r.probs.top1conf)
    nombre = NOMBRES_ES.get(clase, clase)

    if clase == CLASE_NO_YOGA:
        return False, clase, conf, "No se esta detectando una postura de yoga"

    if conf < UMBRAL_CONFIANZA:
        return (
            False, clase, conf,
            f"No se esta detectando una postura de yoga clara "
            f"(parecido: {nombre} {conf:.0%})",
        )

    return True, clase, conf, f"Postura de yoga detectada: {nombre} ({conf:.0%})"


def dibujar_keypoints(imagen_bgr, pose_result):
    """Dibuja esqueleto + puntos etiquetados. Devuelve (imagen, resumen)."""
    anotada = pose_result.plot()  # esqueleto y cajas que ya trae YOLO-Pose

    kps = pose_result.keypoints
    resumen = []
    if kps is not None and kps.data.numel() > 0:
        for person_id, persona in enumerate(kps.data):
            for i, (x, y, conf) in enumerate(persona):
                if conf > KP_CONF:
                    px, py = int(x), int(y)
                    cv2.circle(anotada, (px, py), 4, (0, 255, 0), -1)
                    nombre = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
                    cv2.putText(
                        anotada, nombre, (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
                    )
            validos = int((persona[:, 2] > KP_CONF).sum())
            resumen.append(f"persona {person_id}: {validos}/{len(persona)} puntos")
    else:
        resumen.append("no se detectaron personas")
    return anotada, resumen


def dibujar_banda(imagen, texto, es_yoga):
    """Banda de color en la parte superior con el veredicto."""
    color = (0, 170, 0) if es_yoga else (0, 0, 200)
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], 40), color, -1)
    cv2.putText(
        imagen, texto, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def main():
    if not MODEL_PATH.exists():
        print(f"No se encontro el clasificador en:\n  {MODEL_PATH}")
        print("Entrena primero:  python clasificador/entrenar.py")
        return

    if len(sys.argv) < 2:
        print("Pasa una imagen o una carpeta:")
        print("  python imagenes.py foto.jpg")
        print("  python imagenes.py img/")
        return

    entrada = Path(sys.argv[1])
    if not entrada.exists():
        print(f"No existe: {entrada}")
        return

    # Lista de imagenes a procesar
    if entrada.is_dir():
        imagenes = sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXTS_IMAGEN)
        if not imagenes:
            print(f"No hay imagenes en la carpeta: {entrada}")
            return
    else:
        imagenes = [entrada]

    # Cargamos los dos modelos una sola vez
    cls_model = YOLO(str(MODEL_PATH))
    pose_model = YOLO(POSE_PATH)

    DIR_SALIDA.mkdir(exist_ok=True)

    for ruta in imagenes:
        img = cv2.imread(str(ruta))
        if img is None:
            print(f"[ERROR] no se pudo abrir {ruta}")
            continue

        # 1. Que postura de yoga es (sobre la imagen completa)
        es_yoga, clase, conf, texto = clasificar(cls_model, img)

        # 2. Keypoints de la persona (esqueleto + puntos)
        pose_result = pose_model(img, verbose=False)[0]
        anotada, resumen_kp = dibujar_keypoints(img, pose_result)

        # 3. Veredicto encima
        dibujar_banda(anotada, texto, es_yoga)

        destino = DIR_SALIDA / f"{ruta.stem}_resultado.jpg"
        cv2.imwrite(str(destino), anotada)

        marca = "[YOGA]" if es_yoga else "[  -  ]"
        print(f"{marca} {ruta.name}: {texto}")
        for linea in resumen_kp:
            print(f"        {linea}")
        print(f"        -> {destino}")

    # Si fue una sola imagen, la mostramos en pantalla
    if len(imagenes) == 1:
        salida = DIR_SALIDA / f"{imagenes[0].stem}_resultado.jpg"
        if salida.exists():
            cv2.imshow("Detector de yoga", cv2.imread(str(salida)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
