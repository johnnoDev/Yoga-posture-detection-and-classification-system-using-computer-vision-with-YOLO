"""
Detector de posturas de yoga sobre una imagen.

Uso:
    python imagenes.py                     # analiza la imagen por defecto
    python imagenes.py ruta/a/foto.jpg     # analiza esa imagen
    python imagenes.py ruta/a/carpeta/     # analiza todas las imagenes de la carpeta

Que hace:
    - Pasa la imagen por la IA entrenada (clasificador/runs/.../best.pt)
    - Si reconoce una de las 7 posturas de yoga con confianza suficiente,
      dice cual es.
    - Si la imagen cae en la clase "no_yoga" o la confianza es baja, avisa
      de que NO se esta detectando una postura de yoga.
    - Guarda una copia de la imagen con el veredicto escrito encima.
"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Modelo a usar. Se prefiere la copia estable clasificador/modelo.pt (la que
# viaja con el repo); si no existe, se usa el best.pt recien entrenado.
_MODELO_PUBLICADO = BASE_DIR / "clasificador" / "modelo.pt"
_MODELO_ENTRENADO = BASE_DIR / "clasificador" / "runs" / "classify" / "train" / "weights" / "best.pt"
MODEL_PATH = _MODELO_PUBLICADO if _MODELO_PUBLICADO.exists() else _MODELO_ENTRENADO

# Imagen que se analiza si no pasas ninguna por la linea de comandos
IMAGEN_POR_DEFECTO = BASE_DIR / "img" / "kenichan.png"

# Carpeta donde se guardan las imagenes con el veredicto
DIR_SALIDA = BASE_DIR / "resultados"

# Confianza minima para dar por buena una postura de yoga.
# Por debajo de esto se considera que NO hay una postura clara.
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

EXTS_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clasificar(model, ruta_imagen):
    """Devuelve (es_yoga, etiqueta, confianza, texto_veredicto)."""
    r = model(str(ruta_imagen), verbose=False)[0]

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


def dibujar_veredicto(ruta_imagen, texto, es_yoga):
    """Escribe el veredicto sobre la imagen y la guarda en DIR_SALIDA."""
    img = cv2.imread(str(ruta_imagen))
    if img is None:
        return None

    verde = (0, 170, 0)
    rojo = (0, 0, 200)
    color = verde if es_yoga else rojo

    # Banda de color en la parte superior con el texto encima
    alto_banda = 40
    cv2.rectangle(img, (0, 0), (img.shape[1], alto_banda), color, -1)
    cv2.putText(
        img, texto, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )

    DIR_SALIDA.mkdir(exist_ok=True)
    destino = DIR_SALIDA / f"{ruta_imagen.stem}_resultado.jpg"
    cv2.imwrite(str(destino), img)
    return destino


def main():
    if not MODEL_PATH.exists():
        print(f"No se encontro el modelo entrenado en:\n  {MODEL_PATH}")
        print("Entrena primero:  python clasificador/entrenar.py")
        return

    # Ruta a analizar: argumento de la linea de comandos o la de por defecto
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGEN_POR_DEFECTO
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

    # Cargamos la IA una sola vez
    model = YOLO(str(MODEL_PATH))

    for ruta in imagenes:
        es_yoga, clase, conf, texto = clasificar(model, ruta)

        marca = "[YOGA]" if es_yoga else "[  -  ]"
        print(f"{marca} {ruta.name}: {texto}")

        destino = dibujar_veredicto(ruta, texto, es_yoga)
        if destino:
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
