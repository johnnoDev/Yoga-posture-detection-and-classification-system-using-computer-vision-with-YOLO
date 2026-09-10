"""
Paso 5 del tutorial: usa el modelo ya entrenado para clasificar imagenes.

Notebook original:
    custom_model = YOLO("/content/runs/classify/train/weights/best.pt")
    res = custom_model("/content/train-7-pose-1/test/warrior pose")
    for r in res:
        r.show()

Cambios para uso local:
    - carga best.pt desde clasificador/runs/classify/train/weights/
    - en vez de r.show() (visor de Jupyter): imprime la clase predicha en
      consola y guarda cada imagen anotada en clasificador/predicciones/
    - acepta una ruta por linea de comandos (imagen o carpeta);
      por defecto usa la carpeta test/ del dataset

Uso:
    python clasificador/predecir.py
    python clasificador/predecir.py "clasificador/dataset/test/warrior pose"
    python clasificador/predecir.py ruta/a/una_imagen.jpg
"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import ruta_modelo, DATASET_DIR

# Entrada por defecto: las imagenes de test del dataset
ENTRADA_POR_DEFECTO = DATASET_DIR / "test"

# Carpeta donde guardamos las imagenes anotadas
DIR_SALIDA = Path(__file__).resolve().parent / "predicciones"


def main():
    # Necesitamos el modelo entrenado (copia publicada o best.pt)
    modelo = ruta_modelo()
    if not modelo.exists():
        print(f"No se encontro el modelo entrenado en {modelo}")
        print("Ejecuta primero:  python clasificador/entrenar.py")
        return

    # Ruta de entrada: argumento de linea de comandos o la de por defecto
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else ENTRADA_POR_DEFECTO
    if not entrada.exists():
        print(f"No existe la entrada: {entrada}")
        return

    # Cargamos el modelo YA entrenado
    model = YOLO(str(modelo))

    # YOLO acepta una imagen, una lista de imagenes o una carpeta entera
    resultados = model(str(entrada), verbose=False)

    DIR_SALIDA.mkdir(exist_ok=True)

    for r in resultados:
        # probs.top1 = indice de la clase mas probable; r.names lo traduce a texto
        clase = r.names[r.probs.top1]
        confianza = float(r.probs.top1conf)
        nombre_archivo = Path(r.path).name
        print(f"{nombre_archivo}  ->  {clase}  ({confianza:.1%})")

        # r.plot() devuelve la imagen (BGR) con el texto de la prediccion encima
        anotada = r.plot()
        cv2.imwrite(str(DIR_SALIDA / f"pred_{nombre_archivo}"), anotada)

    print(f"\nImagenes anotadas guardadas en: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
