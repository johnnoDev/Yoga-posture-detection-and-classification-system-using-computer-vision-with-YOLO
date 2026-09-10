"""
Configuracion compartida del clasificador de imagenes con YOLOv11.

Los 4 scripts del modulo (descargar_dataset, entrenar, predecir,
clasificar_en_vivo) importan estos valores. Ajusta aqui las rutas,
el modelo base y los parametros de entrenamiento.
"""
from pathlib import Path

# Carpeta de este modulo -> .../yolotrolll/clasificador
BASE_DIR = Path(__file__).resolve().parent

# Carpeta raiz del repo -> .../yolotrolll
REPO_DIR = BASE_DIR.parent

# Cargamos las variables del archivo .env (raiz del repo) a os.environ,
# si python-dotenv esta instalado. Asi la ROBOFLOW_API_KEY sale del .env
# sin tener que exportarla a mano en la terminal.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_DIR / ".env")
except ImportError:
    pass

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
# Carpeta local donde se descarga el dataset de Roboflow.
# Para clasificacion, YOLO espera esta estructura (una carpeta por clase):
#
#   dataset/
#     train/
#       warrior pose/   *.jpg
#       downdog/        *.jpg
#       ...
#     val/
#       warrior pose/   *.jpg
#       ...
#     test/             (opcional)
#       ...
DATASET_DIR = BASE_DIR / "dataset"

# Datos del dataset del tutorial en Roboflow.
# La API key NO va aqui: se lee de la variable de entorno ROBOFLOW_API_KEY
# (ver descargar_dataset.py). Consiguela gratis en:
#   https://app.roboflow.com  ->  Settings  ->  API
ROBOFLOW_WORKSPACE = "upi-awhgd"
ROBOFLOW_PROJECT = "train-7-pose"
ROBOFLOW_VERSION = 1

# ----------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------
# Modelo base de clasificacion de Ultralytics (se descarga solo la 1a vez).
# Menor = entrena mas rapido en CPU (esta maquina no tiene GPU):
#   yolo11n-cls.pt (nano)    yolo11s-cls.pt (small)
#   yolo11m-cls.pt (medium, el que usa el tutorial)
#   yolo11l-cls.pt (large)   yolo11x-cls.pt (extra large)
MODELO_BASE = "yolo11m-cls.pt"

# Ruta donde entrenar.py deja el modelo ya entrenado.
# predecir.py y clasificar_en_vivo.py lo cargan desde aqui.
DIR_ENTRENAMIENTO = BASE_DIR / "runs" / "classify" / "train"
MODELO_ENTRENADO = DIR_ENTRENAMIENTO / "weights" / "best.pt"

# ----------------------------------------------------------------------
# Parametros de entrenamiento
# ----------------------------------------------------------------------
EPOCHS = 20      # vueltas completas al dataset (igual que el tutorial)
IMGSZ = 224      # tamano de entrada tipico en clasificacion
DEVICE = "cpu"   # "cpu" o "0" para la primera GPU CUDA
