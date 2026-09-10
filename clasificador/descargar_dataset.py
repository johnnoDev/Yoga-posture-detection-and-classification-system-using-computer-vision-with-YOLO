"""
Paso 1 y 2 del tutorial: descarga el dataset personalizado desde Roboflow.

Notebook original:
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR-API-KEY")
    project = rf.workspace("upi-awhgd").project("train-7-pose")
    version = project.version(1)
    dataset = version.download("folder")

Cambios para uso local:
    - la API key se lee de la variable de entorno ROBOFLOW_API_KEY
      (no se escribe en el codigo para no subirla a git)
    - el dataset se descarga en clasificador/dataset/ (no en /content/)
    - se renombra valid/ -> val/ porque Ultralytics espera esa carpeta

Uso:
    PowerShell:  $env:ROBOFLOW_API_KEY = "tu_api_key"
    python clasificador/descargar_dataset.py
"""
import os
import sys

from config import (
    DATASET_DIR,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
)


def main():
    # 1. Leemos la API key del entorno (no del codigo)
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Falta la API key de Roboflow.")
        print("Consiguela en https://app.roboflow.com -> Settings -> API y luego:")
        print('  PowerShell:  $env:ROBOFLOW_API_KEY = "tu_api_key"')
        print("  cmd:         set ROBOFLOW_API_KEY=tu_api_key")
        sys.exit(1)

    # roboflow es una dependencia extra (ver clasificador/requirements.txt)
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Falta la libreria 'roboflow'. Instalala con:")
        print("  pip install roboflow")
        sys.exit(1)

    # 2. Conexion con Roboflow y seleccion de proyecto + version
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    # 3. Descarga en formato "folder" (una subcarpeta por clase),
    #    directamente dentro de clasificador/dataset/
    print(f"Descargando dataset en: {DATASET_DIR}")
    version.download("folder", location=str(DATASET_DIR), overwrite=True)

    # 4. Roboflow exporta la carpeta de validacion como 'valid';
    #    Ultralytics la busca como 'val'. La renombramos si hace falta.
    valid_dir = DATASET_DIR / "valid"
    val_dir = DATASET_DIR / "val"
    if valid_dir.is_dir() and not val_dir.exists():
        valid_dir.rename(val_dir)
        print("Renombrado: valid/ -> val/")

    # 5. Mostramos la estructura resultante para verificar
    print("\nDataset descargado. Contenido:")
    for sub in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        clases = sorted(p.name for p in sub.iterdir() if p.is_dir())
        print(f"  {sub.name}/: {', '.join(clases) if clases else '(sin clases)'}")


if __name__ == "__main__":
    main()
