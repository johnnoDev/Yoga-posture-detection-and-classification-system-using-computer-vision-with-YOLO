"""
Pasos 3 y 4 del tutorial: carga el modelo base YOLOv11-cls y entrena
un clasificador personalizado con el dataset descargado.

Notebook original:
    from ultralytics import YOLO
    model = YOLO("yolo11m-cls.pt")
    data_path = "/content/train-7-pose-1"
    results = model.train(data=data_path, epochs=20)

Cambios para uso local:
    - data apunta a clasificador/dataset/ (no a /content/)
    - device="cpu" explicito (esta maquina no tiene GPU CUDA)
    - project/name/exist_ok fijan la carpeta de salida en
      clasificador/runs/classify/train para que predecir.py encuentre best.pt

Uso:
    python clasificador/entrenar.py
"""
from config import (
    DATASET_DIR,
    MODELO_BASE,
    MODELO_ENTRENADO,
    EPOCHS,
    IMGSZ,
    DEVICE,
    BASE_DIR,
)
from ultralytics import YOLO


def main():
    # Verificamos que el dataset exista antes de empezar
    if not DATASET_DIR.is_dir():
        print(f"No se encontro el dataset en {DATASET_DIR}")
        print("Ejecuta primero:  python clasificador/descargar_dataset.py")
        return

    # 3. Cargamos el modelo base de clasificacion (se descarga la 1a vez)
    model = YOLO(MODELO_BASE)

    # 4. Entrenamos el clasificador personalizado
    model.train(
        data=str(DATASET_DIR),          # carpeta con train/ y val/
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=DEVICE,
        project=str(BASE_DIR / "runs" / "classify"),
        name="train",
        exist_ok=True,                  # sobrescribe la carpeta 'train' anterior
    )

    print("\nEntrenamiento terminado.")
    print(f"Modelo entrenado: {MODELO_ENTRENADO}")


if __name__ == "__main__":
    main()
