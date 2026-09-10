"""
Bonus del tutorial: prueba el clasificador con la webcam en vivo.

Notebook original (era para deteccion; aqui lo adaptamos a clasificacion):
    model = YOLO("best.pt")
    cap = cv2.VideoCapture("./video_001.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        ...
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

Cambios para uso local:
    - entrada = webcam (CAMERA_INDEX) en vez de un archivo de video
    - carga best.pt desde clasificador/runs/classify/train/weights/
    - dibujamos a mano la clase + confianza (en clasificacion no hay cajas)

Uso:
    python clasificador/clasificar_en_vivo.py     (Esc para salir)
"""
import cv2
from ultralytics import YOLO

from config import ruta_modelo

# Indice de la camara (0 = webcam por defecto) o URL de un stream RTSP/IP
CAMERA_INDEX = 0


def main():
    modelo = ruta_modelo()
    if not modelo.exists():
        print(f"No se encontro el modelo entrenado en {modelo}")
        print("Ejecuta primero:  python clasificador/entrenar.py")
        return

    # Cargamos el modelo entrenado
    model = YOLO(str(modelo))

    # Abrimos la camara
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {CAMERA_INDEX}")
        return

    while cap.isOpened():
        # Leemos el frame de la camara
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia de clasificacion sobre el frame
        r = model(frame, verbose=False)[0]

        # Clase mas probable y su confianza
        clase = r.names[r.probs.top1]
        confianza = float(r.probs.top1conf)

        # Dibujamos la etiqueta sobre el frame (fondo + texto)
        cv2.rectangle(frame, (0, 0), (360, 40), (255, 0, 255), -1)
        cv2.putText(
            frame, f"{clase}  {confianza:.1%}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Visualizamos los resultados
        cv2.imshow("YOLOv11 - Clasificacion en vivo", frame)

        # El ciclo se rompe al presionar "Esc"
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
