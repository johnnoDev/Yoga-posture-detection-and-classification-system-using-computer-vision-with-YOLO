from ultralytics import YOLO
import cv2

# Cargamos el modelo YOLO
model = YOLO("yolo11m-pose.pt")

# Cargamos el video de entrada
video_path = "./videos/yoga.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Leemos el frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Realizamos la inferencia de YOLO sobre el frame
    results = model(frame)

    # Extraemos los resultados
    annotated_frame = results[0].plot()

    # Visualizamos los resultados
    cv2.imshow("YOLO Inference", annotated_frame)

    # El ciclo se rompe al presionar "Esc"
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()