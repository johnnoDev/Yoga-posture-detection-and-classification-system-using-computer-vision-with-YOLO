from ultralytics import YOLO
import cv2

# ----------------------------------------------------------------------
# Modulo de imagenes: analiza los puntos (keypoints) con YOLO-Pose
# sobre una imagen fija y dibuja el esqueleto + los landmarks.
# ----------------------------------------------------------------------

# Imagen de entrada
IMAGE_PATH = "./img/kenichan.png"

# Modelo YOLO de pose. Mayor = mas preciso (mas lento):
#   yolo11n-pose.pt  (nano)
#   yolo11m-pose.pt  (medium)
#   yolo11l-pose.pt  (large)
#   yolo11x-pose.pt  (extra large)
MODEL_PATH = "yolo11x-pose.pt"

# Umbral de confianza para detectar una persona
DET_CONF = 0.5

# Umbral de confianza para dibujar un landmark
KP_CONF = 0.5

# Ruta donde se guarda la imagen anotada
OUTPUT_PATH = "./img/camilo_pose.jpg"

# Nombres de los 17 keypoints del formato COCO que usa YOLO-Pose
KEYPOINT_NAMES = [
    "nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der",
    "muneca_izq", "muneca_der", "cadera_izq", "cadera_der",
    "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der",
]


def main():
    # Cargamos el modelo YOLO
    model = YOLO(MODEL_PATH)

    # Cargamos la imagen de entrada
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"No se pudo abrir la imagen: {IMAGE_PATH}")
        return

    # Realizamos la inferencia de YOLO sobre la imagen
    results = model(image, conf=DET_CONF, verbose=False)

    # Frame anotado por YOLO (cajas + esqueleto completo)
    annotated = results[0].plot()

    # Analizamos los puntos de cada persona detectada
    keypoints = results[0].keypoints
    if keypoints is None or keypoints.data.numel() == 0:
        print("No se detectaron personas en la imagen.")
    else:
        for person_id, person in enumerate(keypoints.data):
            for i, (x, y, conf) in enumerate(person):
                if conf > KP_CONF:
                    px, py = int(x), int(y)

                    # Resaltamos el punto
                    cv2.circle(annotated, (px, py), 4, (0, 255, 0), -1)

                    # Etiquetamos el punto con su nombre
                    name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
                    cv2.putText(
                        annotated, name, (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                        cv2.LINE_AA,
                    )

            # Mostramos en consola los puntos validos de la persona
            validos = int((person[:, 2] > KP_CONF).sum())
            print(f"Persona {person_id}: {validos}/{len(person)} puntos detectados")

    # Guardamos la imagen anotada
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"Imagen anotada guardada en: {OUTPUT_PATH}")

    # Visualizamos los resultados
    cv2.imshow("YOLO Pose - Imagen", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
