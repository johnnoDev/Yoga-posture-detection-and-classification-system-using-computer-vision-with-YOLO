from ultralytics import YOLO

# Cargamos el modelo YOLO
model = YOLO("yolo11x-pose.pt")

# Cargamos la imagen
image = "./img/negra.webp"

# Realizamos la inferencia de YOLO
results = model(image, conf=0.7)

# Visualizar los resultados
for res in results:
    res.show()