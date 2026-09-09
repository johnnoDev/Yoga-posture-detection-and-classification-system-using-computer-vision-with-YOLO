from ultralytics import YOLO
import cv2

# Cargamos la imagen de entrada
image = cv2.imread("./img/negra.webp")

# Cargamos el modelo YOLO
model = YOLO("yolo11l-pose.pt")

# Realizamos la inferencia sobre la imagen
results = model(image, conf=0.7)

# Explorando los resultados
print(results[0])
print(results[0].keypoints)

# Acceder a un landmark
x, y, conf = results[0].keypoints.data[0][10]
#cv2.circle(image, (int(x), int(y)), 5, (0,255, 255), 2)

# Acceder a los landmarks de la persona detectada
for x, y, conf in results[0].keypoints.data[0]:
    if conf > 0.5:
        
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.imshow(f"image", image)
        cv2.waitKey(0)
'''
# Acceder a los landmarks de cada persona detectada
for person in results[0].keypoints.data:
    for i, (x, y, conf) in enumerate(person):
        color = (255, 255, 255)
        if conf > 0.5:
            if i <= 4:
                color = (0, 255, 0)
            elif i == 5:
                color = (0, 0, 0)
            elif 6 <= i <= 11:
                color = (255, 255, 0)
            cv2.circle(image, (int(x), int(y)), 5, color, 2)
'''
cv2.imshow(f"image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()