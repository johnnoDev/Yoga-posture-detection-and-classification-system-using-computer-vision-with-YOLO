# Detección de posiciones del cuerpo con YOLO Pose Estimation

Detección de *keypoints* (puntos clave) del cuerpo humano usando **YOLO11 Pose** de
Ultralytics sobre imágenes, videos y cámara en vivo.

## Proyecto de referencia

Este repositorio es una implementación práctica basada en el tutorial de **OMES
(omes-va.com)**:

- **Video:** [🚶 Aprende cómo detectar posiciones del cuerpo con YOLO (¡Pose Estimation fácil!)](https://youtu.be/AcBcTamrHPc?si=Q4lqsY9bsuzaU-QX)
- **Blog del autor:** [Aprende cómo detectar posiciones del cuerpo con YOLO Pose Estimation](https://omes-va.com/aprende-como-detectar-posiciones-del-cuerpo-con-yolo-pose-estimation/)
- **Sitio del autor:** [omes-va.com](https://omes-va.com/)

El proyecto añade sobre el tutorial original un módulo de **cámara en vivo optimizado
para CPU** ([video_en_vivo.py](video_en_vivo.py)) y el etiquetado de cada punto con su
nombre en español.

## ¿Qué es Pose Estimation?

YOLO-Pose detecta a cada persona de la escena y estima **17 puntos clave del formato
COCO** en una sola pasada:

| # | Keypoint | # | Keypoint | # | Keypoint |
|---|----------|---|----------|---|----------|
| 0 | nariz | 6 | hombro derecho | 12 | cadera derecha |
| 1 | ojo izquierdo | 7 | codo izquierdo | 13 | rodilla izquierda |
| 2 | ojo derecho | 8 | codo derecho | 14 | rodilla derecha |
| 3 | oreja izquierda | 9 | muñeca izquierda | 15 | tobillo izquierdo |
| 4 | oreja derecha | 10 | muñeca derecha | 16 | tobillo derecho |
| 5 | hombro izquierdo | 11 | cadera izquierda | | |

Cada punto viene con una coordenada `(x, y)` y una **confianza** entre 0 y 1.

## Requisitos

- **Python 3.10 – 3.12**
- Sistema operativo: Windows, Linux o macOS
- No se requiere GPU (funciona en CPU; una GPU NVIDIA con CUDA lo acelera)

### Librerías principales

| Librería | Para qué se usa |
|----------|-----------------|
| [`ultralytics`](https://docs.ultralytics.com/) | Modelo YOLO11 Pose e inferencia |
| [`opencv-python`](https://pypi.org/project/opencv-python/) | Leer/mostrar imágenes y video, dibujar los puntos |
| `torch` / `torchvision` | Backend de deep learning que usa Ultralytics |
| `numpy`, `matplotlib`, `pillow` | Dependencias de procesamiento e imagen |

El listado completo con versiones fijadas está en [requirements.txt](requirements.txt).

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/johnnoDev/Yoga-posture-detection-and-classification-system-using-computer-vision-with-YOLO.git
cd Yoga-posture-detection-and-classification-system-using-computer-vision-with-YOLO

# 2. Crear y activar un entorno virtual
python -m venv venv

#    Windows (PowerShell)
venv\Scripts\Activate.ps1
#    Windows (cmd)
venv\Scripts\activate.bat
#    Linux / macOS
source venv/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt
```

Instalación mínima (sin versiones fijadas):

```bash
pip install ultralytics opencv-python
```

`torch` y `torchvision` se instalan automáticamente como dependencia de
`ultralytics`.

### Modelos

Los pesos `yolo11*-pose.pt` **se descargan solos** la primera vez que se ejecuta el
script. De menor a mayor tamaño (más lento pero más preciso):

| Modelo | Tamaño | Uso recomendado |
|--------|--------|-----------------|
| `yolo11n-pose.pt` | nano | cámara en vivo / CPU |
| `yolo11m-pose.pt` | medium | video |
| `yolo11l-pose.pt` | large | imágenes |
| `yolo11x-pose.pt` | extra large | máxima precisión |

## Uso del software

Coloca tus archivos de entrada en `img/` (imágenes) y `videos/` (videos), y ajusta la
ruta en la parte superior del script correspondiente.

### 1. Imagen fija — [imagenes.py](imagenes.py)

Detecta la pose sobre una imagen, dibuja el esqueleto + los puntos con su nombre,
guarda el resultado en `img/<nombre>_pose.jpg` y lo muestra en pantalla.

```bash
python imagenes.py
```

Parámetros configurables al inicio del archivo: `IMAGE_PATH`, `MODEL_PATH`,
`DET_CONF` (confianza mínima para detectar persona), `KP_CONF` (confianza mínima para
dibujar un punto), `OUTPUT_PATH`.

### 2. Video — [video.py](video.py)

Procesa un archivo de video cuadro por cuadro y muestra el esqueleto en tiempo real.

```bash
python video.py
```

Ruta del video en la variable `video_path`. Pulsa **Esc** para salir.

### 3. Cámara en vivo — [video_en_vivo.py](video_en_vivo.py)

Análisis de la webcam optimizado para CPU: modelo *nano*, `imgsz` reducido y captura
en un hilo aparte para procesar siempre el cuadro más reciente. Muestra FPS, resalta
cada punto y lo etiqueta con su nombre; imprime en consola cuántos puntos válidos
tiene cada persona.

```bash
python video_en_vivo.py
```

Parámetros al inicio del archivo: `CAMERA_INDEX` (0 = webcam por defecto, o una URL
RTSP/IP), `MODEL_PATH`, `IMGSZ`, `CAM_WIDTH`/`CAM_HEIGHT`, `KP_CONF`. Pulsa **Esc**
para salir.

## Cómo acceder a los puntos por código

```python
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
results = model("./img/camilo.jpeg", conf=0.5)

# Imagen anotada (cajas + esqueleto), como array BGR de OpenCV
annotated = results[0].plot()

# Keypoints: tensor de forma (personas, 17, 3) -> (x, y, confianza)
for persona in results[0].keypoints.data:
    for i, (x, y, conf) in enumerate(persona):
        if conf > 0.5:
            print(i, int(x), int(y))
```

## Estructura del repositorio

```
.
├── imagenes.py         # inferencia sobre una imagen
├── video.py            # inferencia sobre un archivo de video
├── video_en_vivo.py    # inferencia sobre la webcam (optimizado CPU)
├── requirements.txt    # dependencias con versiones
├── img/                # imágenes de entrada
├── videos/             # videos de entrada
└── yolo11*-pose.pt     # pesos del modelo (se descargan automáticamente)
```

> Nota: `.gitignore` excluye `venv/`, los pesos `*.pt` y los archivos multimedia. Si
> `requirements.txt` no aparece en tu control de versiones es por la regla `*.txt`;
> fuérzalo con `git add -f requirements.txt`.

## Créditos

Tutorial original: **OMES** — [omes-va.com](https://omes-va.com/) ·
[video en YouTube](https://youtu.be/AcBcTamrHPc?si=Q4lqsY9bsuzaU-QX).
Modelo: **Ultralytics YOLO11 Pose** — [docs.ultralytics.com/tasks/pose](https://docs.ultralytics.com/tasks/pose/).
