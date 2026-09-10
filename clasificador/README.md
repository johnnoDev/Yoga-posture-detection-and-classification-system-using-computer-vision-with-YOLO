# Clasificador de imágenes con YOLOv11

Adaptación a scripts locales del tutorial de OMES
*[Crea tu propio clasificador de imágenes con YOLOv11](https://omes-va.com/crea-tu-propio-clasificador-de-imagenes-con-yolov11/)*,
que originalmente estaba en un notebook de Google Colab.

A diferencia del resto del repo (que hace *pose estimation*), este módulo
**entrena un clasificador**: dada una imagen, predice a qué clase pertenece
(p. ej. qué postura de yoga es).

## Instalación

```bash
# desde la raíz del repo, con el venv activado
pip install -r clasificador/requirements.txt   # añade 'python-dotenv'
```

> ⚠️ **No instales `roboflow` junto al resto.** Arrastra `opencv-python-headless`,
> que pisa a `opencv-python` y hace que `cv2.imshow` (las ventanas de
> `video.py`, `video_en_vivo.py`, `imagenes.py`) falle con
> *"The function is not implemented"*. `roboflow` solo hace falta para el
> paso 1 (descargar el dataset); instálalo aparte y límpialo después:
>
> ```bash
> pip install roboflow
> python clasificador/descargar_dataset.py
> pip uninstall -y roboflow opencv-python-headless
> pip install --force-reinstall opencv-python
> ```

## Pasos

| # | Script | Qué hace |
|---|--------|----------|
| 1 | `descargar_dataset.py` | Baja el dataset de Roboflow a `clasificador/dataset/` |
| 2 | `preparar_negativos.py` | Añade la clase `no_yoga` (imágenes que **no** son yoga) al dataset |
| 3 | `entrenar.py` | Entrena `yolo11m-cls.pt` con ese dataset (20 epochs, CPU) |
| 4 | `predecir.py` | Clasifica imágenes con el modelo entrenado y guarda el resultado |
| 5 | `clasificar_en_vivo.py` | Clasifica lo que ve la webcam en tiempo real (Esc para salir) |

`config.py` centraliza rutas, modelo base y parámetros de entrenamiento.

El detector final para el usuario es [`../imagenes.py`](../imagenes.py):
`python imagenes.py foto.jpg` → dice qué postura de yoga es, o avisa de que
la imagen **no** es una postura de yoga (clase `no_yoga` o confianza baja).

### 1. Descargar el dataset

Necesitas tu *Private API Key* de [Roboflow](https://app.roboflow.com)
(icono de tu cuenta → **Settings** → **API Keys**). **No se escribe en el
código**: se lee de la variable de entorno `ROBOFLOW_API_KEY`.

La forma más cómoda es un archivo `.env` en la raíz del repo (lo carga
`config.py` con `python-dotenv`; está en `.gitignore`, no se sube):

```powershell
Copy-Item .env.example .env      # crea el .env desde la plantilla
notepad .env                     # pega tu key después del '='
python clasificador/descargar_dataset.py
```

El `.env` queda así:

```
ROBOFLOW_API_KEY=abcd1234tuKeyReal
```

Alternativa sin `.env` (variable de entorno a mano, solo esa terminal):

```powershell
$env:ROBOFLOW_API_KEY = "tu_api_key"; python clasificador/descargar_dataset.py
```

O permanente en Windows: `setx ROBOFLOW_API_KEY "tu_api_key"` y reinicia VSCode.

El script descarga en formato *folder* (una subcarpeta por clase) y renombra
`valid/ → val/` porque Ultralytics espera esa carpeta.

### 2. Preparar la clase `no_yoga`

Un clasificador solo elige entre las clases que conoce: con 7 posturas de
yoga, **cualquier** imagen (un perro, una persona de pie) cae en alguna. Para
que el sistema pueda decir "esto no es yoga" hay que enseñarle esa opción.

```bash
python clasificador/preparar_negativos.py            # ~150 picsum + coco128
python clasificador/preparar_negativos.py --picsum 250
```

Descarga imágenes que **no** son yoga (coco128 + picsum) y las reparte en
`dataset/train|val|test/no_yoga/`. Para afinar el caso difícil (personas de
pie, que el modelo confunde con *mountain pose*), deja tus propias fotos en
`clasificador/negativos_extra/` antes de ejecutarlo.

### 3. Entrenar

```bash
python clasificador/entrenar.py
```

Genera el modelo en `clasificador/runs/classify/train/weights/best.pt`.
En CPU cada epoch tarda; si es muy lento, cambia `MODELO_BASE` en `config.py`
a `yolo11s-cls.pt` o `yolo11n-cls.pt`, o baja `EPOCHS`.

### 4. Predecir

```bash
python clasificador/predecir.py                                   # carpeta test/ del dataset
python clasificador/predecir.py "clasificador/dataset/test/warrior pose"
python clasificador/predecir.py ruta/a/imagen.jpg
```

Imprime la clase y la confianza de cada imagen y guarda las versiones
anotadas en `clasificador/predicciones/`.

### 5. Clasificar en vivo

```bash
python clasificador/clasificar_en_vivo.py
```

Muestra la webcam con la clase predicha superpuesta. **Esc** para salir.

## Usar el repo en otra máquina

El `.gitignore` deja fuera `venv/`, el dataset y `clasificador/runs/`. Para
**solo usar el detector** (inferencia) NO hace falta el dataset ni la API key:

```bash
git clone <repo> && cd <repo>
python -m venv venv && venv\Scripts\Activate.ps1
pip install -r requirements.txt -r clasificador/requirements.txt
python imagenes.py foto.jpg
```

Funciona sin reentrenar **si `clasificador/modelo.pt` está en el repo**. Ese
archivo (~20 MB) es la copia estable del modelo: `entrenar.py` la genera al
terminar. El `.gitignore` ignora todos los `*.pt` **menos** ese (regla
`!clasificador/modelo.pt`), así que solo hay que subirlo:

```bash
git add clasificador/modelo.pt
git commit -m "modelo entrenado (8 clases)"
```

`config.ruta_modelo()` usa `modelo.pt` si existe; si no, cae al
`runs/classify/train/weights/best.pt` recién entrenado.

Reentrenar en la otra máquina solo hace falta si cambias el dataset: ahí sí
necesitas `.env` con la API key + pasos 1-3.

## Qué se cambió del notebook

| Notebook (Colab) | Aquí |
|---|---|
| `!pip install roboflow ultralytics` | `requirements.txt` / `pip install` |
| `api_key="YOUR-API-KEY"` | `ROBOFLOW_API_KEY` en un `.env` (no versionado) |
| rutas `/content/...` | rutas relativas con `pathlib` |
| `version.download("folder")` | `download("folder", location=..., overwrite=True)` |
| `r.show()` | `r.plot()` + `cv2.imwrite` / `cv2.imshow` |
| salida implícita `runs/classify/train` | `project=`/`name=` fijos en `config.py` |
| celdas sueltas | `main()` + `if __name__ == "__main__"` |
