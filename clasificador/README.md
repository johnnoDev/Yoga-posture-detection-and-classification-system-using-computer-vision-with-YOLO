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
pip install -r clasificador/requirements.txt   # añade 'roboflow'
```

## Pasos

| # | Script | Qué hace |
|---|--------|----------|
| 1 | `descargar_dataset.py` | Baja el dataset de Roboflow a `clasificador/dataset/` |
| 2 | `entrenar.py` | Entrena `yolo11m-cls.pt` con ese dataset (20 epochs, CPU) |
| 3 | `predecir.py` | Clasifica imágenes con el modelo entrenado y guarda el resultado |
| 4 | `clasificar_en_vivo.py` | Clasifica lo que ve la webcam en tiempo real (Esc para salir) |

`config.py` centraliza rutas, modelo base y parámetros de entrenamiento.

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

### 2. Entrenar

```bash
python clasificador/entrenar.py
```

Genera el modelo en `clasificador/runs/classify/train/weights/best.pt`.
En CPU cada epoch tarda; si es muy lento, cambia `MODELO_BASE` en `config.py`
a `yolo11s-cls.pt` o `yolo11n-cls.pt`, o baja `EPOCHS`.

### 3. Predecir

```bash
python clasificador/predecir.py                                   # carpeta test/ del dataset
python clasificador/predecir.py "clasificador/dataset/test/warrior pose"
python clasificador/predecir.py ruta/a/imagen.jpg
```

Imprime la clase y la confianza de cada imagen y guarda las versiones
anotadas en `clasificador/predicciones/`.

### 4. Clasificar en vivo

```bash
python clasificador/clasificar_en_vivo.py
```

Muestra la webcam con la clase predicha superpuesta. **Esc** para salir.

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
