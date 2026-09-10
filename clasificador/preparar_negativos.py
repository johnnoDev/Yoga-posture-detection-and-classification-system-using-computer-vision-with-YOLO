"""
Prepara la clase "no_yoga": junta imagenes que NO son posturas de yoga y las
reparte en dataset/train|val|test/no_yoga/ para poder reentrenar el
clasificador con una clase extra que signifique "esto no es yoga".

Fuentes de negativos:
  1. coco128  -> 128 fotos reales variadas (personas de pie/sentadas,
     calle, deporte, cocina, animales, objetos)  [se descarga sola]
  2. picsum   -> N fotos genericas aleatorias (paisajes, objetos, comida...)
  3. clasificador/negativos_extra/  -> tus propias fotos (opcional pero muy
     recomendable: personas de pie, caminando, sentadas, selfies, etc.)

Uso:
    python clasificador/preparar_negativos.py
    python clasificador/preparar_negativos.py --picsum 200

Despues:
    python clasificador/entrenar.py        # reentrena con la clase no_yoga
"""
import argparse
import io
import random
import shutil
import sys
import zipfile

import requests
from PIL import Image

from config import DATASET_DIR, BASE_DIR

NOMBRE_CLASE = "no_yoga"

# Carpeta donde puedes dejar tus propias fotos "no yoga"
NEGATIVOS_EXTRA = BASE_DIR / "negativos_extra"

# Reparto entre train / val / test
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}

# Tope de imagenes en train (para no desbalancear demasiado frente a las
# ~98 imagenes que tiene cada clase de yoga)
MAX_TRAIN = 140

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def descargar_coco128(destino):
    """Descarga coco128 y guarda sus 128 imagenes .jpg en `destino`."""
    print("Descargando coco128 (~7 MB)...")
    r = requests.get(COCO128_URL, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    nombres = [n for n in z.namelist() if n.lower().endswith((".jpg", ".png"))]
    for n in nombres:
        data = z.read(n)
        (destino / f"coco_{n.split('/')[-1]}").write_bytes(data)
    print(f"  {len(nombres)} imagenes de coco128")
    return len(nombres)


def descargar_picsum(destino, cuantas):
    """Descarga `cuantas` fotos genericas aleatorias de picsum.photos."""
    print(f"Descargando {cuantas} imagenes de picsum...")
    ok = 0
    for i in range(cuantas):
        try:
            r = requests.get(f"https://picsum.photos/seed/negyoga{i}/256/256", timeout=30)
            r.raise_for_status()
            # Validamos que sea una imagen y la normalizamos a JPG
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(destino / f"picsum_{i:04d}.jpg", "JPEG", quality=90)
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{cuantas}")
        except Exception as e:
            print(f"  (salto {i}: {e})")
    print(f"  {ok} imagenes de picsum")
    return ok


def copiar_extra(destino):
    """Copia las fotos que el usuario haya dejado en negativos_extra/."""
    if not NEGATIVOS_EXTRA.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    n = 0
    for p in NEGATIVOS_EXTRA.iterdir():
        if p.suffix.lower() in exts:
            shutil.copy(p, destino / f"extra_{p.name}")
            n += 1
    if n:
        print(f"  {n} imagenes tuyas de negativos_extra/")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picsum", type=int, default=150, help="cuantas fotos genericas bajar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not DATASET_DIR.is_dir():
        print(f"No existe el dataset en {DATASET_DIR}.")
        print("Ejecuta primero:  python clasificador/descargar_dataset.py")
        return

    # 1. Reunimos todos los negativos en una carpeta temporal
    staging = BASE_DIR / "_negativos_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        descargar_coco128(staging)
    except Exception as e:
        print(f"Fallo la descarga de coco128 ({e}); sigo sin ella.")
    if args.picsum > 0:
        descargar_picsum(staging, args.picsum)
    copiar_extra(staging)

    imagenes = sorted(staging.iterdir())
    if len(imagenes) < 30:
        print(f"Solo se consiguieron {len(imagenes)} negativos; muy pocos. Aborto.")
        shutil.rmtree(staging)
        return

    # 2. Barajamos y repartimos en train / val / test
    random.seed(args.seed)
    random.shuffle(imagenes)

    n = len(imagenes)
    n_train = min(int(n * SPLIT["train"]), MAX_TRAIN)
    n_val = int(n * SPLIT["val"])
    reparto = {
        "train": imagenes[:n_train],
        "val": imagenes[n_train:n_train + n_val],
        "test": imagenes[n_train + n_val:],
    }

    # 3. Copiamos a dataset/<split>/no_yoga/ (limpiando lo anterior)
    for split, archivos in reparto.items():
        dst = DATASET_DIR / split / NOMBRE_CLASE
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for src in archivos:
            shutil.copy(src, dst / src.name)
        print(f"{split}/no_yoga/: {len(archivos)} imagenes")

    shutil.rmtree(staging)
    print("\nListo. Ahora reentrena:  python clasificador/entrenar.py")


if __name__ == "__main__":
    main()
