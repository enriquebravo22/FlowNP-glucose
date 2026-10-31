"""
config.py — Parámetros y rutas compartidas por todos los scripts.

Todo lo que se puede ajustar está aquí, para no tener "números mágicos"
repartidos por el código. Si cambiáis algo, volved a ejecutar los scripts
01 → 02 → 03 en orden.
"""
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"              # aquí van los CSV de GlucoBench (raw_data.zip)
PROCESSED_DIR = ROOT / "data" / "processed"  # salida del paso 01
SPLITS_DIR = ROOT / "data" / "splits"        # salida del paso 02
WINDOWS_DIR = ROOT / "data" / "windows"      # salida del paso 03

# Nombre del fichero CSV de GlucoBench -> nombre que usaremos nosotros.
# (En GlucoBench el dataset "Broll" se guarda como iglu.csv.)
DATASETS = {
    "iglu.csv": "broll",
    "colas.csv": "colas",
    "dubosson.csv": "dubosson",
    "hall.csv": "hall",
    "weinstock.csv": "weinstock",
}

# ---------------------------------------------------------------------------
# Paso 01: preprocesado
# ---------------------------------------------------------------------------
GL_MIN, GL_MAX = 40.0, 400.0   # rango de medida del sensor (mg/dL)
MAX_GAP_MIN = 60.0             # un hueco mayor que esto parte la serie en segmentos

# ---------------------------------------------------------------------------
# Paso 02: partición por paciente
# ---------------------------------------------------------------------------
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_VERSION = "v1"           # súbelo (v2, v3...) si cambiáis la partición

# ---------------------------------------------------------------------------
# Paso 03: ventanas y normalización
# ---------------------------------------------------------------------------
WINDOW_MIN = 6 * 60            # duración de cada ventana: 6 h (hasta 72 lecturas)
STRIDE_MIN = {                 # cada cuánto empieza una ventana nueva
    "train": 3 * 60,           # train: solapamiento del 50 % -> más ventanas
    "val": 6 * 60,             # val/test: sin solapamiento -> evaluación independiente
    "test": 6 * 60,
}
MIN_POINTS_PER_WINDOW = 36     # descartamos ventanas con menos del 50 % de lecturas

# ---------------------------------------------------------------------------
# Paso 04: muestreo de contexto y target
# ---------------------------------------------------------------------------
TASKS_DIR = ROOT / "data" / "tasks"   # salida del paso 04 (tareas fijas de evaluación)
NC_MIN, NC_MAX = 5, 50         # nº de puntos de contexto (entrenamiento y E1)
MIN_TARGETS = 5                # siempre dejamos al menos 5 puntos como target
TASK_SEED = 0                  # semilla de las tareas de evaluación

E2_CONTEXT_SIZES = [5, 10, 20, 50]   # 100 no cabe: una ventana de 6 h tiene como mucho 73 lecturas
E2_MIN_POINTS = 60                   # E2 usa solo ventanas con >= 60 lecturas (>= 10 targets con Nc = 50)

E3_GAP_MIN = [30, 60, 120, 180]      # duración de la caída simulada del sensor (min)
E3_HISTORY_MIN = 300                 # predicción: contexto = primeras 5 h de la ventana
E3_HORIZONS_MIN = [30, 60]           # predicción: target = los siguientes 30 o 60 min


# ---------------------------------------------------------------------------
# Función de lectura que usarán también las personas B y C
# ---------------------------------------------------------------------------
def load_windows(split):
    """Devuelve una lista de ventanas del split ('train', 'val' o 'test').

    Cada ventana es un diccionario con:
      x       -> tiempo normalizado en [-1, 1]       (array de tamaño N)
      y       -> glucosa normalizada (z-score)        (array de tamaño N)
      t_min   -> minutos desde el inicio de la ventana
      gl      -> glucosa en mg/dL (sin normalizar)
      censored-> True si la lectura estaba fuera de [40, 400] y se recortó
      patient, dataset -> de quién es la ventana
    N cambia de una ventana a otra (muestreo irregular).
    """
    # Cargamos cada array UNA vez (en un .npz, cada d["x"] vuelve a leer el fichero).
    d = dict(np.load(WINDOWS_DIR / f"{split}.npz"))
    # Las lecturas de todas las ventanas están concatenadas; 'offsets' marca
    # dónde empieza y acaba cada una: ventana i = posiciones offsets[i]:offsets[i+1]
    off = d["offsets"]
    windows = []
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        windows.append({
            "x": d["x"][a:b],
            "y": d["y"][a:b],
            "t_min": d["t_min"][a:b],
            "gl": d["gl"][a:b],
            "censored": d["censored"][a:b],
            "patient": str(d["patient"][i]),
            "dataset": str(d["dataset"][i]),
        })
    return windows
