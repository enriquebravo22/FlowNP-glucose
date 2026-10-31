"""
03_windows.py — Cortar los segmentos en ventanas de 6 h y normalizarlas.

Entrada : data/processed/readings.csv, data/splits/split_v1.json
Salida  : data/windows/train.npz, val.npz, test.npz
          data/windows/norm_stats.json   (media y desviación de train)

Cada ventana es una "función" para el Neural Process: un conjunto de pares
(x, y) con x = instante dentro de la ventana e y = glucosa.

Pasos:
  1. Calcular media y desviación de la glucosa SOLO con pacientes de train.
  2. Recorrer cada segmento con una ventana de 6 h que avanza STRIDE minutos.
  3. Quedarse con las lecturas que caen dentro; descartar ventanas con
     menos de 36 lecturas (menos del 50 % de las 72 posibles).
  4. Normalizar: tiempo -> [-1, 1], glucosa -> z-score.
  5. Guardar todas las ventanas de cada split en un .npz.

Para leerlas: from config import load_windows; ventanas = load_windows("train")
Para volver a mg/dL:  gl = y * std + mean   (valores en norm_stats.json)
"""
import json
import numpy as np
import pandas as pd
import config as C


def windows_from_segment(t, gl, censored, stride):
    """Paso 2 y 3: corta un segmento en ventanas.

    t, gl, censored son arrays del segmento, ordenados por tiempo.
    Devuelve una lista de (t_relativo, gl, censored), una por ventana.
    """
    result = []
    start = 0.0
    while start + C.WINDOW_MIN <= t[-1]:
        # Lecturas con start <= t < start + 6 h
        inside = (t >= start) & (t < start + C.WINDOW_MIN)
        if inside.sum() >= C.MIN_POINTS_PER_WINDOW:
            # Tiempo relativo al INICIO DE LA VENTANA (no a la primera lectura),
            # así un hueco al principio de la ventana se ve como hueco.
            result.append((t[inside] - start, gl[inside], censored[inside]))
        start += stride
    return result


def build_split(readings, patients, stride, mean, std):
    """Construye y normaliza todas las ventanas de un conjunto de pacientes."""
    data = readings[readings["patient"].isin(patients)]
    x_all, y_all, t_all, gl_all, cens_all = [], [], [], [], []
    patient_ids, dataset_ids = [], []

    for (dataset, patient, _), seg in data.groupby(["dataset", "patient", "segment"]):
        seg = seg.sort_values("t_min")
        for t, gl, cens in windows_from_segment(
            seg["t_min"].to_numpy(), seg["gl"].to_numpy(), seg["censored"].to_numpy(), stride
        ):
            # Paso 4: normalización.
            # Tiempo: [0, 360) min -> [-1, 1)
            x = 2.0 * t / C.WINDOW_MIN - 1.0
            # Glucosa: z-score con media y desviación de TRAIN.
            y = (gl - mean) / std

            x_all.append(x); y_all.append(y)
            t_all.append(t); gl_all.append(gl); cens_all.append(cens)
            patient_ids.append(patient); dataset_ids.append(dataset)

    # Paso 5: concatenamos todo y guardamos dónde empieza cada ventana (offsets).
    lengths = [len(x) for x in x_all]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    arrays = {
        "x": np.concatenate(x_all).astype(np.float32),
        "y": np.concatenate(y_all).astype(np.float32),
        "t_min": np.concatenate(t_all).astype(np.float32),
        "gl": np.concatenate(gl_all).astype(np.float32),
        "censored": np.concatenate(cens_all),
        "offsets": offsets,
        "patient": np.array(patient_ids),
        "dataset": np.array(dataset_ids),
    }
    return arrays, lengths


def main():
    C.WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    readings = pd.read_csv(C.PROCESSED_DIR / "readings.csv")
    with open(C.SPLITS_DIR / f"split_{C.SPLIT_VERSION}.json") as f:
        split = json.load(f)

    # Paso 1: estadísticos de normalización SOLO con train
    # (usar val/test aquí sería fuga de información).
    train_gl = readings.loc[readings["patient"].isin(split["train"]), "gl"]
    mean, std = float(train_gl.mean()), float(train_gl.std())
    stats = {"gl_mean": mean, "gl_std": std, "window_min": C.WINDOW_MIN,
             "split_version": C.SPLIT_VERSION}
    with open(C.WINDOWS_DIR / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Normalización (train): media = {mean:.1f} mg/dL, desviación = {std:.1f} mg/dL\n")

    for name in ["train", "val", "test"]:
        arrays, lengths = build_split(readings, split[name], C.STRIDE_MIN[name], mean, std)
        np.savez_compressed(C.WINDOWS_DIR / f"{name}.npz", **arrays)
        print(f"{name:5s}: {len(lengths):6d} ventanas | lecturas por ventana: "
              f"mín {min(lengths)}, mediana {int(np.median(lengths))}, máx {max(lengths)}")


if __name__ == "__main__":
    main()
