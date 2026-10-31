"""
02_split.py — Partición train / val / test POR PACIENTE.

Entrada : data/processed/readings.csv
Salida  : data/splits/split_v1.json  -> {"train": [ids], "val": [ids], "test": [ids]}

Reglas:
  - Un paciente está en un único conjunto (evita fuga de información:
    el modelo nunca ve en test a alguien que ya vio en train).
  - Repartimos DENTRO de cada dataset (70/15/15), para que los cinco
    datasets aparezcan en los tres conjuntos.
  - Semilla fija -> la partición es siempre la misma.

Este JSON es el "contrato" del equipo: todos (A, B y C) deben usarlo.
"""
import json
import numpy as np
import pandas as pd
import config as C


def split_patients(patients, rng):
    """Baraja una lista de pacientes y la corta en train / val / test."""
    patients = sorted(patients)          # orden fijo antes de barajar
    rng.shuffle(patients)
    n = len(patients)

    # Al menos 1 paciente en val y en test, aunque el dataset sea pequeño
    # (Broll tiene 5 pacientes y Dubosson 8).
    n_test = max(1, round(n * C.SPLIT_FRACTIONS["test"]))
    n_val = max(1, round(n * C.SPLIT_FRACTIONS["val"]))
    n_train = n - n_val - n_test

    return {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:],
    }


def main():
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    readings = pd.read_csv(C.PROCESSED_DIR / "readings.csv", usecols=["dataset", "patient"])
    rng = np.random.default_rng(C.SPLIT_SEED)

    split = {"train": [], "val": [], "test": []}
    for dataset, group in readings.groupby("dataset"):
        parts = split_patients(group["patient"].unique().tolist(), rng)
        for name in split:
            split[name] += parts[name]
        print(f"{dataset:10s} train={len(parts['train']):3d}  val={len(parts['val']):3d}  test={len(parts['test']):3d}")

    # Comprobación: ningún paciente repetido entre conjuntos.
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "¡Paciente en dos conjuntos!"

    out = C.SPLITS_DIR / f"split_{C.SPLIT_VERSION}.json"
    with open(out, "w") as f:
        json.dump({"version": C.SPLIT_VERSION, "seed": C.SPLIT_SEED, **split}, f, indent=2)
    print(f"\nTOTAL      train={len(tr):3d}  val={len(va):3d}  test={len(te):3d}")
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
