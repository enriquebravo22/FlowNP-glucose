"""
01_preprocess.py — Limpieza de las series CGM de GlucoBench.

Entrada : data/raw/*.csv              (los 5 CSV de raw_data.zip de GlucoBench)
Salida  : data/processed/readings.csv (una fila por lectura de glucosa)

Pasos:
  1. Leer cada dataset y quedarnos con las columnas id, time, gl.
  2. Dar a cada paciente un identificador único: "<dataset>_<id>".
  3. Quitar lecturas vacías y marcas de tiempo duplicadas.
  4. Recortar la glucosa al rango del sensor [40, 400] mg/dL y marcar
     las lecturas recortadas (columna 'censored').
  5. Partir cada serie en segmentos cuando hay un hueco > 60 min.
  6. Pasar el tiempo a minutos desde el inicio de cada segmento.
  7. Descartar los segmentos más cortos que una ventana (6 h).

IMPORTANTE: no interpolamos nada. Los huecos de menos de 60 min se quedan
como huecos: esa irregularidad es justo lo que queremos que modelen los NP.
"""
import pandas as pd
import config as C


def load_dataset(csv_name, dataset_name):
    """Paso 1 y 2: leer un CSV y dejarlo con columnas comunes."""
    df = pd.read_csv(C.RAW_DIR / csv_name, usecols=["id", "time", "gl"])

    # Los ids vienen en formatos distintos ("Subject 1", 1.0, "1636-69-001").
    # Los pasamos a texto, quitamos el ".0" de los que se leyeron como float,
    # y añadimos el nombre del dataset para que no se repitan entre datasets.
    ids = df["id"].astype(str).str.removesuffix(".0")
    df["patient"] = dataset_name + "_" + ids
    df["dataset"] = dataset_name
    df["time"] = pd.to_datetime(df["time"])
    return df[["dataset", "patient", "time", "gl"]]


def clean(df):
    """Paso 3 y 4: quitar vacíos/duplicados y tratar valores fuera de rango."""
    n0 = len(df)
    df = df.dropna(subset=["gl"])
    df = df.sort_values(["patient", "time"])
    df = df.drop_duplicates(subset=["patient", "time"], keep="first")
    print(f"  quitadas {n0 - len(df)} lecturas vacías o duplicadas")

    # Los sensores no miden fuera de ~[40, 400]. Weinstock, por ejemplo, guarda
    # "LOW" como 39 y "HIGH" como 401. Recortamos al rango y dejamos una marca
    # para saber que ese valor real era "≤ 40" o "≥ 400", no exactamente 40/400.
    df["censored"] = (df["gl"] < C.GL_MIN) | (df["gl"] > C.GL_MAX)
    df["gl"] = df["gl"].clip(C.GL_MIN, C.GL_MAX)
    print(f"  recortadas {df['censored'].sum()} lecturas fuera de [{C.GL_MIN:.0f}, {C.GL_MAX:.0f}]")
    return df


def make_segments(df):
    """Paso 5, 6 y 7: partir en segmentos y medir el tiempo en minutos."""
    # Minutos transcurridos desde la lectura anterior del mismo paciente.
    gap = df.groupby("patient")["time"].diff().dt.total_seconds() / 60

    # Empieza un segmento nuevo cuando el hueco es > 60 min (o es la 1.ª lectura).
    # cumsum convierte esas "marcas de inicio" en un número de segmento 1, 2, 3...
    new_segment = gap.isna() | (gap > C.MAX_GAP_MIN)
    df["segment"] = new_segment.groupby(df["patient"]).cumsum()

    # Tiempo en minutos desde el inicio de su segmento.
    seg_start = df.groupby(["patient", "segment"])["time"].transform("min")
    df["t_min"] = (df["time"] - seg_start).dt.total_seconds() / 60

    # Un segmento más corto que una ventana no da ninguna ventana completa.
    seg_len = df.groupby(["patient", "segment"])["t_min"].transform("max")
    n_seg_before = df.groupby(["patient", "segment"]).ngroups
    df = df[seg_len >= C.WINDOW_MIN]
    n_seg_after = df.groupby(["patient", "segment"]).ngroups
    print(f"  segmentos: {n_seg_before} -> {n_seg_after} tras quitar los de menos de {C.WINDOW_MIN // 60} h")
    return df


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for csv_name, dataset_name in C.DATASETS.items():
        print(f"[{dataset_name}]")
        df = load_dataset(csv_name, dataset_name)
        df = clean(df)
        df = make_segments(df)
        print(f"  -> {df['patient'].nunique()} pacientes, {len(df)} lecturas")
        all_dfs.append(df)

    readings = pd.concat(all_dfs, ignore_index=True)
    cols = ["dataset", "patient", "segment", "t_min", "gl", "censored"]
    readings[cols].to_csv(C.PROCESSED_DIR / "readings.csv", index=False)
    print(f"\nGuardado: {C.PROCESSED_DIR / 'readings.csv'}  ({len(readings)} lecturas, "
          f"{readings['patient'].nunique()} pacientes)")


if __name__ == "__main__":
    main()
