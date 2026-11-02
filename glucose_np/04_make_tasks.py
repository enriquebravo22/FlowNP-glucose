"""
04_make_tasks.py — Genera UNA VEZ las tareas fijas de evaluación.

Entrada : data/windows/val.npz, test.npz
Salida  : data/tasks/<experimento>.npz   y   data/tasks/summary.json

Una "tarea" = una ventana partida en contexto (lo que ve el modelo) y
target (lo que tiene que predecir). Guardarlas en disco garantiza que todos
los modelos se evalúan sobre exactamente los mismos puntos.

Experimentos:
  e1_val, e1_test   E1: Nc ~ U[5, 50] puntos al azar; target = el resto.
                    (e1_val sirve para early stopping / elegir hiperparámetros)
  e2_nc5 ... nc50   E2: Nc fijo. Contextos anidados (los 5 puntos de nc5 están
                    dentro de los 10 de nc10, etc.) y el MISMO target en todos.
  e3_gap30 ... 180  E3: se quita un bloque contiguo de 30/60/120/180 min
                    (caída del sensor). Target = lecturas del bloque.
  e3_fc30, e3_fc60  E3: predicción. Contexto = 4 h de historia (de 1 h a 5 h
                    de la ventana); target = lecturas de los siguientes 30/60 min.
"""
import json
import numpy as np
import config as C
from config import load_windows
from sampling import save_tasks


def take(w, ic, it):
    """Construye una tarea (xc, yc, xt, yt, censored_t) a partir de índices."""
    ic, it = np.sort(ic), np.sort(it)
    return (w["x"][ic], w["y"][ic], w["x"][it], w["y"][it], w["censored"][it])


def e1_tasks(windows, rng):
    """E1: igual que en entrenamiento, pero fijado con semilla."""
    out = []
    for i, w in enumerate(windows):
        n = len(w["x"])
        nc = rng.integers(C.NC_MIN, min(C.NC_MAX, n - C.MIN_TARGETS) + 1)
        perm = rng.permutation(n)
        out.append((i, take(w, perm[:nc], perm[nc:])))
    return out


def e2_tasks(windows, rng):
    """E2: una permutación por ventana. Los primeros k puntos son el contexto
    de tamaño k, y todos los puntos a partir del 50 son el target (común)."""
    out = {k: [] for k in C.E2_CONTEXT_SIZES}
    nc_max = max(C.E2_CONTEXT_SIZES)
    for i, w in enumerate(windows):
        if len(w["x"]) < C.E2_MIN_POINTS:
            continue                       # misma selección de ventanas para todos los k
        perm = rng.permutation(len(w["x"]))
        for k in C.E2_CONTEXT_SIZES:
            out[k].append((i, take(w, perm[:k], perm[nc_max:])))
    return out


def e3_gap_tasks(windows, gap_min, rng):
    """E3 huecos: un bloque contiguo [s, s + gap) desaparece.

    - Target: las lecturas dentro del bloque.
    - Contexto: las lecturas fuera del bloque. Si hay más de 50, nos quedamos
      con 50 al azar, para no salirnos del rango de Nc visto en entrenamiento.
    """
    out = []
    for i, w in enumerate(windows):
        t = w["t_min"]
        for _ in range(10):                          # hasta 10 intentos por ventana
            s = rng.uniform(0, C.WINDOW_MIN - gap_min)
            in_gap = (t >= s) & (t < s + gap_min)
            if in_gap.sum() >= 1:                    # puede caer sobre un hueco real
                break
        else:
            continue                                 # sin suerte: saltamos la ventana
        ic, it = np.flatnonzero(~in_gap), np.flatnonzero(in_gap)
        if len(ic) > C.NC_MAX:
            ic = rng.choice(ic, size=C.NC_MAX, replace=False)
        out.append((i, take(w, ic, it)))
    return out


def e3_forecast_tasks(windows, horizon_min):
    """E3 predicción: contexto = historia [1 h, 5 h); target = [5 h, 5 h + horizonte).

    Usamos 4 h de historia (como mucho 48 lecturas) para no pasar de los
    50 puntos de contexto vistos en entrenamiento. No hay azar.
    """
    out = []
    t0 = C.E3_HISTORY_MIN
    for i, w in enumerate(windows):
        t = w["t_min"]
        ic = np.flatnonzero((t >= t0 - 4 * 60) & (t < t0))
        it = np.flatnonzero((t >= t0) & (t < t0 + horizon_min))
        if len(ic) >= C.NC_MIN and len(it) >= 1:
            out.append((i, take(w, ic, it)))
    return out


def save(name, pairs, windows, summary):
    """Guarda las tareas de un experimento junto con quién es cada una."""
    idx = [i for i, _ in pairs]
    meta = {"window": idx,
            "dataset": [windows[i]["dataset"] for i in idx],
            "patient": [windows[i]["patient"] for i in idx]}
    save_tasks(C.TASKS_DIR / f"{name}.npz", [t for _, t in pairs], meta)
    nc = [len(t[0]) for _, t in pairs]
    nt = [len(t[2]) for _, t in pairs]
    summary[name] = {"n_tasks": len(pairs), "nc_mean": round(float(np.mean(nc)), 1),
                     "nt_mean": round(float(np.mean(nt)), 1)}
    print(f"{name:12s} {len(pairs):5d} tareas | Nc medio {np.mean(nc):5.1f} | Nt medio {np.mean(nt):5.1f}")


def main():
    C.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    val, test = load_windows("val"), load_windows("test")
    summary = {}

    # Cada experimento tiene su propio generador con semilla fija, así cambiar
    # uno no altera los demás.
    save("e1_val", e1_tasks(val, np.random.default_rng(C.TASK_SEED)), val, summary)
    save("e1_test", e1_tasks(test, np.random.default_rng(C.TASK_SEED + 1)), test, summary)

    for k, pairs in e2_tasks(test, np.random.default_rng(C.TASK_SEED + 2)).items():
        save(f"e2_nc{k}", pairs, test, summary)

    for j, g in enumerate(C.E3_GAP_MIN):
        save(f"e3_gap{g}", e3_gap_tasks(test, g, np.random.default_rng(C.TASK_SEED + 10 + j)),
             test, summary)

    for h in C.E3_HORIZONS_MIN:
        save(f"e3_fc{h}", e3_forecast_tasks(test, h), test, summary)

    with open(C.TASKS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
