"""
sampling.py — Batches de contexto/target en el formato del repo de TNP.

Cada batch es un diccionario (al que también se accede con punto) con:
    batch.x,  batch.y    todos los puntos  [B, Nc + Nt, 1]   (primero contexto, luego target)
    batch.xc, batch.yc   contexto          [B, Nc, 1]
    batch.xt, batch.yt   target            [B, Nt, 1]
    batch.task_id        índice de cada tarea [B]  (solo en evaluación; lo usa metrics.py)

x = tiempo normalizado en [-1, 1];  y = glucosa normalizada (z-score).

Dos usos:
  1. Entrenamiento:  sampler = TrainSampler(load_windows("train"))
                     batch = sampler.sample(batch_size=16, device="cuda")
  2. Evaluación:     for batch in task_batches("e1_test", batch_size=16): ...
     (las tareas se generan una sola vez con 04_make_tasks.py)
"""
import numpy as np
import torch
import config as C


class Batch(dict):
    """Diccionario al que se puede acceder como batch.xc (igual que AttrDict en TNP).

    Al ser un dict, el bucle típico de TNP sigue funcionando:
        for k, v in batch.items(): batch[k] = v.cuda()
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def make_batch(tasks, device="cpu"):
    """Convierte una lista de tareas (xc, yc, xt, yt) en un Batch de tensores.

    Todas las tareas de la lista deben tener el mismo Nc y el mismo Nt.
    """
    def stack(i):
        # [B, N] -> tensor [B, N, 1]
        arr = np.stack([t[i] for t in tasks]).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(-1).to(device)

    b = Batch(xc=stack(0), yc=stack(1), xt=stack(2), yt=stack(3))
    b.x = torch.cat([b.xc, b.xt], dim=1)
    b.y = torch.cat([b.yc, b.yt], dim=1)
    return b


# ---------------------------------------------------------------------------
# 1. Entrenamiento: contexto y target aleatorios en cada batch
# ---------------------------------------------------------------------------
class TrainSampler:
    """Genera batches aleatorios a partir de las ventanas de train.

    En cada batch:
      1. Elegimos B ventanas al azar.
      2. Elegimos un nº de contexto Nc ~ Uniforme[5, 50], el mismo para todo el batch.
      3. En cada ventana, Nc puntos al azar son el contexto y el RESTO es el target.
         Como las ventanas tienen longitudes distintas, todas usan Nt = (la ventana
         más corta del batch) - Nc, para que los tensores tengan la misma forma.
    """

    def __init__(self, windows, seed=None):
        self.windows = windows
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size=16, device="cpu"):
        # Paso 1: ventanas del batch.
        idx = self.rng.choice(len(self.windows), size=batch_size, replace=False)
        wins = [self.windows[i] for i in idx]

        # Paso 2: Nc común. Dejamos siempre al menos MIN_TARGETS puntos para target.
        n_min = min(len(w["x"]) for w in wins)
        nc = self.rng.integers(C.NC_MIN, min(C.NC_MAX, n_min - C.MIN_TARGETS) + 1)
        nt = n_min - nc

        # Paso 3: permutación aleatoria de los puntos; los primeros nc son contexto
        # y los nt siguientes, target. Ordenamos por tiempo dentro de cada parte
        # (a los modelos les da igual el orden; a las figuras, no).
        tasks = []
        for w in wins:
            perm = self.rng.permutation(len(w["x"]))
            ic, it = np.sort(perm[:nc]), np.sort(perm[nc:nc + nt])
            tasks.append((w["x"][ic], w["y"][ic], w["x"][it], w["y"][it]))
        return make_batch(tasks, device)


# ---------------------------------------------------------------------------
# 2. Evaluación: tareas fijas guardadas en disco
# ---------------------------------------------------------------------------
def save_tasks(path, tasks, meta):
    """Guarda una lista de tareas de longitud variable en un .npz.

    tasks: lista de (xc, yc, xt, yt, censored_t)
    meta:  dict con listas de la misma longitud (dataset, patient, window...)
    Igual que con las ventanas, concatenamos todo y guardamos 'offsets'.
    """
    nc = np.array([len(t[0]) for t in tasks])
    nt = np.array([len(t[2]) for t in tasks])
    np.savez_compressed(
        path,
        xc=np.concatenate([t[0] for t in tasks]), yc=np.concatenate([t[1] for t in tasks]),
        xt=np.concatenate([t[2] for t in tasks]), yt=np.concatenate([t[3] for t in tasks]),
        censored_t=np.concatenate([t[4] for t in tasks]),
        off_c=np.concatenate([[0], np.cumsum(nc)]), off_t=np.concatenate([[0], np.cumsum(nt)]),
        **{k: np.array(v) for k, v in meta.items()},
    )


def load_tasks(name):
    """Lee data/tasks/<name>.npz y devuelve (lista de tareas, metadatos).

    Cada tarea es un dict con xc, yc, xt, yt, censored_t (arrays 1D).
    """
    d = dict(np.load(C.TASKS_DIR / f"{name}.npz"))
    oc, ot = d["off_c"], d["off_t"]
    tasks = []
    for i in range(len(oc) - 1):
        tasks.append({
            "xc": d["xc"][oc[i]:oc[i + 1]], "yc": d["yc"][oc[i]:oc[i + 1]],
            "xt": d["xt"][ot[i]:ot[i + 1]], "yt": d["yt"][ot[i]:ot[i + 1]],
            "censored_t": d["censored_t"][ot[i]:ot[i + 1]],
        })
    meta = {k: d[k] for k in ["dataset", "patient", "window"]}
    return tasks, meta


def task_batches(name, batch_size=16, device="cpu"):
    """Recorre TODAS las tareas de un experimento, en batches.

    Solo se pueden juntar en un batch tareas con el mismo (Nc, Nt), así que
    agrupamos por esa pareja y partimos cada grupo en trozos de batch_size.
    Cada batch lleva batch.task_id para que metrics.py sepa de qué tarea es.
    """
    tasks, _ = load_tasks(name)
    groups = {}
    for i, t in enumerate(tasks):
        groups.setdefault((len(t["xc"]), len(t["xt"])), []).append(i)

    for ids in groups.values():
        for start in range(0, len(ids), batch_size):
            chunk = ids[start:start + batch_size]
            b = make_batch([(tasks[i]["xc"], tasks[i]["yc"], tasks[i]["xt"], tasks[i]["yt"])
                            for i in chunk], device)
            b.task_id = torch.tensor(chunk, device=device)
            yield b
