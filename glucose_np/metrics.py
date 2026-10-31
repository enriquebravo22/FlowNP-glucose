"""
metrics.py — Métricas comunes para todos los modelos.

Uso (el mismo bucle para cualquier modelo):

    from metrics import Evaluator
    from sampling import task_batches

    ev = Evaluator(task="e1_test", model="CNP", seed=0)   # seed = semilla del entrenamiento
    for batch in task_batches("e1_test", batch_size=16):
        # Modelos gaussianos (CNP, ANP, TNP, GP):
        mu, sigma = model(batch)                 # [B, Nt, 1] cada uno
        ev.add_gaussian(batch, mu, sigma)
        # FlowNP:
        #   samples = ...                        # [S, B, Nt, 1]
        #   loglik  = ...                        # [B, Nt] por punto, o [B] conjunta (ver abajo)
        #   ev.add_samples(batch, samples, loglik)
    ev.save()             # -> results/CNP/seed0/e1_test.json

Para la tabla con media ± desviación, entrenad cada modelo con varias semillas
(p. ej. 0, 1, 2) y evaluad cada una con su seed. plots.py las agrega.

El Evaluator guarda además, sin que tengáis que hacer nada, las predicciones de
unas pocas tareas de ejemplo (una por dataset) para la figura de trayectorias.

Todo lo que entra (y, mu, sigma, samples, loglik) está en ESPACIO NORMALIZADO,
que es en el que trabajan los modelos. Aquí se pasa a mg/dL donde toca.

Métricas (todas por punto de target, promediando todos los puntos del experimento):
  loglik_norm   log-verosimilitud en espacio normalizado          (más alto = mejor)
  loglik_mgdl   la misma, en mg/dL: loglik_norm - log(std_train)  (más alto = mejor)
  rmse_mgdl     raíz del error cuadrático medio, en mg/dL         (más bajo = mejor)
  mae_mgdl      error absoluto medio, en mg/dL                    (más bajo = mejor)
  crps_mgdl     CRPS, en mg/dL                                    (más bajo = mejor)
  cov50/90/95   fracción de targets dentro del intervalo del 50/90/95 %
                (bien calibrado = 0.50 / 0.90 / 0.95)
  calibration   curva de cobertura nominal (5 %...95 %) frente a observada
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
import config as C

LEVELS_TABLE = [0.50, 0.90, 0.95]                  # coberturas de la tabla
LEVELS_CURVE = np.round(np.arange(0.05, 1.0, 0.05), 2)   # puntos de la curva de calibración
RESULTS_DIR = C.ROOT / "results"


def to_numpy(a):
    """Acepta tensores de torch (también en GPU) o arrays de numpy."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fórmulas para modelos gaussianos: predicción N(mu, sigma²) en cada punto
# ---------------------------------------------------------------------------
def gaussian_loglik(y, mu, sigma):
    """log N(y; mu, sigma²), punto a punto."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2


def gaussian_covered(y, mu, sigma, level):
    """¿Cae y dentro del intervalo central mu ± z·sigma de nivel 'level'?
    Para el 95 %, z = 1.96 (el cuantil 0.975 de la normal estándar)."""
    z = norm.ppf(0.5 + level / 2)
    return np.abs(y - mu) <= z * sigma


def gaussian_crps(y, mu, sigma):
    """CRPS de una normal, fórmula cerrada (Gneiting y Raftery, 2007):
    CRPS = sigma · [ z·(2Φ(z) − 1) + 2φ(z) − 1/√π ],  con z = (y − mu) / sigma."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# Fórmulas para FlowNP: S muestras en cada punto (eje 0)
# ---------------------------------------------------------------------------
def sample_covered(y, samples, level):
    """Intervalo central con cuantiles empíricos: para el 90 %, del 5 % al 95 %."""
    lo = np.quantile(samples, 0.5 - level / 2, axis=0)
    hi = np.quantile(samples, 0.5 + level / 2, axis=0)
    return (y >= lo) & (y <= hi)


def sample_covered_all(y, samples, levels):
    """Lo mismo que sample_covered para varios niveles a la vez.
    Pedir todos los cuantiles en una sola llamada evita reordenar las
    muestras una vez por nivel (mucho más rápido)."""
    levels = np.asarray(levels)
    q = np.quantile(samples, np.concatenate([0.5 - levels / 2, 0.5 + levels / 2]), axis=0)
    lo, hi = q[:len(levels)], q[len(levels):]
    return (y >= lo) & (y <= hi)                     # [nº niveles, ...]


def sample_crps(y, samples):
    """Estimador por muestras:  CRPS = E|X − y| − ½·E|X − X'|.

    El segundo término compararía todas las parejas de muestras (S² operaciones).
    Con las muestras ordenadas x(1) ≤ ... ≤ x(S) se puede calcular en orden S·log S:
        E|X − X'| = 2/S² · Σ_i (2i − S − 1) · x(i)
    """
    S = samples.shape[0]
    term1 = np.abs(samples - y).mean(axis=0)
    xs = np.sort(samples, axis=0)
    w = (2 * np.arange(1, S + 1) - S - 1).reshape(-1, *([1] * y.ndim))
    term2 = 2.0 / S**2 * (w * xs).sum(axis=0)
    return term1 - 0.5 * term2


# ---------------------------------------------------------------------------
# Evaluator: acumula resultados batch a batch y calcula el resumen
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, task, model, seed=0):
        self.task, self.model, self.seed = task, model, seed
        with open(C.WINDOWS_DIR / "norm_stats.json") as f:
            st = json.load(f)
        self.mean, self.std = st["gl_mean"], st["gl_std"]

        # Metadatos de cada tarea (para métricas por dataset y para excluir censurados).
        d = np.load(C.TASKS_DIR / f"{task}.npz")
        self.task_dataset = d["dataset"]
        off_t, cens = d["off_t"], d["censored_t"]
        self.task_censored = [cens[off_t[i]:off_t[i + 1]] for i in range(len(off_t) - 1)]

        # Una entrada por punto de target; se rellenan en add_gaussian / add_samples.
        self.cols = {k: [] for k in
                     ["y", "pred_mean", "pred_median", "loglik", "crps", "covered",
                      "dataset", "censored"]}
        self.seen = set()

        # Tareas de ejemplo: la primera de cada dataset (siempre las mismas para
        # todos los modelos). Guardamos sus predicciones para dibujarlas.
        self.example_ids = {int(np.flatnonzero(self.task_dataset == name)[0])
                            for name in sorted(set(self.task_dataset))}
        self.examples = {}

    # -- comprobaciones y metadatos comunes ----------------------------------
    def _meta(self, batch, nt):
        ids = to_numpy(batch.task_id).astype(int)
        assert not (set(ids) & self.seen), "Hay tareas evaluadas dos veces"
        self.seen.update(ids)
        self.cols["dataset"].append(np.repeat(self.task_dataset[ids], nt))
        cens = np.stack([self.task_censored[i] for i in ids])        # [B, Nt]
        self.cols["censored"].append(cens.ravel())

    def _save_examples(self, batch, center, q, samples=None):
        """Guarda contexto, target y predicción (en mg/dL) de las tareas de ejemplo.

        center: [B, Nt] predicción puntual; q: dict nivel -> (lo, hi), cada uno [B, Nt]
        samples: [S, B, Nt] (solo FlowNP); guardamos 20 para dibujarlas.
        """
        mg = lambda a: (np.asarray(a) * self.std + self.mean).tolist()
        hours = lambda x: ((np.asarray(x) + 1) / 2 * C.WINDOW_MIN / 60).tolist()
        ids = to_numpy(batch.task_id).astype(int)
        for b, i in enumerate(ids):
            if i not in self.example_ids:
                continue
            ex = {"dataset": str(self.task_dataset[i]),
                  "t_context_h": hours(to_numpy(batch.xc)[b, :, 0]),
                  "gl_context": mg(to_numpy(batch.yc)[b, :, 0]),
                  "t_target_h": hours(to_numpy(batch.xt)[b, :, 0]),
                  "gl_target": mg(to_numpy(batch.yt)[b, :, 0]),
                  "pred": mg(center[b])}
            for p, (lo, hi) in q.items():
                ex[f"lo{int(p * 100)}"], ex[f"hi{int(p * 100)}"] = mg(lo[b]), mg(hi[b])
            if samples is not None:
                ex["samples"] = [mg(s) for s in samples[:20, b]]
            self.examples[str(i)] = ex

    def _store(self, y, pred_mean, pred_median, loglik, crps, covered):
        for k, v in [("y", y), ("pred_mean", pred_mean), ("pred_median", pred_median),
                     ("loglik", loglik), ("crps", crps)]:
            self.cols[k].append(v.ravel())
        # covered: [nº niveles de la curva, nº puntos]
        self.cols["covered"].append(covered.reshape(len(LEVELS_CURVE), -1))

    # -- entrada de modelos gaussianos ---------------------------------------
    def add_gaussian(self, batch, mu, sigma):
        """mu, sigma: [B, Nt, 1] (o [B, Nt]) en espacio normalizado."""
        y = to_numpy(batch.yt)[..., 0]                       # [B, Nt]
        mu = to_numpy(mu).reshape(y.shape)
        sigma = to_numpy(sigma).reshape(y.shape)
        self._meta(batch, y.shape[1])
        covered = np.stack([gaussian_covered(y, mu, sigma, p) for p in LEVELS_CURVE])
        q = {p: (mu - norm.ppf(0.5 + p / 2) * sigma, mu + norm.ppf(0.5 + p / 2) * sigma)
             for p in [0.5, 0.9]}
        self._save_examples(batch, mu, q)
        # En una normal, media = mediana = mu.
        self._store(y, mu, mu, gaussian_loglik(y, mu, sigma), gaussian_crps(y, mu, sigma), covered)

    # -- entrada de FlowNP ----------------------------------------------------
    def add_samples(self, batch, samples, loglik=None):
        """samples: [S, B, Nt, 1] en espacio normalizado.

        loglik (en espacio normalizado), dos formatos aceptados:
          [B, Nt] o [B, Nt, 1]: log-verosimilitud de cada punto de target.
          [B]: log p(yt | contexto) CONJUNTA de todo el target de cada tarea.
               Se reparte entre los Nt puntos (se divide entre Nt), que es la
               "log-verosimilitud por punto de target" que comparamos.
          None: aún no disponible; esas métricas saldrán como NaN.
        """
        y = to_numpy(batch.yt)[..., 0]                       # [B, Nt]
        s = to_numpy(samples)
        s = s.reshape(s.shape[0], *y.shape)                  # [S, B, Nt]
        self._meta(batch, y.shape[1])

        if loglik is None:
            ll = np.full(y.shape, np.nan)
        else:
            ll = to_numpy(loglik)
            ll = np.repeat((ll / y.shape[1])[:, None], y.shape[1], axis=1) if ll.ndim == 1 \
                else ll.reshape(y.shape)

        covered = sample_covered_all(y, s, LEVELS_CURVE)
        q = {p: (np.quantile(s, 0.5 - p / 2, axis=0), np.quantile(s, 0.5 + p / 2, axis=0))
             for p in [0.5, 0.9]}
        self._save_examples(batch, s.mean(axis=0), q, samples=s)
        # Media para el RMSE (minimiza el error cuadrático) y mediana para el
        # MAE (minimiza el error absoluto).
        self._store(y, s.mean(axis=0), np.median(s, axis=0), ll, sample_crps(y, s), covered)

    # -- resumen ---------------------------------------------------------------
    def _metrics(self, m):
        """Calcula las métricas sobre los puntos seleccionados por la máscara m."""
        c = {k: (np.concatenate(v) if k != "covered" else np.concatenate(v, axis=1))
             for k, v in self.cols.items()}
        y_mg = c["y"][m] * self.std + self.mean                        # y en mg/dL
        mean_mg = c["pred_mean"][m] * self.std + self.mean
        median_mg = c["pred_median"][m] * self.std + self.mean
        ll = float(np.mean(c["loglik"][m]))
        cov = c["covered"][:, m].mean(axis=1)
        out = {
            "n_points": int(m.sum()),
            "loglik_norm": ll,
            # Cambio de variable y_mg = y·std + mean  ->  log p(y_mg) = log p(y) − log(std)
            "loglik_mgdl": ll - np.log(self.std),
            "rmse_mgdl": float(np.sqrt(np.mean((mean_mg - y_mg) ** 2))),
            "mae_mgdl": float(np.mean(np.abs(median_mg - y_mg))),
            # El CRPS tiene las unidades de y: basta con multiplicar por std.
            "crps_mgdl": float(np.mean(c["crps"][m]) * self.std),
        }
        for p in LEVELS_TABLE:
            out[f"cov{int(p * 100)}"] = float(cov[list(LEVELS_CURVE).index(p)])
        out["calibration"] = {"nominal": LEVELS_CURVE.tolist(), "observed": cov.tolist()}
        return out

    def summary(self, exclude_censored=False):
        """Métricas globales y por dataset.

        exclude_censored=True quita los targets que estaban fuera de [40, 400] mg/dL
        (su valor real era ≤ 40 o ≥ 400, no exactamente ese número).
        """
        n_tasks = len(self.task_dataset)
        if len(self.seen) != n_tasks:
            print(f"AVISO: evaluadas {len(self.seen)} de {n_tasks} tareas de {self.task}")
        ds = np.concatenate(self.cols["dataset"])
        keep = ~np.concatenate(self.cols["censored"]) if exclude_censored else np.ones(len(ds), bool)

        res = {"model": self.model, "task": self.task, "seed": self.seed,
               "exclude_censored": exclude_censored,
               "n_tasks": len(self.seen), "overall": self._metrics(keep), "by_dataset": {}}
        for name in sorted(set(ds)):
            res["by_dataset"][name] = self._metrics(keep & (ds == name))
        res["examples"] = self.examples
        return res

    def save(self, exclude_censored=False):
        """Guarda el resumen en results/<modelo>/seed<k>/<tarea>.json y lo devuelve."""
        res = self.summary(exclude_censored)
        out = RESULTS_DIR / self.model / f"seed{self.seed}" / f"{self.task}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        return res


# ---------------------------------------------------------------------------
# E4: tiempo de inferencia (lo rellena la persona B)
# ---------------------------------------------------------------------------
def save_timing(model, rows):
    """Guarda los resultados de E4 en results/<modelo>/e4_timing.json.

    rows: lista de diccionarios, uno por nº de pasos del solver, p. ej.
        {"steps": 10, "time_per_task_s": 0.012, "crps_mgdl": 15.3, "loglik_mgdl": -4.5}
    time_per_task_s = tiempo medio de muestreo por tarea de e1_test (mismo
    hardware y batch_size para todos los pasos). crps_mgdl y loglik_mgdl salen
    del Evaluator ejecutado con ese nº de pasos (pueden faltar).
    """
    out = RESULTS_DIR / model / "e4_timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": model, "rows": sorted(rows, key=lambda r: r["steps"])}, f, indent=2)
