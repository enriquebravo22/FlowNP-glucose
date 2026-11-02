"""
06_baselines.py — Dos baselines triviales de referencia, evaluados en TODOS los experimentos.

Sirven para dos cosas:
  1. Dar un "suelo" en las tablas: un NP que no mejore claramente a estos no aprende.
  2. Probar el flujo completo (Evaluator -> results/ -> plots.py) antes de que
     lleguen los modelos de B y C. Por eso cada uno usa una de las dos entradas:

  "Media contexto"  (vía gaussiana, como CNP/ANP/TNP/GP)
        mu = media del contexto, sigma = desviación del contexto.
  "Interp. lineal"  (vía muestras, como FlowNP)
        Interpolación lineal entre los puntos de contexto (fuera de ellos, repite el
        último valor: en predicción equivale a "la glucosa se queda como está").
        Incertidumbre que crece con la distancia al contexto más cercano:
        sigma = 0.1 + 2 · distancia (en unidades de x). Se evalúa con 200 muestras.

Se ejecuta con 3 semillas (solo afecta a las muestras) para probar la
agregación media ± desviación de plots.py.

Salida: results/<baseline>/seed<k>/<tarea>.json
"""
import numpy as np
import torch
import config as C
from metrics import Evaluator, gaussian_loglik
from sampling import task_batches

TASKS = (["e1_test"] + [f"e2_nc{k}" for k in C.E2_CONTEXT_SIZES]
         + [f"e3_gap{g}" for g in C.E3_GAP_MIN] + [f"e3_fc{h}" for h in C.E3_HORIZONS_MIN])
SEEDS = [0, 1, 2]
N_SAMPLES = 200


def context_mean(batch):
    mu = batch.yc.mean(dim=1, keepdim=True).expand_as(batch.yt)
    sigma = batch.yc.std(dim=1, keepdim=True).clamp(min=0.05).expand_as(batch.yt)
    return mu, sigma


def linear_interp(batch):
    """Interpolación lineal tarea a tarea con np.interp (necesita x ordenado)."""
    xc, yc, xt = batch.xc[..., 0].numpy(), batch.yc[..., 0].numpy(), batch.xt[..., 0].numpy()
    mu, sigma = np.zeros_like(xt), np.zeros_like(xt)
    for b in range(xt.shape[0]):
        o = np.argsort(xc[b])
        mu[b] = np.interp(xt[b], xc[b][o], yc[b][o])
        dist = np.abs(xt[b][:, None] - xc[b][None, :]).min(axis=1)
        sigma[b] = 0.1 + 2.0 * dist
    return mu[..., None], sigma[..., None]


def main():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for task in TASKS:
            ev_mean = Evaluator(task, "Media contexto", seed=seed)
            ev_interp = Evaluator(task, "Interp. lineal", seed=seed)
            for batch in task_batches(task, batch_size=32):
                ev_mean.add_gaussian(batch, *context_mean(batch))

                mu, sigma = linear_interp(batch)
                samples = mu + sigma * rng.normal(size=(N_SAMPLES, *mu.shape))
                ll = gaussian_loglik(batch.yt.numpy(), mu, sigma)[..., 0]
                ev_interp.add_samples(batch, samples, ll)
            r1, r2 = ev_mean.save(), ev_interp.save()
            print(f"seed {seed} {task:10s} | RMSE media {r1['overall']['rmse_mgdl']:5.1f} "
                  f"| RMSE interp {r2['overall']['rmse_mgdl']:5.1f} mg/dL")


if __name__ == "__main__":
    main()
