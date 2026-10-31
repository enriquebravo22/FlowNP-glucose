"""
05_check_metrics.py — Comprueba que metrics.py funciona antes de usarlo con modelos reales.

1. Prueba con datos sintéticos: si y ~ N(mu, sigma²) de verdad, la cobertura
   observada debe coincidir con la nominal, y el CRPS por muestras con el exacto.
2. Un mismo modelo (media y desviación del contexto) evaluado de dos formas
   en E1 tiene que dar lo mismo:
     a) como modelo gaussiano (add_gaussian), y
     b) como si fuera FlowNP: 2000 muestras de esa misma normal (add_samples).
   Una versión "sobreconfiada" (sigma / 2) debe salir por debajo de la diagonal.

No escribe nada en results/ (para no mezclarlo con los modelos de verdad).
Salida: checks/calibration_check.png
"""
import numpy as np
import torch
import config as C
from metrics import (Evaluator, gaussian_covered, gaussian_crps, sample_covered,
                     sample_crps, gaussian_loglik)
from sampling import task_batches
import plots

rng = np.random.default_rng(0)

# --- 1. Prueba sintética -------------------------------------------------------
mu, sigma = rng.normal(size=20000), rng.uniform(0.5, 2, size=20000)
y = mu + sigma * rng.normal(size=20000)
samples = mu + sigma * rng.normal(size=(2000, 20000))
print("1. Prueba sintética (bien calibrado)")
for p in [0.5, 0.9, 0.95]:
    a, s = gaussian_covered(y, mu, sigma, p).mean(), sample_covered(y, samples, p).mean()
    print(f"   cobertura {p:.0%}: analítica {a:.3f} | muestras {s:.3f}")
    assert abs(a - p) < 0.01 and abs(s - p) < 0.01
g, s = gaussian_crps(y, mu, sigma).mean(), sample_crps(y, samples).mean()
print(f"   CRPS: exacto {g:.4f} | muestras {s:.4f}\n")
assert abs(g - s) < 0.005


# --- 2. Gaussiano frente a muestras en E1 -----------------------------------------
def context_gaussian(batch, shrink=1.0):
    """mu = media del contexto, sigma = desviación del contexto (mínimo 0.05)."""
    mu = batch.yc.mean(dim=1, keepdim=True).expand_as(batch.yt)
    sigma = (batch.yc.std(dim=1, keepdim=True).clamp(min=0.05) * shrink).expand_as(batch.yt)
    return mu, sigma


TASK = "e1_test"
ev_g = Evaluator(TASK, "Gaussiano")
ev_s = Evaluator(TASK, "Muestras")
ev_o = Evaluator(TASK, "Sobreconfiado (σ/2)")
torch.manual_seed(0)
for batch in task_batches(TASK, batch_size=16):
    mu, sigma = context_gaussian(batch)
    ev_g.add_gaussian(batch, mu, sigma)
    s = mu + sigma * torch.randn(2000, *mu.shape)                              # [S, B, Nt, 1]
    ll = gaussian_loglik(batch.yt.numpy(), mu.numpy(), sigma.numpy())[..., 0]  # [B, Nt]
    ev_s.add_samples(batch, s, ll)
    ev_o.add_gaussian(batch, *context_gaussian(batch, shrink=0.5))

results = [ev.summary() for ev in (ev_g, ev_s, ev_o)]
keys = ["loglik_mgdl", "rmse_mgdl", "mae_mgdl", "crps_mgdl", "cov50", "cov90", "cov95"]
print(f"2. Mismo modelo evaluado de dos formas ({TASK})")
print(f"   {'':22s}" + "".join(f"{k:>12s}" for k in keys))
for r in results:
    print(f"   {r['model']:22s}" + "".join(f"{r['overall'][k]:12.3f}" for k in keys))
a, b = results[0]["overall"], results[1]["overall"]
assert abs(a["rmse_mgdl"] - b["rmse_mgdl"]) < 0.1 and abs(a["cov90"] - b["cov90"]) < 0.005

# Curva de calibración de la comprobación.
out = C.ROOT / "checks"
out.mkdir(exist_ok=True)
plots.FIGURES = out
plots.MODEL_ORDER = [r["model"] for r in results]
fake = {(r["model"], "e1_test"): [r] for r in results}
plots.fig_calibration(fake)
(out / "calibration_e1.png").replace(out / "calibration_check.png")
print(f"\nTodo correcto. Figura: {out / 'calibration_check.png'}")
