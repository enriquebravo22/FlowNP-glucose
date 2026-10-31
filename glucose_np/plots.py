"""
plots.py — Genera TODAS las tablas y figuras a partir de results/.

Entrada : results/<modelo>/seed<k>/<tarea>.json   (los escribe Evaluator.save())
          results/<modelo>/e4_timing.json          (lo escribe save_timing(), persona B)
Salida  : tables/*.csv y *.md     figures/*.png

Se puede ejecutar en cualquier momento: dibuja los modelos y experimentos que
haya en results/ y se salta lo que falte. Cuando lleguen los resultados de
B y C, basta con volver a ejecutarlo.

    python plots.py

Qué genera:
  tables/e1_test.md / .csv          Tabla principal de E1 (media ± desviación entre semillas)
  tables/e1_test_by_dataset.csv     E1 desglosado por dataset
  tables/all_results.csv            Todas las métricas de todos los experimentos (formato largo)
  figures/calibration_e1.png        Curva de calibración (nominal frente a observada)
  figures/e2_context.png            Métricas frente al tamaño de contexto
  figures/e3_gaps.png               Métricas frente a la duración del hueco
  figures/e3_forecast.png           Métricas por horizonte de predicción
  figures/e4_timing.png             Tiempo y CRPS frente a los pasos del solver
  figures/examples_<tarea>.png      Trayectorias de ejemplo (contexto, verdad, predicción)
"""
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

RESULTS = C.ROOT / "results"
FIGURES = C.ROOT / "figures"
TABLES = C.ROOT / "tables"

# ---------------------------------------------------------------------------
# Estilo común
# ---------------------------------------------------------------------------
# Cada modelo tiene SIEMPRE el mismo color y marcador en todas las figuras.
# Paleta validada para daltonismo; el marcador distinto evita depender del color.
MODEL_ORDER = ["FlowNP", "TNP", "ANP", "CNP", "GP", "Interp. lineal", "Media contexto"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a7a75", "#6e4fc9"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"

# Métricas que se dibujan: (clave, etiqueta, ¿más alto es mejor?)
METRICS = [("loglik_mgdl", "Log-verosimilitud", True),
           ("rmse_mgdl", "RMSE (mg/dL)", False),
           ("crps_mgdl", "CRPS (mg/dL)", False)]
TABLE_COLS = ["loglik_mgdl", "rmse_mgdl", "mae_mgdl", "crps_mgdl", "cov50", "cov90", "cov95"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 150,
})


def style(models):
    """Color y marcador de cada modelo según su posición en MODEL_ORDER."""
    order = MODEL_ORDER + sorted(m for m in models if m not in MODEL_ORDER)
    return {m: (COLORS[order.index(m) % len(COLORS)], MARKERS[order.index(m) % len(MARKERS)])
            for m in models}


def sort_models(models):
    return sorted(models, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))


def label(key, higher):
    return f"{dict((k, l) for k, l, _ in METRICS)[key]} {'↑' if higher else '↓'}"


# ---------------------------------------------------------------------------
# 1. Leer resultados
# ---------------------------------------------------------------------------
def load_results():
    """Devuelve dict[(modelo, tarea)] -> lista de resultados (uno por semilla)."""
    res = defaultdict(list)
    for f in sorted(RESULTS.glob("*/seed*/*.json")):
        with open(f) as fh:
            r = json.load(fh)
        res[(r["model"], r["task"])].append(r)
    return res


def mean_std(runs, key, dataset=None):
    """Media y desviación entre semillas de una métrica."""
    vals = [(r["overall"] if dataset is None else r["by_dataset"][dataset])[key] for r in runs]
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


# ---------------------------------------------------------------------------
# 2. Tablas
# ---------------------------------------------------------------------------
def make_tables(res):
    # Formato largo: una fila por modelo, tarea y métrica.
    rows = []
    for (model, task), runs in res.items():
        for key in TABLE_COLS:
            m, s, n = mean_std(runs, key)
            rows.append({"model": model, "task": task, "metric": key, "mean": m, "std": s, "n_seeds": n})
    pd.DataFrame(rows).sort_values(["task", "model", "metric"]).to_csv(
        TABLES / "all_results.csv", index=False)

    # Tabla principal E1: "media ± desviación".
    models = sort_models({m for m, t in res if t == "e1_test"})
    if not models:
        return
    table = []
    for model in models:
        runs = res[(model, "e1_test")]
        row = {"Modelo": model, "Semillas": len(runs)}
        for key in TABLE_COLS:
            m, s, _ = mean_std(runs, key)
            row[key] = f"{m:.3f} ± {s:.3f}" if key.startswith("cov") else f"{m:.2f} ± {s:.2f}"
        table.append(row)
    df = pd.DataFrame(table).rename(columns={
        "loglik_mgdl": "LL ↑", "rmse_mgdl": "RMSE ↓", "mae_mgdl": "MAE ↓", "crps_mgdl": "CRPS ↓",
        "cov50": "Cob. 50 %", "cov90": "Cob. 90 %", "cov95": "Cob. 95 %"})
    df.to_csv(TABLES / "e1_test.csv", index=False)
    with open(TABLES / "e1_test.md", "w") as f:
        f.write("E1 (test). LL por punto de target en mg/dL; RMSE, MAE y CRPS en mg/dL. "
                "Media ± desviación entre semillas.\n\n")
        f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")

    # E1 por dataset (solo medias).
    rows = []
    for model in models:
        runs = res[(model, "e1_test")]
        for ds in runs[0]["by_dataset"]:
            row = {"model": model, "dataset": ds, "n_points": runs[0]["by_dataset"][ds]["n_points"]}
            for key in TABLE_COLS:
                row[key] = round(mean_std(runs, key, ds)[0], 3)
            rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES / "e1_test_by_dataset.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Figuras
# ---------------------------------------------------------------------------
def header(fig, axes, title, ncol=4, top=None):
    """Título arriba a la izquierda y una sola leyenda debajo, para toda la figura."""
    top = top or 1 - 0.55 / fig.get_figheight()          # ~0.55 pulgadas para título y leyenda
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, x=0.01, y=0.99, ha="left", va="top", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(handles, labels, loc="lower left", ncol=min(len(labels), ncol),
               bbox_to_anchor=(0.01, top), borderaxespad=0.2)


def fig_calibration(res):
    models = sort_models({m for m, t in res if t == "e1_test"})
    if not models:
        return
    st = style(models)
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#9a9a96", label="Calibración perfecta")
    for m in models:
        runs = res[(m, "e1_test")]
        nominal = runs[0]["overall"]["calibration"]["nominal"]
        observed = np.mean([r["overall"]["calibration"]["observed"] for r in runs], axis=0)
        ax.plot(nominal, observed, lw=2, color=st[m][0], marker=st[m][1], ms=4, label=m)
    ax.set(xlabel="Cobertura nominal", ylabel="Cobertura observada", xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    ax.set_title("Calibración en E1 (test)\n", loc="left", fontweight="bold")
    ax.text(0, 1.025, "Por encima de la diagonal, intervalos demasiado anchos; "
            "por debajo, demasiado estrechos", fontsize=7.5, color=MUTED, transform=ax.transAxes)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_e1.png")
    plt.close(fig)


def fig_sweep(res, tasks, xvals, xlabel, title, fname, logx=False):
    """Paneles (uno por métrica) con una línea por modelo (media ± desviación).

    tasks: nombres de las tareas en orden; xvals: valor del eje x de cada una.
    """
    models = sort_models({m for m, t in res if t in tasks})
    if not models:
        return
    st = style(models)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(10, 3.4))
    for ax, (key, _, higher) in zip(axes, METRICS):
        for m in models:
            pts = [(x, *mean_std(res[(m, t)], key)[:2]) for x, t in zip(xvals, tasks) if (m, t) in res]
            if not pts:
                continue
            x, mu, sd = map(np.array, zip(*pts))
            ax.errorbar(x, mu, yerr=sd, color=st[m][0], marker=st[m][1], ms=5, lw=2,
                        capsize=3, label=m)
        ax.set_xticks(xvals, [str(v) for v in xvals])
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(xvals, [str(v) for v in xvals])
            ax.minorticks_off()
        ax.set(xlabel=xlabel, title=label(key, higher))
    header(fig, axes, title)
    fig.savefig(FIGURES / fname)
    plt.close(fig)


def fig_timing():
    files = sorted(RESULTS.glob("*/e4_timing.json"))
    if not files:
        print("  (sin datos de E4 todavía: se salta e4_timing.png)")
        return
    data = [json.load(open(f)) for f in files]
    st = style([d["model"] for d in data])
    panels = [("time_per_task_s", "Tiempo por tarea (s) ↓"), ("crps_mgdl", "CRPS (mg/dL) ↓")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, (key, ttl) in zip(axes, panels):
        for d in data:
            rows = [r for r in d["rows"] if r.get(key) is not None]
            if rows:
                ax.plot([r["steps"] for r in rows], [r[key] for r in rows], lw=2, ms=5,
                        color=st[d["model"]][0], marker=st[d["model"]][1], label=d["model"])
        ax.set_xscale("log")
        if key == "time_per_task_s":
            ax.set_yscale("log")
        ax.set(xlabel="Pasos del solver ODE", title=ttl)
    header(fig, axes, "E4: coste de inferencia")
    fig.savefig(FIGURES / "e4_timing.png")
    plt.close(fig)


def fig_examples(res, task, title):
    """Cuadrícula: una fila por tarea de ejemplo (dataset) y una columna por modelo."""
    models = sort_models({m for m, t in res if t == task})
    if not models:
        return
    st = style(models)
    first = res[(models[0], task)][0]["examples"]
    ids = sorted(first, key=lambda i: first[i]["dataset"])
    fig, axes = plt.subplots(len(ids), len(models), figsize=(3.2 * len(models), 2.1 * len(ids)),
                             sharex=True, sharey="row", squeeze=False)
    for r, i in enumerate(ids):
        for c, m in enumerate(models):
            ax = axes[r, c]
            ex = res[(m, task)][0]["examples"][i]              # semilla 0 (la primera)
            o = np.argsort(ex["t_target_h"])                   # ordenar por tiempo para las bandas
            t = np.array(ex["t_target_h"])[o]
            g = lambda k: np.array(ex[k])[o]
            color = st[m][0]
            ax.fill_between(t, g("lo90"), g("hi90"), color=color, alpha=0.18, lw=0, label="90 %")
            ax.fill_between(t, g("lo50"), g("hi50"), color=color, alpha=0.35, lw=0, label="50 %")
            for s in ex.get("samples", [])[:3]:                # algunas muestras (solo FlowNP)
                ax.plot(t, np.array(s)[o], color=color, lw=0.5, alpha=0.6)
            ax.plot(t, g("pred"), color=color, lw=1.5, label="Predicción")
            ax.scatter(t, g("gl_target"), s=6, color=INK, alpha=0.5, lw=0, label="Verdad (target)")
            ax.scatter(ex["t_context_h"], ex["gl_context"], s=16, color=INK, marker="x",
                       lw=1, label="Contexto")
            if r == 0:
                ax.set_title(m, color=INK)
            if c == 0:
                ax.set_ylabel(f"{ex['dataset']}\nmg/dL")
            if r == len(ids) - 1:
                ax.set_xlabel("Horas desde el inicio de la ventana")
    header(fig, axes[0], title, ncol=5)
    fig.savefig(FIGURES / f"examples_{task}.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    res = load_results()
    print(f"Resultados encontrados: {len(res)} (modelo, tarea) | "
          f"modelos: {sort_models({m for m, _ in res})}")

    make_tables(res)
    fig_calibration(res)
    fig_sweep(res, [f"e2_nc{k}" for k in C.E2_CONTEXT_SIZES], C.E2_CONTEXT_SIZES,
              "Puntos de contexto", "E2: sensibilidad al tamaño del contexto", "e2_context.png", logx=True)
    fig_sweep(res, [f"e3_gap{g}" for g in C.E3_GAP_MIN], C.E3_GAP_MIN,
              "Duración del hueco (min)", "E3: huecos contiguos del sensor", "e3_gaps.png")
    fig_sweep(res, [f"e3_fc{h}" for h in C.E3_HORIZONS_MIN], C.E3_HORIZONS_MIN,
              "Horizonte de predicción (min)", "E3: predicción del futuro", "e3_forecast.png")
    fig_timing()
    fig_examples(res, "e1_test", "Ejemplos de E1: contexto aleatorio")
    fig_examples(res, "e3_gap120", "Ejemplos de E3: hueco de 2 h")
    fig_examples(res, "e3_fc60", "Ejemplos de E3: predicción a 60 min")
    print(f"Tablas en {TABLES}\nFiguras en {FIGURES}")


if __name__ == "__main__":
    main()
