# Datos y evaluación para FlowNP en series CGM (Persona A)

Esta carpeta contiene todo lo que comparten los tres miembros del grupo: los datos ya procesados, la partición por paciente, las tareas fijas de evaluación (E1–E3), el módulo de métricas y el script que genera todas las tablas y figuras.

## 1. Ejecución

```bash
pip install -r requirements.txt
# Datos: los CSV de GlucoBench vienen en su repo (raw_data.zip)
git clone https://github.com/IrinaStatsLab/GlucoBench.git
unzip GlucoBench/raw_data.zip -d tmp && mkdir -p data/raw && cp tmp/raw_data/*.csv data/raw/
bash run_all.sh            # ejecuta los pasos 01 → 06 y plots.py (unos 2 min)
```

El zip ya incluye las salidas (`data/splits`, `data/windows`, `data/tasks`, `results`, `tables`, `figures`), así que **B y C no necesitan ejecutar nada** para empezar a trabajar.

## 2. Archivos

| Archivo | Qué hace | Salida |
|---|---|---|
| `config.py` | Todos los parámetros y rutas, más `load_windows(split)` | — |
| `01_preprocess.py` | Limpia las series, recorta a [40, 400] mg/dL y corta en segmentos por huecos de más de 60 min. **No interpola.** | `data/processed/readings.csv` |
| `02_split.py` | Reparte los pacientes 70/15/15 dentro de cada dataset (semilla 42) | `data/splits/split_v1.json` |
| `03_windows.py` | Ventanas de 6 h; tiempo → [-1, 1]; glucosa → z-score con estadísticos de train | `data/windows/*.npz`, `norm_stats.json` |
| `sampling.py` | Batches en formato TNP: `TrainSampler` (entrenamiento) y `task_batches` (evaluación) | — |
| `04_make_tasks.py` | Genera una sola vez las tareas fijas de E1, E2 y E3 | `data/tasks/*.npz` |
| `metrics.py` | `Evaluator`: LL, RMSE, MAE, CRPS, coberturas, calibración y ejemplos. `save_timing` para E4 | `results/<modelo>/seed<k>/<tarea>.json` |
| `05_check_metrics.py` | Comprueba el módulo de métricas (datos sintéticos; gaussiano frente a muestras) | `checks/calibration_check.png` |
| `06_baselines.py` | Dos baselines de referencia en todos los experimentos y con 3 semillas | `results/...` |
| `plots.py` | Lee `results/` y genera todas las tablas y figuras. Se vuelve a ejecutar cuando llegan modelos nuevos | `tables/`, `figures/` |
| `run_all.sh` | Ejecuta todo en orden | — |

## 3. Cómo lo usan B y C

```python
from config import load_windows
from sampling import TrainSampler, task_batches
from metrics import Evaluator, save_timing

# --- Entrenamiento (misma API que GPSampler del repo de TNP) ---
sampler = TrainSampler(load_windows("train"), seed=0)
batch = sampler.sample(batch_size=16, device="cuda")   # batch.xc, yc, xt, yt, x, y: [B, N, 1]

# --- Validación / early stopping: tareas fijas de val ---
for batch in task_batches("e1_val", batch_size=16, device="cuda"): ...

# --- Evaluación: repetir para cada experimento y cada semilla de entrenamiento ---
ev = Evaluator(task="e1_test", model="TNP", seed=0)
for batch in task_batches("e1_test", batch_size=16, device="cuda"):
    mu, sigma = model(batch)                       # gaussianos: [B, Nt, 1]
    ev.add_gaussian(batch, mu, sigma)
    # FlowNP: ev.add_samples(batch, samples [S, B, Nt, 1], loglik [B, Nt] o [B])
ev.save()                                           # -> results/TNP/seed0/e1_test.json

# --- E4 (solo B): una fila por nº de pasos del solver ---
save_timing("FlowNP", [{"steps": 10, "time_per_task_s": 0.012, "crps_mgdl": 15.3}, ...])
```

Reglas del contrato:
- Todo lo que entra al `Evaluator` va en **espacio normalizado**. Él lo pasa a mg/dL.
- **Nombres de modelo:** `FlowNP`, `TNP`, `ANP`, `CNP` y `GP`. Así cada uno mantiene su color en todas las figuras.
- **Experimentos a evaluar:** `e1_test`, `e2_nc5`, `e2_nc10`, `e2_nc20`, `e2_nc50`, `e3_gap30`, `e3_gap60`, `e3_gap120`, `e3_gap180`, `e3_fc30` y `e3_fc60`. La lista está en `06_baselines.py` (`TASKS`).
- **Semillas:** al menos 3 entrenamientos por modelo (seed 0, 1 y 2), para la media ± desviación.
- **Log-verosimilitud de FlowNP:** si la ODE da la log-verosimilitud conjunta del target, se pasa con forma `[B]` y el módulo la divide entre Nt.
- **Trayectorias:** no hay que hacer nada; el `Evaluator` guarda solo las predicciones de una tarea por dataset.

## 4. Datos resultantes

| | Pacientes | Ventanas |
|---|---|---|
| Train | 333 | 15.350 |
| Val | 72 | 1.792 |
| Test | 72 | 1.922 |

Hay 477 pacientes y 871.020 lecturas de los cinco datasets de GlucoBench (Broll, Colas, Dubosson, Hall y Weinstock). Para la normalización, la media es 156,2 mg/dL y la desviación, 78,4 mg/dL, calculadas solo con train.

## 5. Experimentos (`data/tasks/`)

| Nombre | Contexto | Target | Tareas |
|---|---|---|---|
| `e1_val` / `e1_test` | 5–50 puntos al azar | el resto | 1.792 / 1.922 |
| `e2_nc5 … nc50` | 5 / 10 / 20 / 50 puntos (anidados) | el mismo en los cuatro | 1.841 |
| `e3_gap30 … gap180` | fuera del bloque (máx. 50) | bloque contiguo de 30–180 min | 1.922 |
| `e3_fc30` / `e3_fc60` | 4 h de historia (de 1 h a 5 h) | los siguientes 30 / 60 min | 1.918 / 1.922 |

## 6. Métricas

Todas son por punto de target, promediadas sobre todos los puntos del experimento, globales y por dataset.

| Métrica | Modelos gaussianos | FlowNP (muestras) |
|---|---|---|
| `loglik_norm`, `loglik_mgdl` ↑ | log N(y; μ, σ); en mg/dL se resta log σ_train | la que calcula B con la ODE; mismo cambio de unidades |
| `rmse_mgdl`, `mae_mgdl` ↓ | sobre μ | RMSE sobre la media y MAE sobre la mediana de las muestras |
| `crps_mgdl` ↓ | fórmula cerrada | estimador por muestras E\|X−y\| − ½E\|X−X'\| |
| `cov50`, `cov90`, `cov95` | μ ± zσ | cuantiles empíricos |
| `calibration` | nominal frente a observada, del 5 % al 95 % | ídem |

## 7. Tablas y figuras (`plots.py`)

| Archivo | Contenido |
|---|---|
| `tables/e1_test.md` / `.csv` | Tabla principal de E1, media ± desviación entre semillas |
| `tables/e1_test_by_dataset.csv` | E1 por dataset |
| `tables/all_results.csv` | Todas las métricas de todos los experimentos (formato largo) |
| `figures/calibration_e1.png` | Curva de calibración (hipótesis 2) |
| `figures/e2_context.png` | LL, RMSE y CRPS frente al tamaño de contexto |
| `figures/e3_gaps.png` | LL, RMSE y CRPS frente a la duración del hueco |
| `figures/e3_forecast.png` | LL, RMSE y CRPS por horizonte de predicción |
| `figures/e4_timing.png` | Tiempo y CRPS frente a los pasos del solver (aparece cuando B guarde E4) |
| `figures/examples_*.png` | Trayectorias: contexto, verdad, predicción, intervalos del 50/90 % y muestras |

## 8. Decisiones tomadas

**Datos**
- **Valores fuera de rango: recortar y marcar.** Weinstock guarda "LOW"/"HIGH" como 39/401. Se recortan a 40/400 y se marcan con `censored`. `summary(exclude_censored=True)` los excluye.
- **Segmentos de menos de 6 h descartados**, porque no dan ninguna ventana. Se pierde un 1,5 % de las lecturas.
- **Ventana de 6 h** (hasta 72 lecturas). Muchos pacientes solo tienen 1–3 días de datos, y es un tamaño parecido al de las tareas 1D de los repos de NP.
- **Paso entre ventanas:** 3 h en train (más datos) y 6 h en val/test (ventanas independientes).
- **Partición estratificada por dataset**, para que los cinco estén en train, val y test.

**Muestreo**
- **Entrenamiento:** Nc común por batch y target = el resto de puntos, con Nt ajustado a la ventana más corta del batch.
- **E2 sin contexto de 100:** no cabe en una ventana de 6 h. Usamos las mismas ventanas (≥ 60 lecturas) y el mismo target para los cuatro tamaños, de modo que solo cambia el contexto.
- **E3, contexto ≤ 50 puntos**, como en entrenamiento. Un único hueco por ventana, con duración fija por experimento.

**Métricas**
- **FlowNP:** media de las muestras para el RMSE y mediana para el MAE, porque son los estimadores óptimos de cada error.

## 9. Resultados de los baselines (E1, test)

| Modelo | LL (mg/dL) | RMSE | MAE | CRPS | Cob. 90 % |
|---|---|---|---|---|---|
| Interp. lineal | −3,82 | 11,1 | 5,5 | 5,8 | 0,99 |
| Media contexto | −4,76 | 40,8 | 28,0 | 19,2 | 0,89 |

**Importante:** la interpolación lineal es un rival fuerte, con un RMSE de 3,9 mg/dL cuando hay 50 puntos de contexto en E2, porque la glucosa CGM es una señal suave. Los NP deben superarla claramente, sobre todo en log-verosimilitud y calibración. Su incertidumbre es demasiado ancha: queda por encima de la diagonal. Los baselines se pueden dejar en las tablas como referencia o borrar sus carpetas de `results/`.

## 10. A tener en cuenta

- Weinstock aporta el 74 % de las ventanas de train. Conviene reportar también por dataset (`tables/e1_test_by_dataset.csv`).
- El 46 % de las ventanas de train tiene huecos reales, es decir, menos de 72 lecturas.
- En "Interp. lineal" las muestras son independientes en cada punto, por eso se ven dentadas en las figuras. Las de FlowNP serán trayectorias suaves.
