# Flow Matching Neural Processes for CGM glucose trajectories

Code for *Probabilistic Imputation and Forecasting of Glucose Trajectories with Flow Matching Neural Processes* (Álvaro Inclán Balán, Enrique Bravo Díaz, Jaime Morata Bermúdez — Universidad Autónoma de Madrid, Máster en IA Generativa, 2026).

We compare **FlowNP** (Hamad & Rosenbaum, NeurIPS 2025) with **CNP**, **ANP**, an autoregressive **TNP** and a **Matérn-5/2 Gaussian process** on the five public CGM datasets of [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench), with patient-level splits and fixed evaluation tasks:

| Experiment | Question |
|---|---|
| E1 | General imputation (5–50 random context points) |
| E2 | Effect of the context size (5 / 10 / 20 / 50 points, nested) |
| E3 | Realistic gaps (30–180 min) and 30–60 min forecasting |
| E4 | Sampling time versus solver steps (T4 GPU) |

## Tutorial

**[`tutorial.ipynb`](tutorial.ipynb)** walks through the whole pipeline: data and tasks, training a small FlowNP and CNP, sampling trajectories, scoring them, and reproducing the main results. It needs no GPU.

**Locally** (Python 3.10–3.12, CPU is enough):

```bash
git clone https://github.com/enriquebravo22/FlowNP-glucose.git
cd FlowNP-glucose
pip install torch flow-matching==1.0.10 attrdictionary pyyaml tqdm numpy pandas scipy matplotlib jupyter
jupyter notebook tutorial.ipynb
```

**On Google Colab**: open `tutorial.ipynb` from GitHub (*File → Open notebook → GitHub*) and run all cells; the first cell clones the repository and installs the two extra packages.

In both cases the notebook downloads GlucoBench and builds the data itself (a few minutes the first time).

## Repository layout

```
tutorial.ipynb  step-by-step tutorial (start here)
glucose_np/   data pipeline, fixed tasks, metrics and figures (shared by all models)
  01_preprocess.py … 06_baselines.py, run_all.sh
  config.py, sampling.py (TrainSampler, task_batches), metrics.py (Evaluator, save_timing)
  plots.py      -> tables/ and figures/ from results/
  results/      metrics of every model, task and seed (JSON)
flowNP/       official FlowNP code (danrsm/flowNP) + our scripts
  glucose.py            train / eval / E4 for FlowNP
  baselines.py          train / eval / E4 for CNP, ANP and TNP
  gp_matern.py          Matérn-5/2 GP (fit + eval)
  models/tnpd.py        TNP-D (optional Gaussian TNP)
  configs/glucose/*.yaml
  colab_glucose_v2.ipynb, colab_persona_c.ipynb   Colab notebooks used for the experiments
```

## Reproducing

Everything was run on Google Colab with a T4 GPU.

```bash
# 1. Data (GlucoBench raw CSVs are not redistributed here)
cd glucose_np
pip install -r requirements.txt
git clone https://github.com/IrinaStatsLab/GlucoBench.git
unzip GlucoBench/raw_data.zip -d tmp && mkdir -p data/raw && cp tmp/raw_data/*.csv data/raw/
bash run_all.sh                      # windows, splits, fixed tasks, reference baselines (~2 min)

# 2. Models (100k steps each, 3 seeds)
cd ../flowNP
pip install flow-matching==1.0.10 attrdictionary
python glucose.py   --mode=train --seed=0              # FlowNP
python glucose.py   --mode=eval  --seed=0
python baselines.py --model=cnp  --mode=train --seed=0 # also anp, tnpa
python baselines.py --model=cnp  --mode=eval  --seed=0
python gp_matern.py --mode=fit   --seed=0 && python gp_matern.py --mode=eval --seed=0

# 3. E4 timing
python baselines.py --model=tnpa --mode=e4 --seed=0 --eval_batch_size=16 --num_samples=100

# 4. Tables and figures
cd ../glucose_np && python plots.py
```

Training is resumable (`--resume`) and evaluation skips tasks that already have results.

## Main results (`e1_test`, mean over 3 seeds)

| Model | LL ↑ | RMSE ↓ | CRPS ↓ | Cov. 90 % |
|---|---|---|---|---|
| **FlowNP** | **−2.190** | 10.59 | **3.59** | 0.890 |
| TNP-A | −2.269 | 10.98 | 3.83 | 0.932 |
| ANP | −3.065 | **10.40** | 3.82 | 0.924 |
| GP Matérn | −3.306 | 12.88 | 4.70 | 0.939 |
| CNP | −3.745 | 14.61 | 6.56 | 0.899 |
| Linear interp. | −3.818 | 11.08 | 5.78 | 0.991 |

LL in nats per point (mg/dL); RMSE and CRPS in mg/dL. On a T4, FlowNP samples in 242 ms per task versus 472 ms for TNP-A, with better CRPS. Full tables in `glucose_np/tables/`.

## Credits

`flowNP/` is based on the official FlowNP implementation (https://github.com/danrsm/flowNP) and keeps its license. Data: GlucoBench (Sergazinov et al., 2024).
