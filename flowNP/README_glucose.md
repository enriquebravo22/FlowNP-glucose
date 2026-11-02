# FlowNP sobre glucosa CGM (persona B)

Este es el repo oficial de FlowNP (`danrsm/flowNP`) con un script nuevo, `glucose.py`. El modelo (`models/fnp.py`) se usa sin cambios. Los datos, las tareas y las métricas salen de `../glucose_np/` (persona A): `TrainSampler`, `task_batches`, `Evaluator` y `save_timing`.

## Estructura esperada

```
PROYECTO/
├── glucose_np/            # persona A (datos, tareas, métricas, plots)
└── flowNP/                # este repo
    ├── glucose.py         # train / eval / e4
    ├── configs/glucose/fnp.yaml
    ├── colab_glucose.ipynb
    └── output/results/glucose/<expname>/fnp/seed<k>/   # ckpt.tar, best.tar, logs
```

## Entorno: todo se ejecuta en Google Colab (T4)

1. Sube `PROYECTO_colab.zip` (carpetas `glucose_np/` y `flowNP/`) a la raíz de Drive.
2. Abre `flowNP/colab_glucose.ipynb` en Colab con GPU T4 y ejecuta las celdas en orden. La primera descomprime el zip en `MyDrive/PROYECTO/`.
3. Los checkpoints (`flowNP/output/`) y las métricas (`glucose_np/results/FlowNP/`) se escriben en Drive. Si la sesión expira no se pierde nada: al volver a ejecutar, se reanuda o se salta lo ya hecho.
4. Para que A y C usen los resultados, basta con compartirles la carpeta `PROYECTO/` de Drive.

Colab ya trae torch; solo hace falta `pip install flow-matching==1.0.10 attrdictionary`. Las dependencias completas están en `requirements_glucose.txt`.

## Uso

```bash
python glucose.py --mode=train --seed=0        # 100k pasos; best.tar = mínimo de la pérdida FM en e1_val
python glucose.py --mode=e4    --seed=0        # E4 -> glucose_np/results/FlowNP/e4_timing.json
python glucose.py --mode=eval  --seed=0        # 11 experimentos -> glucose_np/results/FlowNP/seed0/
```

`--max_batches N` limita la evaluación a los primeros batches, para depurar. En ese caso hay que usar `--model_name` con otro nombre para no ensuciar `results/FlowNP`.

## Decisiones

- **Hiperparámetros:** los del repo para GP 1D (d_model 64, 6 capas, 4 cabezas, posenc 20, Adam 5e-4 con cosine, 100k pasos y batch 16).
- **Early stopping:** pérdida de flow matching sobre las 1.792 tareas fijas de `e1_val`, con ruido y tiempos de flujo fijos (semilla 0). Así es determinista y comparable entre checkpoints. La LL por ODE sería mejor criterio, pero es demasiado cara para calcularla durante el entrenamiento.
- **Muestras:** ODE determinista de t=0 a t=1 con paso fijo (Euler o midpoint), 100 muestras por tarea. No usamos el `predict()` del repo, que añade ruido en cada paso (pseudo-SDE). Así las muestras vienen del mismo flujo cuya densidad se evalúa.
- **Log-verosimilitud:**
  - Se calcula la LL conjunta `log p(yt | contexto)` con la fórmula de cambio de variable, integrando hacia atrás con midpoint en 100 pasos, y se pasa al `Evaluator` con forma `[B]` (él divide entre Nt).
  - La divergencia es la traza del Jacobiano Nt×Nt de cada tarea. Se puede calcular exacta (`--ll_div exact`, Nt pasadas backward) o con Hutchinson (`--ll_div hutchinson --ll_probes K`), que es insesgado.
  - **No se usa `compute_likelihood(exact_divergence=True)` de `flow_matching`.** Trata cada una de las B·Nt filas como independiente y devuelve la suma de columnas del Jacobiano, no la traza (comprobado numéricamente: 0,069 frente a 0,162). La opción por defecto del repo, Hutchinson, sí es correcta en esperanza.
- **E4:**
  - Mide el tiempo de muestreo puro (con `cuda.synchronize`) en `e1_test`, con 100 muestras y batch 16, para 1–200 pasos con Euler y midpoint. Midpoint hace 2 evaluaciones de la red por paso; la columna `nfe` lo recoge.
  - La figura de A usa el primer método. Todo queda en `e4_all.json`.

## Coste (medido en CPU durante el desarrollo)

- Muestreo de un batch (B=16, S=100, midpoint 100 pasos): ~210 s en CPU.
- LL Hutchinson / exacta: 3,4 s / 26 s por batch.
- La evaluación completa tiene 2.229 batches por semilla (A agrupa por (Nc, Nt)), así que se hace en GPU. E4 da los tiempos en la T4.
