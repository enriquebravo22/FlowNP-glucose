#!/bin/bash
# Ejecuta toda la parte de datos y evaluación en orden.
# Requiere los CSV de GlucoBench en data/raw/ (ver README).
set -e
python 01_preprocess.py     # limpieza y segmentos
python 02_split.py          # partición por paciente
python 03_windows.py        # ventanas de 6 h normalizadas
python 04_make_tasks.py     # tareas fijas de E1, E2 y E3
python 05_check_metrics.py  # comprobación del módulo de métricas
python 06_baselines.py      # baselines de referencia en todos los experimentos
python plots.py             # tablas y figuras
