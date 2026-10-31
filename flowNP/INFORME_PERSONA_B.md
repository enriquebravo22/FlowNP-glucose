# Borrador del informe: parte de B (FlowNP)

> Borrador para pegar en el informe del grupo. Los números salen de `glucose_np/results/FlowNP/` (3 semillas, media ± desviación entre semillas, ddof=0). Las comparaciones con CNP/ANP/TNP/GP se completan cuando C tenga sus resultados.

## 1. Adaptación de FlowNP a series CGM

Partimos del código oficial de FlowNP (Hamad y Rosenbaum, 2025; repo `danrsm/flowNP`). El modelo (`models/fnp.py`) no se ha modificado: un transformer que recibe los puntos de contexto `(x, y)` y los puntos objetivo `(x, y_t, t)` y predice el campo de velocidad del flujo para los objetivos. Usamos la configuración que el repo propone para regresión 1D con GP: posenc de 20 dimensiones, d_model 64, 6 capas, 4 cabezas y feed-forward 128 (224.449 parámetros).

La adaptación consiste en el script `flowNP/glucose.py`, que sustituye al generador de GP del repo por los datos de A (`TrainSampler`, `task_batches`) y conecta la inferencia con el módulo de métricas común (`Evaluator`, `save_timing`). El tiempo de cada ventana de 6 h va en [-1, 1] y la glucosa en z-score con estadísticos de train, igual que para el resto de modelos.

## 2. Entrenamiento

- 100.000 pasos, batch 16, Adam (lr 5·10⁻⁴) con decaimiento coseno, pérdida de flow matching del repo.
- Selección del checkpoint: cada 2.000 pasos se calcula la pérdida de flow matching sobre las 1.792 tareas fijas de `e1_val`, con ruido y tiempos de flujo fijados (semilla 0), de modo que la cifra es determinista y comparable entre checkpoints. La log-verosimilitud por ODE sería un criterio más directo, pero es demasiado cara para evaluarla cada 2.000 pasos.
- 3 semillas (0, 1, 2), ~33 min por semilla en una T4 de Colab. La pérdida de validación del mejor checkpoint fue ≈0,098–0,099 en las tres, con el mínimo cerca del final (paso 96.000 en la semilla 0): el modelo no llega a sobreajustar en 100k pasos.

## 3. Inferencia

**Muestras.** Generamos 100 trayectorias por tarea integrando la ODE del flujo de t=0 (ruido gaussiano) a t=1 con un solver de paso fijo. No usamos el `predict()` del repo, que añade ruido en cada paso (una variante tipo SDE): con la ODE determinista las muestras proceden exactamente del mismo flujo cuya densidad evaluamos. El número de pasos se fijó con E4 (sección 4): midpoint con 20 pasos (40 evaluaciones de la red).

**Log-verosimilitud.** FlowNP define una densidad conjunta sobre todos los puntos objetivo. La calculamos con la fórmula de cambio de variable, integrando la ODE hacia atrás (midpoint, 100 pasos) y acumulando la divergencia del campo de velocidad. Detalles relevantes:

- La divergencia es la traza del Jacobiano Nt×Nt de cada tarea, porque la atención acopla todos los puntos objetivo. Al validar la implementación encontramos que la opción `exact_divergence=True` de la librería `flow_matching` **no es correcta** para FlowNP: trata cada una de las B·Nt filas como si fuera independiente y devuelve sumas de columnas del Jacobiano en lugar de su traza (comprobado numéricamente: 0,069 frente a 0,162 en el mismo caso). Por eso implementamos la divergencia nosotros, exacta (Nt productos vector-Jacobiano) o con el estimador de Hutchinson, que es insesgado.
- Precisión frente a coste (5 batches de `e2_nc20`): con divergencia exacta, la LL por punto es −2,507 / −2,649 / −2,678 / −2,685 / −2,687 con 20 / 50 / 100 / 200 / 400 pasos. Con 100 pasos el error de discretización es < 0,01 nats. Hutchinson con 100 pasos da −2,655 y es ~6 veces más rápido (9 s frente a 52 s); su ruido se promedia sobre las ~1.900 tareas de cada experimento. Configuración final: midpoint 100 + Hutchinson.
- La LL conjunta se entrega al `Evaluator`, que la divide entre Nt y la pasa a mg/dL. Así es comparable con la LL por punto de los modelos gaussianos.

La evaluación completa (11 experimentos) tarda ~75 min por semilla en una T4.

## 4. E4: tiempo de muestreo frente a pasos del solver

Medimos el tiempo de muestreo puro (sin LL, con `cuda.synchronize`) en `e1_test` con 100 muestras por tarea y batch 16 en una T4, para Euler (1–200 pasos) y midpoint (1–50 pasos; cada paso son 2 evaluaciones de la red). Semilla 0.

| Solver | Pasos | NFE | ms / tarea | CRPS (mg/dL) | Cob. 90 % |
|---|---:|---:|---:|---:|---:|
| Euler | 1 | 1 | 5,2 | 4,43 | 0,416 |
| Euler | 5 | 5 | 29,2 | 3,78 | 0,552 |
| Euler | 10 | 10 | 60,3 | 3,66 | 0,689 |
| Euler | 20 | 20 | 121,2 | 3,60 | 0,791 |
| Euler | 50 | 50 | 303,5 | 3,59 | 0,867 |
| Euler | 100 | 100 | 607,8 | 3,59 | 0,893 |
| Euler | 200 | 200 | 1.215 | 3,60 | 0,905 |
| Midpoint | 1 | 2 | 11,7 | 3,84 | 0,692 |
| Midpoint | 5 | 10 | 60,4 | 3,61 | 0,804 |
| Midpoint | 10 | 20 | 121,1 | 3,59 | 0,864 |
| **Midpoint** | **20** | **40** | **241,9** | **3,59** | **0,896** |
| Midpoint | 50 | 100 | 607,3 | 3,60 | 0,913 |

Observaciones:

- El coste es lineal en el número de evaluaciones de la red (~6 ms por NFE y tarea con 100 muestras) y no depende del número de puntos objetivo: todos se generan en paralelo.
- El CRPS converge muy pronto (3,59 mg/dL desde ~20 NFE). Ya con 1 paso de Euler (5 ms) el CRPS, 4,43, es mejor que el de la interpolación lineal (5,78).
- La cobertura converge mucho más despacio que el CRPS. Con pocos pasos las muestras salen demasiado concentradas: el solver grueso "encoge" la dispersión del flujo. Midpoint alcanza la misma cobertura que Euler con aproximadamente la mitad de NFE, y por eso lo elegimos.
- Elegimos midpoint con 20 pasos: CRPS ya convergido y cobertura 0,896, frente a 0,905 de Euler 200, que es 5 veces más caro.

La figura `e4_timing.png` muestra solo la curva de Euler (1–200 pasos). Los datos de midpoint, incluida la configuración final, están en la tabla y en `e4_all.json`.

La comparación con TNP (H3) se hace con el tiempo que mida C en las mismas condiciones (T4, 100 muestras, batch 16, `e1_test`). Un TNP autorregresivo necesita una pasada secuencial por cada punto objetivo (~42 de media en `e1_test`) y, además, cada pasada procesa una secuencia más larga (contexto + objetivos).

## 5. Resultados de FlowNP (E1–E3)

LL en nats por punto en mg/dL (↑). RMSE, MAE y CRPS en mg/dL (↓). Cobertura nominal del 90 %. Media ± desviación entre 3 semillas. Las baselines son deterministas (desviación ≈ 0).

### E1: tarea general (`e1_test`, contexto 5–50 puntos al azar)

| Modelo | LL | RMSE | MAE | CRPS | Cob. 90 % |
|---|---|---|---|---|---|
| **FlowNP** | **−2,190 ± 0,006** | **10,59 ± 0,07** | **4,85 ± 0,00** | **3,59 ± 0,00** | 0,890 ± 0,007 |
| Interp. lineal | −3,818 | 11,08 | 5,54 | 5,78 | 0,991 |
| Media contexto | −4,761 | 40,84 | 28,04 | 19,22 | 0,894 |

### E2: tamaño del contexto (mismo objetivo, contextos anidados)

| Nc | FlowNP LL | Interp. LL | FlowNP RMSE | Interp. RMSE | FlowNP CRPS | Interp. CRPS | FlowNP cob. 90 |
|---|---|---|---|---|---|---|---|
| 5 | −3,152 ± 0,005 | −4,513 | 25,54 ± 0,68 | 25,09 | 10,53 ± 0,16 | 13,15 | 0,940 |
| 10 | −2,811 ± 0,006 | −4,106 | 14,19 ± 0,14 | 15,21 | 5,70 ± 0,02 | 8,04 | 0,908 |
| 20 | −2,417 ± 0,004 | −3,772 | 7,65 ± 0,02 | 8,56 | 3,02 ± 0,00 | 5,11 | 0,880 |
| 50 | −1,914 ± 0,005 | −3,491 | 3,48 ± 0,01 | 3,88 | 1,42 ± 0,00 | 3,37 | 0,848 |

### E3: huecos y predicción

| Experimento | FlowNP LL | Interp. LL | FlowNP RMSE | Interp. RMSE | FlowNP CRPS | Interp. CRPS | FlowNP cob. 90 |
|---|---|---|---|---|---|---|---|
| Hueco 30 min | −2,260 ± 0,005 | −3,864 | 7,31 ± 0,02 | 8,06 | 3,09 | 5,37 | 0,876 |
| Hueco 60 min | −2,392 ± 0,006 | −4,168 | 11,20 ± 0,07 | 12,72 | 5,04 | 7,81 | 0,861 |
| Hueco 120 min | −2,527 ± 0,017 | −4,563 | 18,87 ± 0,21 | 20,74 | 8,70 | 12,32 | 0,852 |
| Hueco 180 min | −2,593 ± 0,034 | −4,856 | 28,70 ± 0,97 | 28,37 | 13,05 | 16,96 | 0,881 |
| Predicción 30 min | −2,568 ± 0,006 | −4,206 | 13,42 ± 0,05 | 16,10 | 6,07 | 8,71 | 0,873 |
| Predicción 60 min | −2,621 ± 0,012 | −4,610 | 21,81 ± 0,03 | 25,28 | 9,96 | 13,57 | 0,837 |

### E1 por dataset (media de 3 semillas)

| Dataset | Puntos | FlowNP LL | Interp. LL | FlowNP RMSE | Interp. RMSE | FlowNP CRPS | Interp. CRPS | FlowNP cob. 90 |
|---|---:|---|---|---|---|---|---|---|
| Broll | 1.457 | −1,89 | −3,79 | 6,83 | 6,95 | 2,62 | 5,15 | 0,869 |
| Colas | 9.015 | −0,97 | −3,77 | 5,40 | 5,48 | 1,59 | 4,92 | 0,927 |
| Dubosson | 657 | −1,66 | −3,85 | 15,58 | 14,57 | 4,31 | 6,51 | 0,900 |
| Hall | 9.375 | −2,20 | −3,80 | 7,94 | 8,19 | 3,04 | 5,36 | 0,876 |
| Weinstock | 60.431 | −2,38 | −3,83 | 11,50 | 12,08 | 3,98 | 5,98 | 0,888 |

### Análisis

1. **FlowNP supera a las baselines en todos los experimentos en LL, CRPS y MAE.** La ganancia en LL es de 1,3–2,3 nats por punto. El CRPS baja entre un 20 % (Nc=5) y un 58 % (Nc=50) respecto a la interpolación lineal.
2. **En RMSE la ventaja es menor.** En los dos escenarios con menos información, Nc=5 y hueco de 180 min, FlowNP empata con la interpolación (25,5 ± 0,7 frente a 25,1 y 28,7 ± 1,0 frente a 28,4). Ahí la media condicional está mal determinada y la interpolación lineal ya es un estimador razonable de la tendencia. La ventaja de FlowNP en esos casos está en la distribución (LL, CRPS), no en la predicción puntual.
3. **La ventaja crece con la dificultad del hueco y del horizonte.** Frente a la interpolación, la diferencia de LL pasa de 1,60 nats (hueco de 30 min) a 2,26 (180 min), y de 1,64 a 1,99 en predicción de 30 a 60 min. FlowNP aprende la dinámica típica de la glucosa (subidas posprandiales, vuelta a la media) en lugar de unir puntos con rectas.
4. **Frente a la interpolación, la ventaja no crece cuando el contexto es escaso.** La diferencia en LL es casi constante (1,36 / 1,30 / 1,36 / 1,58 nats para Nc = 5/10/20/50) y la mejora relativa en CRPS es mayor con más contexto. La hipótesis H2 compara FlowNP con CNP/ANP, así que hay que contrastarla con los resultados de C. Este dato solo indica que la interpolación no es la referencia adecuada para H2.
5. **Calibración.** La interpolación lineal es muy conservadora (cobertura del 90 % entre 0,97 y 0,99, con intervalos demasiado anchos). FlowNP está cerca del nominal en E1: 0,521 / 0,890 / 0,936 para los intervalos del 50 / 90 / 95 %. La curva de calibración (`calibration_e1.png`) va casi sobre la diagonal, ligeramente por encima en los niveles bajos y ligeramente por debajo en el 90–95 %: el centro de la predictiva es algo ancho y las colas algo ligeras. FlowNP pero se vuelve algo sobreconfiado con mucho contexto (0,848 con Nc=50) y en predicción a 60 min (0,837). Con pocos datos (Nc=5) es algo conservador (0,940). Parte de la sobreconfianza se debe al solver: en E4 la cobertura de `e1_test` sube de 0,896 (midpoint 20) a 0,913 (midpoint 50) sin que cambie el CRPS. Con más pasos mejoraría la calibración a cambio de 2,5 veces más tiempo.
6. **Por dataset.** FlowNP mejora la LL y el CRPS en los cinco datasets. En RMSE gana en cuatro y pierde en Dubosson (15,6 frente a 14,6 mg/dL), el dataset más pequeño (657 puntos de test), poco representado en train y con picos posprandiales bruscos que el modelo no puede anticipar sin información de comidas (ver `examples_e3_gap120.png`). La LL varía mucho entre datasets (−0,97 en Colas frente a −2,38 en Weinstock): Colas tiene series más estables y Weinstock, con el 75 % de los puntos, domina la media global. La LL de la interpolación es casi idéntica en todos (≈ −3,8), lo que indica que su incertidumbre no se adapta a la variabilidad de cada población.
7. **Estabilidad entre semillas.** La desviación entre semillas es muy pequeña (LL ≤ 0,034 nats; CRPS ≤ 0,4 mg/dL) y las diferencias con las baselines son de un orden de magnitud mayor que esa variabilidad.

## 6. Limitaciones

- Hiperparámetros tomados del repo para GP 1D, sin búsqueda. No se exploraron modelos más grandes.
- La LL usa Hutchinson con 1 vector de prueba: es insesgada por tarea, pero su estimación individual es ruidosa. Las medias sobre ~1.900 tareas son estables.
- La pérdida de flow matching usada para elegir el checkpoint no es la LL. Un criterio basado en LL podría elegir otro checkpoint, aunque la curva de validación es casi plana al final del entrenamiento.
- La cobertura depende del número de pasos del solver (sección 4). Las métricas puntuales y el CRPS ya han convergido con la configuración elegida.
- Las ventanas de train están dominadas por Weinstock (74 %). Conviene revisar los resultados por dataset (`glucose_np/tables/e1_test_by_dataset.csv`).
