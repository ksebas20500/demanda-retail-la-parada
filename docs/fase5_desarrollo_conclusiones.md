# Desarrollo, Conclusiones e Índice (texto para la monografía)

Fuente de todas las cifras: `logs/fase5_resumen.csv` (5 repeticiones por configuración; media ± desviación), corrida final del 2026-10-13 (21:30-21:37).
Equipo: i5-12600KF (10 núcleos / 16 hilos, híbrido), 15,9 GB RAM, RTX 2060 6 GB (cc 7.5), Windows 11.
Speedup = t(C secuencial gcc, 1 hilo) / t(modo). El tiempo es solo el entrenamiento (300 épocas, lote completo); excluye la lectura de datos.
Detalle de los constructos de OpenMP: `docs/fase6_constructos_resultados.md`; marco teórico: `docs/fundamentos_teoria.md`.

## Índice propuesto
1. Alcance y entorno (Fase 0) · 2. Generador de datos sintéticos (Fase 1) · 3. Línea base secuencial (Fase 2) ·
4. Paralelización con OpenMP (Fase 3) · 5. MPI y CUDA (Fase 4) · 6. Validación de resultados (Fase 5) ·
7. Constructos de OpenMP y fundamentos teóricos (Unidades 1 y 2) · 8. Conclusiones y recomendaciones · 9. Limitaciones

## Desarrollo

**Objetivo específico 1 – Datos.** Se generó un conjunto sintético con la demanda mensual de 12 categorías en comercios de La Parada (2015-2025), con estacionalidad por categoría, sensibilidad a la TRM, eventos fronterizos y ruido autorregresivo. La TRM es la serie oficial de datos.gov.co (4.081 días). Tamaños: 11.088, 99.792 y 999.504 registros. El mes del año explica el 13,9 % de la varianza de log-demanda (0,14 % con meses permutados) y el tipo de evento el 40,1 % (`logs/experimentos.csv`, fase 1).

**Objetivo específico 2 – Línea base.** División temporal (entrenamiento 2016-01..2022-12, validación 2023-01..2024-06, prueba 2024-07..2025-12), variables de rezago sin fuga de información. Regresión lineal por descenso de gradiente: MAE de prueba 139,8 / 195,4 / 176,8 unidades (10k/100k/1M); árbol de decisión (prof. 12): 136,6 / 177,2 / 152,3. La regresión lineal **no supera** al pronóstico ingenuo del mes anterior en ningún tamaño (MAE ingenuo lag-1: 116,8 / 177,8 / 170,5); solo el árbol lo supera en 100k y 1M (`logs/fase2_baseline.csv`). El modelo lineal es una línea base para medir paralelismo, no un modelo optimizado.

**Objetivo específico 3 – Paralelismo.** El kernel (gradiente de lote completo) se reimplementó en C y se paralelizó con OpenMP (gcc), MPI (MSVC + MS-MPI, varios procesos en una máquina) y CUDA (nvcc, sm_75). Todas las versiones reproducen el MAE/RMSE de Python (diferencias < 0,001 unidades): la paralelización no altera la precisión, solo el tiempo.

| Dataset 1M (636.048 filas de entrenamiento) | Config | Tiempo (s) | Speedup vs C seq |
|---|---|---|---|
| Python (numpy, 1 hilo) | 1 hilo | 3,398 ± 0,097 | 0,64 |
| C secuencial (gcc) | 1 hilo | 2,159 ± 0,009 | 1,00 |
| OpenMP | 16 hilos (mejor) | 1,265 ± 0,021 | 1,71 |
| OpenMP | 6 hilos | 1,903 ± 0,085 | 1,13 |
| MPI | 12 procesos (mejor) | 1,393 ± 0,062 | 1,55 |
| CUDA | todo en VRAM (74 MiB) | 0,445 ± 0,005 | **4,85** |
| CUDA | lotes de 262.144 filas | 2,200 ± 0,068 | 0,98 |

En 100k los mejores fueron MPI con 12 procesos (3,44×), OpenMP con 16 hilos (3,40×) y CUDA (2,57×); en 10k, MPI con 8 procesos (3,25×, con variación de 19 % entre repeticiones). Respecto a numpy en 1M, CUDA da 3,398/0,445 ≈ 7,6×.

Gráficas: `figs/fase5_speedup_hilos_procesos.png`, `figs/fase5_escalabilidad_datos.png`, `figs/fase5_precision_por_modo.png`.

**Variabilidad entre corridas (importante).** Una corrida previa del mismo experimento (cuyo CSV se sobrescribió, por eso no es trazable a un archivo) dio en 1M OpenMP 6 hilos ≈2,0× y MPI 6 procesos ≈1,9×, frente a 1,13× y 1,39× en la corrida final; en cambio CUDA con datos residentes fue prácticamente igual (4,83× vs 4,85×). Es decir, **las cifras de CPU varían hasta ~±30 % entre corridas**, mientras que la de GPU es estable. Además, en la corrida final el speedup de CPU **no crece monótonamente** (OpenMP: 4 hilos 1,36×, 6 hilos 1,13×, 8 hilos 1,61×, 16 hilos 1,71×). No se investigó la causa; es compatible con la planificación de una CPU híbrida (núcleos P y E) y con aplicaciones en segundo plano, pero eso es una hipótesis.

## Conclusiones y recomendaciones (cada una trazable a la tabla)
1. **La precisión no cambia con el modo de ejecución** (MAE 176,83 y RMSE 307,62 en 1M para los cinco modos).
2. **El speedup de CPU es modesto y se satura**: máximo ≈1,7× (OpenMP) y ≈1,55× (MPI) en 1M, muy lejos del ideal lineal. Una explicación coherente es que el kernel está limitado por ancho de banda de memoria (en 100k, donde los datos caben mejor en caché, se alcanza ≈3,4×), pero **no se midió el ancho de banda directamente**: es una hipótesis.
3. **CUDA es el mejor modo en 1M (4,85×) solo si los datos residen en VRAM**; si se transmiten por lotes por PCIe, el tiempo (≈2,2 s) iguala al secuencial (≈0,98×) y el beneficio desaparece, independientemente del tamaño de lote probado (16.384 a 262.144).
4. **Con 10k registros el paralelismo aporta poco y es ruidoso**: tiempos de milisegundos dominados por la sobrecarga.
5. **Constructos OpenMP** (ver `fase6_constructos_resultados.md`): una carrera de datos produjo un gradiente con 45-81 % de error; `atomic` fue ~50-70× más lento que `reduction`; `schedule(dynamic/guided)` mejoró ≈1,6-1,8× solo con carga irregular; `sections` dio 1,46×; `single` y `task` no mejoraron este problema.
6. Recomendación: para este modelo lineal, usar CUDA con datos residentes para volúmenes ≥1M; en CPU, 4-16 hilos/procesos dan ganancias modestas. Para ganar precisión hay que mejorar el modelo, no la paralelización.

## Limitaciones (declararlas en la monografía)
- Datos sintéticos; la calibración con Favorita/M5 (Kaggle, requieren cuenta) y DANE-EMC **no se realizó**. Las fechas y magnitudes de los eventos son aproximaciones, no datos observados.
- Compiladores distintos: OpenMP con gcc 16.1, MPI con MSVC; el speedup de MPI se calcula contra el C secuencial de gcc (MPI con 1 proceso: 2,62 s vs 2,16 s de gcc, 21 % más lento solo por el compilador).
- CPU híbrida (6 P + 4 E) y equipo de uso general. Se descartaron y repitieron corridas contaminadas por aplicaciones abiertas y por el *power throttling* (EcoQoS) de Windows 11, que degradaba los procesos en segundo plano hasta ~2× entre repeticiones; la corrida final desactiva ese límite en todos los programas (`SetProcessInformation`).
- El límite de 6 GB de VRAM **no se alcanzó**: el 1M ocupa 74 MiB. No se probó un tamaño mayor.
- La generación de datos no se distribuyó con MPI (solo el entrenamiento) y MPI se probó en una sola máquina, no en varios nodos.
- Para 10k/100k, numpy es más rápido que el C secuencial usado como referencia (los speedups vs numpy son menores).
- Los ejecutables compilados se recompilaron después de la medición solo para añadir la opción de guardar los pesos (no cambia el código cronometrado).
