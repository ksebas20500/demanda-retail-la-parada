# Fundamentos de programación paralela aplicados al proyecto (Unidades 1 y 2)

Cada concepto del temario se enlaza con el archivo del proyecto donde se ve en la práctica. Las cifras citadas están en
`logs/fase5_resumen.csv` (rendimiento por modo) y `logs/fase6_omp_constructos.csv` (constructos de OpenMP).

## Unidad 1

### 1. Programación secuencial, concurrente, paralela y distribuida
| Concepto | Idea | Dónde se ve en el proyecto |
|---|---|---|
| **Secuencial** | Una instrucción tras otra, un solo flujo. | `src/baseline.py` (numpy, 1 hilo) y `gd_omp.c` compilado **sin** `-fopenmp` (`gd_seq.exe`). Es la referencia de todos los *speedups*. |
| **Concurrente** | Varias tareas progresan *solapadas*; pueden compartir un solo núcleo. Aparecen condiciones de carrera y necesidad de sincronizar. | La carrera de datos de `omp_constructos.c` (`g_carrera`): varios hilos suman en el mismo `g[j]` sin protección y el gradiente sale **incorrecto**. |
| **Paralela** | Varias tareas se ejecutan *a la vez* en distintas unidades de cálculo para ir más rápido. | OpenMP (hilos), CUDA (miles de hilos en GPU). |
| **Distribuida** | Procesos con memoria **separada** que cooperan enviándose mensajes (posiblemente en distintas máquinas). | MPI (`gd_mpi.c`): cada proceso lee su bloque de filas y combina el gradiente con `MPI_Allreduce`. **Limitación:** se probó en una sola máquina (procesos locales), no en varios nodos. |

### 2. Componentes físicos y arquitecturas paralelas
- **CPU multinúcleo con memoria compartida (SMP).** i5-12600KF: 10 núcleos físicos / 16 hilos, **híbrida** (6 núcleos de rendimiento con *hyper-threading* + 4 de eficiencia). Todos los hilos ven la misma RAM → modelo de OpenMP.
- **Jerarquía de memoria.** Cachés L1/L2/L3 y RAM. El gradiente recorre ~70 MB por época (1M de registros), más de lo que cabe en caché: el rendimiento queda limitado por el **ancho de banda de memoria**, lo que explica que el *speedup* de CPU se sature (hipótesis coherente con los datos; no se midió el ancho de banda directamente).
- **Memoria distribuida.** Cada proceso MPI tiene su propio espacio de direcciones; para compartir resultados hay que comunicar (`Allreduce` del vector de 28 gradientes por época).
- **SIMD.** Las instrucciones vectoriales (AVX2) del compilador aceleran el producto punto interno; es paralelismo *dentro* de un hilo.
- **GPU (SIMT).** RTX 2060, arquitectura Turing (cc 7.5), 6 GB de VRAM: miles de hilos ligeros ejecutan el mismo kernel sobre distintas filas. Requiere mover datos por el bus PCIe: con los datos residentes en VRAM el *speedup* es el mayor medido; si se transmiten por lotes en cada época, el bus domina y la ganancia casi desaparece (`gd_cuda.cu`, parámetro de lote).
- **Taxonomía de Flynn:** secuencial = SISD; SIMD = vectores; GPU = SIMT (variante de SIMD); OpenMP/MPI con distintos datos por hilo = MIMD.

### 3. Herramientas para programar en paralelo
| Herramienta | Modelo | Uso en el proyecto |
|---|---|---|
| OpenMP (gcc 16.1 y MSVC) | Memoria compartida, directivas `#pragma omp` | `gd_omp.c`, `omp_constructos.c` |
| MPI (MS-MPI 10.1, MSVC) | Paso de mensajes | `gd_mpi.c` |
| CUDA 13.4 (`nvcc`, `sm_75`) | GPU | `gd_cuda.cu` |

### 4. Aplicaciones
Entrenamiento de modelos de *machine learning* (el gradiente es una suma sobre filas: paralelismo de datos), simulación y generación de datos, procesamiento por lotes. En este proyecto: predicción de demanda del comercio minorista fronterizo.

### 5. Sobre el programador en paralelo
Habilidades que el proyecto obligó a ejercitar: (a) **medir antes de afirmar** (varias corridas contaminadas por programas en segundo plano y por el *throttling* de energía de Windows se descartaron y repitieron); (b) **verificar corrección** (todas las versiones reproducen el MAE/RMSE del secuencial); (c) entender que **más hilos no es siempre más rápido** (ley de Amdahl, ancho de banda, sobrecarga de sincronización); (d) documentar la configuración exacta de cada medida.

## Unidad 2 — OpenMP (archivo `src/omp_constructos.c`)

| Tema del temario | Constructo | Aplicación al gradiente |
|---|---|---|
| Crear regiones paralelas | `#pragma omp parallel` + `num_threads`, `default(none)`, `shared`, `private` (variables declaradas dentro), `firstprivate` | `g_critical`, `g_atomic`, `g_carrera`, `g_tasks`, `train_region` |
| Constructo `for` | `#pragma omp for` / `parallel for` + `reduction(+:g[:p])` | núcleo del entrenamiento (`gd_omp.c`) |
| Sincronización | `reduction` vs `critical` vs `atomic` vs sin protección | `demo data_sharing` |
| Dividir bloques entre hilos | `#pragma omp sections` | evaluar validación, prueba y entrenamiento simultáneamente (`demo sections`) |
| Asignar tareas: `single` | `#pragma omp single` dentro de una región `parallel` | un hilo actualiza los pesos y registra el progreso mientras el resto espera en la barrera implícita (`demo single_region`) |
| Asignar tareas: `schedule` | `schedule(static\|dynamic\|guided, chunk)` y `schedule(runtime)` | `demo schedule_uniforme` y `schedule_irregular` |
| Tareas | `#pragma omp task` creadas desde un `single` | un task por bloque de filas (`demo tasks`) |

Ejecutar: `python src/run_omp_constructos.py --tam 1M --reps 5 --pasadas 20` → `logs/fase6_omp_constructos.csv` y `figs/fase6_omp_constructos.png`.
Los resultados interpretados están en `docs/fase6_constructos_resultados.md`.
