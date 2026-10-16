# Predicción de demanda del comercio minorista · La Parada (Norte de Santander)

Prototipo de la monografía de **Sistemas Inteligentes** (Universidad de Pamplona): predice la demanda mensual por categoría y por comercio
en la zona de frontera de La Parada (Villa del Rosario / Cúcuta) con *machine learning* sobre datos sintéticos, y **mide el efecto de paralelizar
el entrenamiento** (OpenMP, MPI y CUDA) sobre el tiempo y sobre la precisión.

## Resultados principales (1 millón de registros, 300 épocas)
| Modo | Tiempo | Veces más rápido que C con 1 núcleo |
|---|---|---|
| C secuencial (referencia) | 2,16 s | 1,00× |
| OpenMP (16 hilos) | 1,26 s | 1,71× |
| MPI (12 procesos, 1 PC) | 1,39 s | 1,55× |
| **CUDA, datos en la GPU (RTX 2060)** | **0,45 s** | **4,85×** |
| CUDA, datos por lotes | 2,20 s | 0,98× |

La **precisión es idéntica** en todos los modos (MAE 176,8 unidades). Los tiempos de CPU varían hasta ±30 % entre corridas.
Hardware: Intel i5-12600KF (10 núcleos / 16 hilos) + NVIDIA RTX 2060 (6 GB). Detalle y limitaciones: [`docs/fase5_desarrollo_conclusiones.md`](docs/fase5_desarrollo_conclusiones.md).

> Los datos son **sintéticos** (la TRM sí es la serie oficial de datos.gov.co). El modelo es una regresión lineal básica que hoy **no supera** a «copiar el mes pasado»: es un banco de pruebas para el paralelismo.

## Estructura
| Carpeta / archivo | Contenido |
|---|---|
| `src/generar_datos.py`, `explorar_datos.py` | Generador sintético (3 tamaños) y gráficas exploratorias |
| `src/features.py`, `baseline.py` | Variables sin fuga de información, división temporal y línea base secuencial |
| `src/gd_omp.c`, `gd_mpi.c`, `gd_cuda.cu`, `omp_constructos.c` | Entrenamiento en C: secuencial/OpenMP, MPI, CUDA y constructos de OpenMP |
| `src/run_*.py`, `consolidar.py` | Ejecutan los benchmarks y consolidan resultados |
| `app/` | Interfaz web (Streamlit): **Recorrido guiado** (feria) y **Modo experto** |
| `build/*.bat` | Scripts de compilación del PC con Windows (MSVC, gcc de MSYS2, CUDA, MS-MPI) |
| `logs/` | Resultados medidos (CSV) y registro de experimentos |
| `figs/` | Gráficas finales |
| `docs/` | Alcance, resultados, teoría de las Unidades 1-2 y [guion de presentación](docs/guion_presentacion.md) |
| `README_DEMO.md`, `empaquetar_demo.py` | Paquete portátil para la demo en otro equipo (p. ej. MacBook sin CUDA) |

## Inicio rápido
**Interfaz (cualquier equipo con Python 3.11+):**
```
python -m pip install -r requirements_demo.txt
python -m streamlit run app/app.py        # Windows: abrir_interfaz.bat · macOS: iniciar_mac.command
```
Los datos se generan solos al abrir la interfaz. Guía completa para la feria: [`README_DEMO.md`](README_DEMO.md).

**Reproducir los experimentos (PC con Windows + toolchain):**
```
python -m pip install -r requirements.txt
python src/generar_datos.py --trm-csv data/externos/trm_diaria.csv   # data/ventas_{10k,100k,1M}.csv (semilla 42)
python src/features.py                                               # binarios para C/CUDA en data/proc
build\compilar_todo.bat                                              # requiere VS Build Tools, MSYS2/gcc, CUDA 13.x, MS-MPI
python src/baseline.py --reps 5
python src/run_openmp.py --reps 5 && python src/run_mpi.py --reps 5 && python src/run_cuda.py --reps 5
python src/run_omp_constructos.py --tam 1M
python src/consolidar.py                                             # logs/fase5_resumen.csv + figs/fase5_*.png
```
Para tiempos fiables cierra programas pesados antes de medir. En Windows 11 los programas desactivan el *power throttling* (EcoQoS) que degradaba las mediciones.

## Documentación
- [`docs/fase0_alcance.md`](docs/fase0_alcance.md) · alcance, tarea y hardware
- [`docs/fase5_desarrollo_conclusiones.md`](docs/fase5_desarrollo_conclusiones.md) · Desarrollo, Conclusiones y Limitaciones (cifras finales)
- [`docs/fase6_constructos_resultados.md`](docs/fase6_constructos_resultados.md) · constructos de OpenMP medidos
- [`docs/fundamentos_teoria.md`](docs/fundamentos_teoria.md) · teoría de las Unidades 1 y 2 ligada al proyecto
- [`docs/guion_presentacion.md`](docs/guion_presentacion.md) · guion para presentar el sistema

## Notas sobre el repositorio
- No se versionan los datos generados ni los ejecutables: se recrean con los comandos de arriba (`data/`, `build/*.exe`, `dist/` están en `.gitignore`).
- `data/externos/trm_diaria.csv` (TRM oficial 2014-2026) sí se incluye: es pequeño y necesario para generar los datos.
