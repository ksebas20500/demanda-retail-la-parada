# Fase 0 — Alcance (2026-10-13)

## Tarea de predicción exacta
Predecir la **demanda mensual (unidades vendidas) por categoría de producto y por comercio**
en el sector comercial "La Parada" (Villa del Rosario / Cúcuta, Norte de Santander).
- Unidad de observación: (mes, comercio, categoría). 132 meses (2015-01 a 2025-12) × 12 categorías.
- Variables: fecha, comercio, zona, categoría, precio (COP), tasa de cambio (COP/USD), evento fronterizo (0/1 + tipo), demanda.
- Validación temporal: train 2015-01..2022-12, val 2023-01..2024-06, test 2024-07..2025-12 (sin mezcla aleatoria).
- Métricas: MAE y RMSE (unidades) en test.

## Alcance tecnológico
Python (generador y línea base) + C/C++ con **OpenMP**, **MPI** (varios procesos en una máquina) y **CUDA** (cc 7.5).

## Hardware real de prueba (medido 2026-10-13)
| Recurso | Valor |
|---|---|
| CPU | Intel Core i5-12600KF, 10 núcleos físicos / 16 hilos (6P + 4E) |
| RAM | 15,9 GB |
| GPU | NVIDIA GeForce RTX 2060, 6144 MiB, compute capability 7.5, driver 610.74 |
| SO | Windows 11 Pro 10.0.26200 |
| Python | 3.12.10 |
| Compilador C/C++, CUDA Toolkit, MPI | **No instalados** al inicio (ver decisiones abiertas) |

Nota: al ser una CPU híbrida (P+E), el speedup con >6 hilos no será lineal; se documentará.

## Tamaños de dataset
≈10 mil, ≈100 mil y ≈1 millón de registros (7, 63 y 631 comercios × 1584 filas). Para GPU el millón se procesará por lotes.

## Fuentes de calibración (no sustituyen el caso de estudio)
Favorita, M5, TRM (Banrep), DANE-EMC. **Estado:** no descargadas todavía (requieren decisión/permiso, ver abajo);
el generador actual usa parámetros de negocio explícitos y una TRM *simulada* anclada a niveles aproximados.

## Decisiones abiertas para fases 3-4
1. Instalar toolchain: MSYS2/MinGW-w64 (gcc + OpenMP) o Visual Studio Build Tools, CUDA Toolkit (~3 GB+) y MS-MPI.
2. Descargar TRM real (Banrep) y/o DANE-EMC para sustituir la TRM simulada.
