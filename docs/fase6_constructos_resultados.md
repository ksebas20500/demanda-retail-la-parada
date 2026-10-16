# Resultados de los constructos de OpenMP (Unidad 2)

Fuente: `logs/fase6_omp_constructos.csv` (media de 5 repeticiones, dataset 1M = 636.048 filas de entrenamiento, 28 variables; gcc 16.1 `-O3 -march=native -fopenmp`; 2026-10-13).
Cada “pasada” = un cálculo completo del gradiente. Los tiempos absolutos de este programa **no son comparables** con los de la Fase 3 (otro código de prueba); lo que importa es la comparación entre variantes dentro de cada demo.

## 1. `parallel` + cláusulas de datos y sincronización (`demo data_sharing`)
| Variante | 2 hilos | 6 hilos | 16 hilos | Error del gradiente |
|---|---|---|---|---|
| `reduction(+:g[:p])` | 6,05 ms | 3,60 ms | 3,92 ms | ≈ 0 (1e-15) |
| `critical` (acumulador privado + fusión) | 6,40 ms | 4,09 ms | 4,09 ms | ≈ 0 |
| `atomic` en cada suma | 207,6 ms | 256,0 ms | 282,4 ms | ≈ 0 |
| **sin protección (carrera de datos)** | 22,9 ms | 30,9 ms | 51,2 ms | **45 % / 78 % / 81 %** |

- La carrera de datos produce un gradiente **incorrecto** (error relativo de 45-81 %) y además es más lenta que `reduction` (los hilos se pelean por las mismas líneas de caché). Es la demostración práctica de por qué hay que sincronizar.
- `reduction` y `critical` con acumulador privado son equivalentes en tiempo; `atomic` por elemento es ~50-70× más lento.

## 2. `for` + `schedule`
**Carga uniforme** (todas las filas cuestan lo mismo): `static`, `dynamic` y `guided` quedan dentro del ruido (3,7-4,4 ms con 6 hilos; variación entre repeticiones 4-20 %). Solo `static,1` (reparto cíclico fila a fila) es peor: 6,6 ms (6 hilos) y 7,4 ms (16 hilos), probablemente por perder localidad de caché (no medido directamente).

**Carga irregular** (coste que crece con el índice de fila; prueba sintética para mostrar la semántica del `schedule`):
| Variante | 6 hilos | 16 hilos |
|---|---|---|
| `static` (bloques contiguos) | 23,3 ms | 11,3 ms |
| `static,64` | 13,3 ms | 9,0 ms |
| `dynamic,64` | 12,8 ms | 7,3 ms |
| `dynamic,1024` | 12,8 ms | 7,2 ms |
| `dynamic,16384` | 13,7 ms | 8,7 ms |
| `guided,64` | 12,2 ms | 7,1 ms |

Con carga desbalanceada, `dynamic`/`guided` son ≈1,8× (6 hilos) y ≈1,6× (16 hilos) más rápidos que `static`, porque el último hilo del reparto estático se queda con las filas más caras. En el problema **real** (carga uniforme) no hay ganancia: por eso el proyecto usa `static`.

## 3. `sections`
Evaluar validación, prueba y entrenamiento en tres `section` simultáneas: 11,1 ms → 7,6 ms (**1,46×**). Está limitado porque las tres secciones no tienen el mismo tamaño (la de entrenamiento es la más larga) y solo hay 3 unidades de trabajo.

## 4. `single` (región única) frente a `parallel for` por época
Una sola región `parallel` con `single` para actualizar pesos y registrar progreso, frente a abrir/cerrar una región en cada una de las 300 épocas:
1,61 s vs 1,62 s (2 hilos), 1,37 s vs 1,34 s (6 hilos), 1,39 s vs 1,34 s (16 hilos). **No hay diferencia apreciable**: el coste de crear la región (microsegundos) es despreciable frente a los ~4 ms de trabajo por época. Ambas versiones dan exactamente los mismos pesos que la de 1 hilo (error 0).

## 5. `task`
Un `task` por bloque de filas creado desde un `single`: 3,72 ms (bloques de 1024), 3,90 ms (16384), 3,64 ms (65536), igual que `parallel for` + `reduction` (3,6-4,1 ms). Las tareas aportan flexibilidad (útiles con trabajo irregular o recursivo) pero aquí no aceleran.

## Lectura general
Elegir el constructo adecuado importa por **corrección** (sincronización) y por **balanceo de carga** (`schedule` en trabajo irregular); con una carga uniforme y regular como el gradiente, `parallel for + reduction + schedule(static)` ya es la mejor opción y el resto no mejora los tiempos.
