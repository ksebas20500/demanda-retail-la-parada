# Guion de presentación · Predicción de demanda en La Parada

Material para el presentador (no aparece en la interfaz). Todas las cifras salen de las mediciones del proyecto
(`logs/fase5_resumen.csv`, `logs/fase2_baseline.csv`, `logs/fase6_omp_constructos.csv`).

**Contenido:** 1. Resumen en 1 minuto · 2. Cómo funciona por dentro · 3. Computación paralela explicada · 4. Guion estación por estación · 5. Versiones de 2 y 5 minutos · 6. Preguntas y respuestas · 7. Qué NO decir · 8. Cifras para memorizar

---

## 1. Resumen en 1 minuto
> «Los comercios de La Parada, en la frontera de Villa del Rosario y Cúcuta, tienen ventas muy cambiantes: suben en diciembre, dependen del dólar y se desploman cuando cierran la frontera.
> Construimos un sistema que **aprende de 11 años de ventas** y predice cuánto se venderá por categoría y por comercio.
> Pero entrenarlo con muchos datos es lento, así que probamos tres formas de **repartir el trabajo entre varios procesadores y la tarjeta gráfica** (OpenMP, MPI y CUDA) y medimos cuánto acelera, y si la respuesta cambia. Resultado: **la tarjeta gráfica entrenó casi 5 veces más rápido y la precisión no cambió.**»

Tres ideas que el público debe llevarse: **(1)** datos → modelo → pronóstico; **(2)** paralelizar es repartir el trabajo; **(3)** la GPU ganó, pero solo si los datos están dentro de ella.

---

## 2. Cómo funciona por dentro (para entender y responder)

### 2.1 Los datos
- Cada **fila** es: un mes + un comercio + una categoría de producto, con su precio, el dólar (TRM), si hubo evento de frontera y las **unidades vendidas** (la demanda).
- 132 meses (2015-2025) × 12 categorías × N comercios. Tamaños: **10 mil** (11.088 filas, 7 comercios), **100 mil** (99.792 filas, 63 comercios) y **1 millón** (999.504 filas, 631 comercios).
- Los datos son **sintéticos**: un generador aplica reglas realistas (temporada por categoría, sensibilidad al dólar, eventos de frontera y ruido). **El dólar es la serie oficial** (Superfinanciera, datos.gov.co). No existe un registro público de las ventas de La Parada.
- Eventos incluidos: cierre de frontera 2015, cierre de puentes 2019, pandemia, reapertura 2022 y paros aleatorios.

### 2.2 Las «pistas» (variables) que usa el modelo — 27
Ventas del mes pasado, de hace 2, 3 y 6 meses y del mismo mes del año pasado; promedio de los últimos 3 y 12 meses; nivel y cambio del dólar; precio; si hay evento (este mes y el pasado); época del año; zona; y la categoría. **Nunca usa información del futuro** (sin «trampa»).

### 2.3 Cómo se divide la historia (importante para que la prueba sea justa)
- **Entrenar:** 2016 → 2022. **Validar:** 2023 → jun 2024. **Probar:** jul 2024 → 2025.
- El modelo se evalúa con meses que **nunca vio**.

### 2.4 El modelo y cómo aprende
- Es una **regresión lineal**: predice sumando cada pista multiplicada por un **peso**. (Trabaja con el logaritmo de la demanda y lo devuelve a unidades.)
- **Descenso de gradiente**: al inicio todos los pesos valen 0 (predice 0). En cada **época**:
  1. Recorre **todas** las filas de entrenamiento y calcula cuánto se equivocó en cada una.
  2. Suma esos errores en 28 números (el *gradiente*), que indican hacia dónde mover cada peso.
  3. Corrige los pesos un poquito.
- Se repite **300 veces**. En la pantalla se ve cómo la línea naranja (predicción) se acerca a la azul (real).

### 2.5 Dónde entra el paralelismo
El paso 1 (recorrer todas las filas, 300 veces) es el trabajo pesado: con 1 millón de filas son cientos de millones de cuentas. Como cada fila se calcula **independiente** de las demás, se pueden **repartir las filas** entre varios trabajadores y luego **sumar** sus resultados parciales.

### 2.6 Cómo se mide
Se entrena siempre lo mismo (300 épocas, mismos parámetros) en cada modo y se compara el **tiempo** y el **error en datos de prueba**. Cada cifra es el promedio de 5 repeticiones.
Referencia para el *speedup*: el mismo programa escrito en C usando **1 núcleo** (= 1×).

### 2.7 Pronóstico del futuro
Para pronosticar meses futuros se usa la proyección **recursiva**: el mes pronosticado alimenta las pistas del siguiente. Los **escenarios** (dólar, precios, evento) son supuestos del usuario. Para el futuro el modelo se **reentrena con todo el historial** (práctica habitual tras validar); la evaluación de precisión sigue usando el modelo entrenado solo hasta 2022.

---

## 3. Computación paralela explicada (lenguaje simple)

| Término | Idea | Analogía |
|---|---|---|
| **Secuencial** | Un solo trabajador hace todo, una cosa tras otra | Una sola cajera atendiendo toda la fila |
| **OpenMP** | Varios **hilos** de un mismo programa que **comparten la memoria** | Varias cajeras en el mismo local que ven la misma lista |
| **MPI** | Varios **procesos separados** que se **envían mensajes** (podrían estar en computadores distintos) | Sucursales distintas que se llaman para sumar sus ventas |
| **CUDA / GPU** | **Miles** de hilos pequeños en la tarjeta gráfica | Un salón con miles de calculadoras sencillas trabajando a la vez |
| **Speedup** | Cuántas veces más rápido que la referencia | 2× = la mitad del tiempo |
| **Carrera de datos** | Dos trabajadores escriben a la vez en el mismo dato y se pisan | Dos personas tachando la misma casilla en una hoja |

### Qué se midió (1 millón de filas, entrenamiento de 300 épocas)
| Modo | Tiempo | Veces más rápido que C con 1 núcleo |
|---|---|---|
| Python, 1 núcleo | 3,40 s | 0,64× |
| **C, 1 núcleo (referencia)** | 2,16 s | 1,00× |
| OpenMP (mejor: 16 hilos) | 1,26 s | 1,71× |
| MPI (mejor: 12 procesos) | 1,39 s | 1,55× |
| **Tarjeta gráfica, datos dentro de la GPU** | **0,45 s** | **4,85×** |
| Tarjeta gráfica, datos enviados por lotes | 2,20 s | 0,98× |

**Lecturas clave**
- **La precisión es idéntica en todos los modos** (error medio de 176,8 unidades con 1 millón de filas): paralelizar cambia el tiempo, no la respuesta.
- **La GPU gana solo si los datos ya están dentro**: si hay que enviarlos por el cable PCIe en cada época, el viaje se come la ganancia.
- **En CPU la ganancia es modesta (≈1,5–1,7×)** y se estanca al sumar núcleos. Una explicación coherente es el cuello de botella de la memoria (es una hipótesis, no se midió directamente).
- Los tiempos de CPU **varían hasta ±30 % entre corridas**; los de la GPU son muy estables.
- Medido en el PC del laboratorio: Intel i5-12600KF (10 núcleos, 16 hilos) + NVIDIA RTX 2060 (6 GB). El Mac de la feria **no tiene GPU NVIDIA**: ahí se muestran estas mediciones.

### Los constructos de OpenMP (si preguntan por la parte técnica de la clase)
- **Carrera de datos:** sin proteger la suma el gradiente salió con **45–81 % de error**; con `reduction` o `critical` el error fue ≈ 0.
- **`atomic`** en cada suma fue **50–70 veces más lento** que `reduction`.
- **`schedule`:** `dynamic`/`guided` ganaron ≈1,6–1,8× **solo con carga irregular**; con carga uniforme (el caso real) no hay diferencia.
- **`sections`** (evaluar tres conjuntos a la vez): 1,46×. **`single`** y **`task`**: no mejoraron este problema.

---

## 4. Guion estación por estación

> Navegación: botones **Siguiente ▶ / ◀ Anterior** abajo, o la barra de estaciones arriba. Tiempo orientativo para el recorrido de 5 minutos entre paréntesis.

### Inicio (0:00)
**Muestra:** título y 4 tarjetas (registros, comercios, categorías, años).
**Di:** abre con una pregunta: *«¿Cuánto creen que vende una tienda de La Parada en diciembre?»* Explica en una frase qué es La Parada y anuncia las tres partes: datos, cómo aprende, por qué paralelo. *«Todo lo que van a ver se calcula en vivo en este computador.»*
**Haz:** pulsa *Comenzar el recorrido*.

### 1 · El problema (0:30)
**Muestra:** gráfico de la demanda total mensual con el dólar y los eventos (franjas rojas).
**Di:** *«Las ventas suben por temporada, dependen del dólar y se desploman con los cierres de la frontera.»* Señala las franjas rojas (cierre 2015, cierre 2019, pandemia, reapertura 2022) y las tarjetas: diciembre vende ≈ +14 % sobre un mes normal (13,7 % con 100 mil registros; 12,8 % con 1 millón); el dólar pasó de ≈2.400 a ≈4.900 pesos; 43 de los 132 meses tuvieron algún evento de frontera.
**Haz:** apaga/enciende el dólar y los eventos con los interruptores para que vean el efecto.
**Mensaje:** *un negocio que no anticipa esto compra de más o se queda sin inventario.*

### 2 · Los datos (1:15)
**Muestra:** selector de tamaño (10 mil / 100 mil / 1 millón), categorías, gráfico de estacionalidad y tabla de ejemplo.
**Di:** *«Cada fila es un mes, un comercio y una categoría.»* Aclara que son datos **simulados con reglas reales** y el dólar es el real.
**Haz:** pide a alguien del público que elija una categoría (juguetes → diciembre; útiles escolares → enero). Lee el mensaje verde («vende más en…»).
**Mensaje:** *más datos = más trabajo al entrenar* (con 1 millón es ~100 veces más que con 10 mil). Si cambias a 1 millón en un portátil, avisa que tarda unos segundos en prepararse.

### 3 · Cómo aprende (2:00) — la estación estrella
**Muestra:** botón *Entrenar el modelo en vivo*, dos gráficos que se animan.
**Haz:** elige una categoría, velocidad *Normal* y **pulsa Entrenar**.
**Di mientras corre:** *«Empieza sin saber nada (predice cero). En cada época compara con lo que realmente se vendió y corrige. Miren la línea naranja acercarse a la azul.»* Señala el gráfico de la derecha: el error baja.
**Al terminar:** muestra el gráfico *«¿Qué aprendió?»*: las pistas que más pesan (ventas del mes pasado, promedio del año…). *«La mejor pista del futuro es el pasado reciente.»*
**Gancho:** *«Ojo: recorrer todas las filas 300 veces es lo que vamos a acelerar con computación paralela.»*

### 4 · ¿Acierta? (3:00)
**Muestra:** gráfico real vs modelo vs «copiar el mes pasado», con la zona verde de meses de prueba.
**Di:** *«La zona verde son meses que el modelo nunca vio.»* Explica la línea gris: una regla ingenua («vendo lo mismo que el mes pasado») como vara de comparación.
**Haz:** que el público elija comercio y categoría.
**Sé honesto:** *«Nuestro modelo es una regresión lineal básica; hoy copiar el mes pasado acierta un poco más. Es a propósito: lo que medimos en este proyecto es cuánto acelera la computación paralela, no construir el mejor modelo.»* (Con 100 mil registros: error del modelo 195 vs 178 de copiar.)

### 5 · ¿Qué pasará? (3:40)
**Muestra:** controles de meses, dólar, precios y evento; gráfico con historia, año pasado y pronóstico.
**Haz:** pide un escenario al público: *«¿y si el dólar sube 20 %?»*, *«¿y si cierran la frontera?»* y mueve los controles.
**Di:** *«El sistema recalcula en vivo; cada mes pronosticado alimenta al siguiente.»* Aclara: son **escenarios hipotéticos**, no una predicción del dólar ni de la frontera. Menciona que para el futuro el modelo se reentrena con todo el historial.

### 6 · Computación paralela (4:20)
**Muestra:** barra con las filas repartidas entre trabajadores; luego el gráfico de velocidad.
**Haz:** mueve el control de trabajadores (1 → 4 → 16) para que vean cómo se reparten las filas.
**Di:** *«Repartimos las filas entre trabajadores; cada uno calcula su parte del error y al final se suman los resultados.»* Presenta OpenMP, MPI y CUDA con las analogías de la sección 3.
**Muestra el gráfico:** *«La tarjeta gráfica fue casi 5 veces más rápida y la precisión no cambió.»* Luego la honestidad: *«En procesadores la ganancia fue modesta, 1,5 a 1,7 veces: más núcleos no siempre acelera.»*
**Si preguntan por qué la GPU «solo a veces»:** abre el desplegable (datos dentro de la GPU 0,45 s vs enviados por lotes 2,2 s).
**Importante:** en el Mac se **muestran mediciones del laboratorio** (lo dice la pantalla). En el PC con GPU existe el botón para repetir la medición en vivo.

### 7 · Conclusiones (5:00)
**Muestra:** las 5 conclusiones y el recuadro de honestidad.
**Di:** cierra con la frase: *«Datos → modelo → casi 5 veces más rápido con la GPU, sin perder precisión.»* Invita preguntas; las preguntas frecuentes están desplegables en pantalla.

---

## 5. Versiones del recorrido

### 2 minutos (visitante que pasa)
1. Inicio: la pregunta y la frase de 1 minuto (sección 1).
2. Estación 3: pulsar **Entrenar en vivo** y señalar la línea naranja acercándose.
3. Estación 6: el gráfico de velocidad. *«La GPU casi 5 veces más rápido, misma precisión.»*

### 5 minutos (recomendada)
Sigue la sección 4 con los tiempos indicados.

### 10 minutos (con preguntas o público técnico)
Los 5 minutos + **Modo experto** (arriba a la derecha): *Sistema completo* (generar un dataset propio y entrenar con el motor elegido), *Rendimiento paralelo* (pestaña «Constructos de OpenMP» con la carrera de datos) y *Predicciones* (dónde falla el modelo).

---

## 6. Preguntas y respuestas

**¿Los datos son reales?**
No: son simulados con reglas realistas (temporada, eventos de frontera, dólar). El dólar sí es la serie oficial. No existe un registro público de las ventas de La Parada.

**¿Por qué un modelo tan simple?**
Para tener un cálculo claro y repetible con el cual medir el efecto del paralelismo. Mejorar el modelo es el siguiente paso; un modelo más complejo se podría paralelizar con las mismas ideas.

**¿Paralelizar cambia el resultado?**
No. El error es el mismo en los cinco modos (diferencias menores a una milésima de unidad).

**¿Cuál es la diferencia entre OpenMP, MPI y CUDA?**
OpenMP: hilos que comparten memoria en un mismo computador. MPI: procesos separados que se envían mensajes (pueden estar en computadores distintos). CUDA: miles de hilos pequeños en la tarjeta gráfica NVIDIA.

**¿Por qué más núcleos no aceleran proporcionalmente?**
Cada época recorre ~70 MB de datos y la memoria no da abasto para tantos núcleos a la vez (explicación coherente con las mediciones; no se midió directamente). Además cada hilo debe esperar para sumar los resultados.

**¿Por qué la GPU gana solo con los datos dentro?**
Calcula muy rápido, pero los datos viajan por el bus PCIe desde la memoria del computador. Con los datos dentro: 0,45 s; enviándolos por lotes en cada época: 2,2 s.

**¿Funciona sin tarjeta gráfica?**
Sí. Todo el recorrido funciona en cualquier computador (aquí, un MacBook). La GPU solo se usó para las mediciones de velocidad.

**¿Podría entrenar con más de un millón de registros?**
Sí con cambios menores (hoy el generador está limitado a 631 comercios ≈ 1 millón). Límites estimados, **sin medir**: ~10 millones por memoria RAM del PC (16 GB), ~19 millones por el tamaño de archivo de los programas en C, y ~46 millones en una GPU de 6 GB si los datos caben.

**¿Qué es la «carrera de datos»?**
Cuando dos hilos escriben a la vez en el mismo dato sin coordinarse. En la prueba el gradiente salió con un 45–81 % de error. Se evita con `reduction`, `critical` o `atomic`.

**¿Por qué el modelo no supera a «copiar el mes pasado»?**
Es una regresión lineal básica sobre datos con muchos cambios bruscos (eventos de frontera). Con 1 millón de filas: error del modelo 176,8 vs 170,5 de copiar; un árbol de decisión sí lo supera (152,3). No fue el objetivo del proyecto.

**¿Cuánto tarda en cargar con 1 millón?**
En un PC de escritorio, decenas de segundos la primera vez (se calculan las variables). En el portátil puede ser más; por eso la demo usa 100 mil por defecto.

**¿Qué sigue?**
Un modelo más preciso, MPI en varios computadores reales, datasets de decenas de millones de registros y calibrar con datos reales de ventas minoristas.

---

## 7. Qué NO decir (para no exagerar)
- ❌ «Predice las ventas reales de La Parada» → ✅ «predice sobre datos **simulados** con reglas realistas».
- ❌ «La GPU siempre es mejor» → ✅ «ganó **cuando los datos están dentro de ella**; por lotes no ganó».
- ❌ «Más núcleos = proporcionalmente más rápido» → ✅ «en CPU ganamos 1,5–1,7× y se estanca».
- ❌ «El modelo es muy preciso» → ✅ «es un modelo simple a propósito; hoy copiar el mes pasado acierta un poco más».
- ❌ «Probamos MPI en un clúster» → ✅ «MPI con varios procesos en **un solo computador**».
- ❌ «El pronóstico dice qué pasará con el dólar» → ✅ «son **escenarios supuestos**».
- ❌ Dar un tiempo de CPU como exacto → ✅ «varía hasta ±30 % entre corridas».

---

## 8. Cifras para memorizar
| Dato | Valor |
|---|---|
| Registros (10k / 100k / 1M) | 11.088 / 99.792 / 999.504 |
| Filas de entrenamiento (10k / 100k / 1M) | 7.056 / 63.504 / 636.048 |
| Historia | 2015–2025 (132 meses), 12 categorías, 7 / 63 / 631 comercios |
| Variables del modelo | 27 (+1 valor base) · 300 épocas |
| Error medio en prueba (modelo) 10k / 100k / 1M | 139,8 / 195,4 / 176,8 unidades |
| Error de «copiar el mes pasado» 10k / 100k / 1M | 116,8 / 177,8 / 170,5 unidades |
| Mejor speedup (1M) | GPU con datos dentro: **4,85×** (0,45 s vs 2,16 s) |
| CPU (1M) | OpenMP 1,71× (16 hilos) · MPI 1,55× (12 procesos) |
| GPU por lotes (1M) | 0,98× (2,2 s) |
| Memoria que ocupa el millón | ~71 MB de variables · 74 MiB de VRAM (1,2 % de 6 GB) |
| Hardware de medición | Intel i5-12600KF (10 núcleos/16 hilos) + NVIDIA RTX 2060 6 GB |
| Constructos OpenMP | carrera 45–81 % de error · `atomic` 50–70× más lento · `schedule` 1,6–1,8× (carga irregular) · `sections` 1,46× |

---

## Plan B si algo falla
- No abre la interfaz o falla el Mac: mostrar capturas/video y explicar con el guion; el contenido del proyecto está en `docs/fase5_desarrollo_conclusiones.md`.
- La estación 3 tarda o se traba: pulsa *Rápida* en la velocidad o salta a la estación 4 (el modelo ya está entrenado al abrir).
- Se cambia a 1 millón y tarda: es normal la primera vez en un portátil; vuelve a 100 mil.
