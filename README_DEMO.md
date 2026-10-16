# Demo para la feria · Predicción de demanda en La Parada

Sistema que aprende de las ventas y pronostica la demanda, y que muestra cómo la computación paralela (OpenMP, MPI, CUDA) acelera el entrenamiento.
Funciona **en cualquier computador** (probado el flujo completo en Windows; en macOS no he podido probarlo físicamente, ver «Antes de la feria»).

## Opción A · Correr todo en el MacBook Air (recomendada para la feria: no depende de internet ni de otro PC)
1. **Una sola vez, con internet y ANTES del evento:** instalar Python 3.11 o superior desde <https://www.python.org/downloads/>.
2. Descomprimir `demanda_la_parada_demo.zip`, abrir la carpeta y hacer **doble clic en `iniciar_mac.command`**
   (si macOS lo bloquea: clic derecho → Abrir; si no abre: Terminal → `chmod +x iniciar_mac.command` y de nuevo).
   La primera vez crea un entorno e instala 4 paquetes (~1–2 min). Después abre la interfaz sola en <http://localhost:8501>.
3. Las siguientes veces solo doble clic: **no necesita internet**.

## Opción B · Mostrarlo desde el MacBook usando el PC con GPU (tiene los motores reales)
En el PC Windows: doble clic en `abrir_interfaz_red.bat` (muestra la IP). En el Mac, en el navegador: `http://ESA-IP:8501`.
Ambos equipos deben estar en la **misma red WiFi** y puede pedirse permitir Python en el Firewall. Depende de la red del evento: lleva la Opción A como respaldo.

## Qué incluye y qué no depende de la GPU
- **Todo el recorrido** (datos, aprendizaje en vivo, predicciones, pronóstico) se calcula en vivo en cualquier equipo.
- La estación 6 (computación paralela) muestra las **mediciones reales** hechas en el PC del laboratorio (Intel i5-12600KF + RTX 2060) y lo indica en pantalla. En un equipo con los motores compilados (el PC del laboratorio) además permite repetir la medición en vivo.

## Guion de 5 minutos (el guion completo, con explicación del sistema, preguntas y cifras, está en `docs/guion_presentacion.md`)
| Min | Estación | Mensaje clave |
|---|---|---|
| 0:00 | Inicio | Pregunta al público: ¿cuánto vende una tienda de La Parada en diciembre? |
| 0:30 | 1 · El problema | La demanda en la frontera cambia por temporada, dólar y cierres. |
| 1:15 | 2 · Los datos | Cada fila = mes + comercio + categoría. Que el público elija una categoría. |
| 2:00 | 3 · Cómo aprende | **Pulsar «Entrenar en vivo»**: la línea naranja se acerca a la real. |
| 3:00 | 4 · ¿Acierta? | Probar con meses que nunca vio. Ser honestos: es un modelo simple. |
| 3:40 | 5 · ¿Qué pasará? | Escenario del público: «¿y si el dólar sube 20 %?». |
| 4:20 | 6 · Computación paralela | Repartir filas entre trabajadores; **GPU ≈5× más rápida, misma precisión**. |
| 5:00 | 7 · Conclusiones | Preguntas frecuentes desplegables. |

El **modo experto** (arriba a la derecha) sirve para preguntas técnicas: pipeline completo, datasets propios, constructos de OpenMP, etc.

## Antes de la feria (checklist)
- [ ] Ejecutar el sistema **una vez en el Mac con internet** y recorrer las 8 estaciones (así se instalan los paquetes y se comprueba que todo abre).
- [ ] Dejar el dataset en **100 mil** registros para la demo (rápido). Con **1 millón** el primer arranque tarda más (decenas de segundos en un portátil: se calculan las variables); mejor probarlo antes y avisar al público.
- [ ] Cargador conectado y modo «no suspender». Pantalla/navegador a pantalla completa (F11 / ⌃⌘F).
- [ ] Respaldo: llevar las capturas o una exportación en PDF por si falla el equipo (puedo preparar un video corto de la demo).

## Si algo falla
| Síntoma | Qué hacer |
|---|---|
| «No se encontró Python 3» | Instalar Python 3.11+ desde python.org y repetir. |
| No abre el navegador | Abrir manualmente <http://localhost:8501>. |
| Error al instalar paquetes | Falta internet la primera vez; conectar y reintentar. |
| Quedó una versión rota | Borrar la carpeta `.venv` y volver a abrir `iniciar_mac.command`. |
| Puerto ocupado | Cerrar otra sesión de Streamlit o usar `python -m streamlit run app/app.py --server.port 8502`. |

## Honestidad sobre los resultados (para responder preguntas)
- Los datos son **sintéticos** (el dólar es la serie oficial); no existe registro público de ventas de La Parada.
- El modelo (regresión lineal) **hoy no supera** a «copiar el mes pasado»; sirve como banco de pruebas del paralelismo. Para pronosticar el futuro se reentrena con todo el historial.
- Speedup medido con 1 millón de filas (referencia: C en 1 núcleo): GPU con datos residentes ≈4,85×; OpenMP ≈1,7×; MPI ≈1,55× (en CPU varía hasta ±30 % entre corridas).
- La precisión (MAE) es idéntica en todos los modos de ejecución.
