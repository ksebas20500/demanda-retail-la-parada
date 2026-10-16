"""Recorrido guiado (modo feria): 8 estaciones con visuales grandes, lenguaje simple y guion para el presentador."""
import re, time
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nucleo import *

ESTS = ["Inicio", "1 · El problema", "2 · Los datos", "3 · Cómo aprende", "4 · ¿Acierta?", "5 · ¿Qué pasará?", "6 · Computación paralela", "7 · Conclusiones"]
CAT_ES = {c: c.replace("_", " ").capitalize() for c in CATS}
CAT_ES.update({"aseo_hogar": "Aseo y hogar", "utiles_escolares": "Útiles escolares", "electronica": "Electrónica", "ferreteria": "Ferretería", "medicamentos": "Medicamentos"})
FONT = dict(size=15)


def _mov(d):
    i = ESTS.index(st.session_state["est"]); st.session_state["est"] = ESTS[min(max(i + d, 0), len(ESTS) - 1)]

def _cat_pills(key, titulo="Elige una categoría de producto"):
    actual = st.session_state.get("cat_val", "juguetes")
    sel = st.pills(titulo, CATS, default=actual, selection_mode="single", format_func=lambda c: CAT_ES[c], key=key)
    sel = sel or actual; st.session_state["cat_val"] = sel; return sel


def _html(t): return re.sub(r"\*\*(.+?)\*\*", lambda m: "<b>" + m.group(1) + "</b>", t)


def _grande(texto, sub=None):
    st.markdown(f"<div style='font-size:1.25rem;line-height:1.5'>{_html(texto)}</div>", unsafe_allow_html=True)
    if sub: st.caption(sub)


def render():
    if "est" not in st.session_state: st.session_state["est"] = ESTS[0]
    tam = st.session_state.get("tam_val", "100k")
    with st.spinner("Preparando el sistema (datos, variables y modelo)…"):
        D = asegurar_modelo(tam)
    M = st.session_state[f"modelo_{tam}"]; v = ventas(tam, mtime(ruta_datos(tam)))
    i = ESTS.index(st.session_state["est"])
    st.progress(i / (len(ESTS) - 1))
    st.radio("Estación", ESTS, key="est", horizontal=True, label_visibility="collapsed")

    # ============================================================ 0 · INICIO
    if i == 0:
        st.markdown("# ¿Cuánto venderán los comercios de La Parada el próximo mes?")
        _grande("Un sistema inteligente que **aprende de años de ventas** para predecir la demanda, y que usa "
                "**computación paralela** (varios núcleos y la tarjeta gráfica) para entrenarse más rápido.")
        inf = info_dataset(tam); c = st.columns(4)
        c[0].metric("Registros de ventas", f"{inf['filas']:,}"); c[1].metric("Comercios", f"{inf['comercios']}"); c[2].metric("Categorías de producto", "12"); c[3].metric("Años de historia", "11")
        st.markdown("&nbsp;")
        st.button("Comenzar el recorrido ▶", type="primary", on_click=_mov, args=(1,), key="go0")

    # ============================================================ 1 · EL PROBLEMA
    elif i == 1:
        st.markdown("## 1 · El problema: la demanda en la frontera es muy cambiante")
        _grande("Las ventas suben y bajan por la **temporada** (diciembre, enero escolar), por el **precio del dólar** y por los **cierres de la frontera**.")
        a, b = st.columns(2); ver_trm = a.toggle("Mostrar el precio del dólar (TRM)", value=True); ver_ev = b.toggle("Marcar cierres y eventos de la frontera", value=True)
        agg = v.groupby("fecha").agg(dem=("demanda", "sum"), trm=("trm", "first"), ev=("evento_fronterizo", "first")); tipos = v.drop_duplicates("fecha").set_index("fecha").tipo_evento.to_dict()
        f = make_subplots(specs=[[{"secondary_y": True}]])
        if ver_ev: franjas_evento(f, agg, True, tipos)
        f.add_trace(go.Scatter(x=agg.index, y=agg.dem, name="Unidades vendidas por mes", line=dict(color=AZUL, width=3)), secondary_y=False)
        if ver_trm: f.add_trace(go.Scatter(x=agg.index, y=agg.trm, name="Dólar (COP por USD)", line=dict(color=NARANJA, width=2)), secondary_y=True)
        f.update_layout(template=TPL, height=470, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.08), font=FONT)
        f.update_yaxes(title_text="Unidades vendidas", secondary_y=False); f.update_yaxes(title_text="COP por USD", secondary_y=True, showgrid=False); st.plotly_chart(f, width="stretch")
        sin = v[v.evento_fronterizo == 0].assign(mes=lambda x: x.fecha.dt.month).groupby("mes").demanda.sum(); dic = sin.loc[12] / sin.mean() - 1
        c = st.columns(3); c[0].metric("Diciembre vs un mes normal", f"{dic*100:+.0f} %"); c[1].metric("Meses con evento en la frontera", f"{int((agg.ev == 1).sum())} de {len(agg)}")
        c[2].metric("Dólar: mínimo → máximo", f"{agg.trm.min():,.0f} → {agg.trm.max():,.0f}")

    # ============================================================ 2 · LOS DATOS
    elif i == 2:
        st.markdown("## 2 · Los datos: cada fila es un mes, un comercio y una categoría")
        _grande("Los datos son **simulados con reglas reales** (temporada, dólar oficial, eventos de frontera). Elige cuántos usar:")
        nuevo = st.select_slider("¿Cuántos registros usamos?", ["10k", "100k", "1M"], value=tam, format_func=lambda x: {"10k": "10 mil", "100k": "100 mil", "1M": "1 millón"}[x])
        if nuevo != tam: st.session_state["tam_val"] = nuevo; st.rerun()
        inf = info_dataset(tam); rc = recursos(inf); c = st.columns(4)
        c[0].metric("Registros", f"{inf['filas']:,}"); c[1].metric("Para entrenar", f"{inf['ntr']:,}"); c[2].metric("Memoria de las variables", f"{rc['X_mb']:.0f} MB"); c[3].metric("Comercios", inf["comercios"])
        st.caption("Más datos = más trabajo para entrenar. En la estación 6 veremos cómo la computación paralela lo compensa. En un computador portátil, 1 millón puede tardar unos segundos extra en prepararse.")
        cat = _cat_pills("p2")
        h = v[(v.evento_fronterizo == 0) & (v.categoria == cat)].assign(mes=lambda x: x.fecha.dt.month).groupby("mes").demanda.mean(); idx = h / h.mean()
        col = [NARANJA if m == idx.idxmax() else AZUL for m in idx.index]
        f = fig_base(380, title=f"¿Cuándo se vende más «{CAT_ES[cat]}»? (1.0 = un mes promedio)", font=FONT); f.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        f.add_trace(go.Bar(x=[MESES[m - 1] for m in idx.index], y=idx.values, marker_color=col, text=[f"{x:.2f}×" for x in idx.values], textposition="outside"))
        f.add_hline(y=1, line_dash="dash", line_color=GRIS); f.update_yaxes(range=[0, float(idx.max()) * 1.25]); st.plotly_chart(f, width="stretch")
        st.success(f"**{CAT_ES[cat]}** vende más en **{MESES[idx.idxmax() - 1]}** ({idx.max():.2f} veces un mes normal) y menos en **{MESES[idx.idxmin() - 1]}** ({idx.min():.2f}×).")
        st.markdown("**Así se ve una parte de los datos:**")
        ej = v[(v.comercio_id == 0) & (v.categoria == cat)].tail(6)[["fecha", "comercio_id", "zona", "categoria", "precio_cop", "trm", "evento_fronterizo", "demanda"]].copy()
        ej["fecha"] = ej.fecha.dt.strftime("%b %Y"); ej["categoria"] = ej.categoria.map(CAT_ES); ej["evento_fronterizo"] = ej.evento_fronterizo.map({0: "No", 1: "Sí"})
        ej.columns = ["Mes", "Comercio", "Zona", "Categoría", "Precio (COP)", "Dólar", "¿Evento frontera?", "Unidades vendidas"]; st.dataframe(ej, hide_index=True, width="stretch")

    # ============================================================ 3 · CÓMO APRENDE
    elif i == 3:
        st.markdown("## 3 · Cómo aprende el modelo")
        _grande("Empieza **sin saber nada** (predice cero) y en cada *época* compara su predicción con lo que realmente se vendió y **corrige sus errores**. Mira cómo la línea naranja se acerca a la real:")
        a, b = st.columns([3, 1]); cat = _cat_pills("p3", "Categoría a observar")
        vel = b.pills("Velocidad", ["Lenta", "Normal", "Rápida"], default="Normal", selection_mode="single", key="vel") or "Normal"
        pausa = {"Lenta": 0.35, "Normal": 0.15, "Rápida": 0.04}[vel]
        idx = serie_df(D, 0, cat); idx = idx[D["split"][idx] >= 1]; fechas = D["df"].fecha.iloc[idx]; real = D["d"][idx]; Xs = D["X"][idx]
        lanzar = st.button("▶ Entrenar el modelo en vivo", type="primary", key="entrenar")
        cab = st.empty(); g1, g2 = st.columns([3, 2]); ch1, ch2 = g1.empty(), g2.empty()
        def dibujar(e, w, hist, final=False):
            pred = np.expm1(np.clip(Xs @ w, 0, None)); f = fig_base(420, title=f"{CAT_ES[cat]} · comercio 0: lo real y lo que predice el modelo", font=FONT)
            f.add_trace(go.Scatter(x=fechas, y=real, name="Lo que se vendió", line=dict(color=AZUL, width=3))); f.add_trace(go.Scatter(x=fechas, y=pred, name="Predicción del modelo", line=dict(color=NARANJA, width=3)))
            f.update_yaxes(range=[0, float(real.max()) * 1.4], title="Unidades por mes"); ch1.plotly_chart(f, width="stretch")
            h = pd.DataFrame(hist, columns=["e", "tr", "va"]); f2 = fig_base(420, title="Error del modelo (menor = mejor)", font=FONT)
            f2.add_trace(go.Scatter(x=h.e, y=h.tr, name="Con datos de entrenamiento", line=dict(color=AZUL, width=3))); f2.add_trace(go.Scatter(x=h.e, y=h.va, name="Con datos nuevos", line=dict(color=VERDE, width=3, dash="dot")))
            f2.update_yaxes(type="log", title="Error (escala log)"); f2.update_xaxes(range=[0, EPOCHS_DEF], title="Época"); ch2.plotly_chart(f2, width="stretch")
            cab.markdown(f"### Época {e + 1} de {EPOCHS_DEF}" + ("  ✅ ¡Entrenamiento terminado!" if final else ""))
        mostrar = set(list(range(0, 30, 3)) + list(range(30, 120, 10)) + list(range(120, 300, 15)) + [EPOCHS_DEF - 1])
        if lanzar:
            def cb(e, w, hist):
                if e in mostrar: dibujar(e, w, hist, e == EPOCHS_DEF - 1); time.sleep(pausa)
            w, t, n = entrenar_numpy(D, EPOCHS_DEF, LR_DEF, 1.0, cb); mae, rmse = guardar_modelo(tam, w, "Python (numpy, 1 hilo)", t, D, epocas=EPOCHS_DEF, frac=1.0, n=n)
            M = st.session_state[f"modelo_{tam}"]; c = st.columns(3)
            c[0].metric("Registros usados para aprender", f"{n:,}"); c[1].metric("Error promedio en datos nuevos", f"{mae:.0f} unidades/mes"); c[2].metric("Tiempo de cálculo (sin pausas de animación)", "ver estación 6")
            s = pd.Series(w, index=D["cols"]).drop("bias"); top = s.abs().sort_values(ascending=False).head(8).index; s = s[top][::-1]
            fb = fig_base(380, title="¿Qué aprendió? Las pistas que más pesan para predecir", font=FONT); fb.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            fb.add_trace(go.Bar(x=s.values, y=[nombre_var(c_) for c_ in s.index], orientation="h", marker_color=[VERDE if x >= 0 else ROJO for x in s.values])); st.plotly_chart(fb, width="stretch")
            st.caption("Verde: sube la demanda prevista · Rojo: la baja. La demanda de meses pasados es la mejor pista.")
        else:
            dibujar(-1, np.zeros(len(D["cols"]), np.float32), [(0, 1.0, 1.0)], False); cab.markdown("### Pulsa «Entrenar el modelo en vivo»")

    # ============================================================ 4 · ¿ACIERTA?
    elif i == 4:
        st.markdown("## 4 · ¿Acierta? Probemos con meses que el modelo nunca vio")
        _grande("El modelo se entrenó con datos hasta 2022. Aquí se compara con lo que **realmente pasó** entre 2023 y 2025.")
        cat = _cat_pills("p4"); n_com = info_dataset(tam)["comercios"]
        com = st.slider("Comercio", 0, n_com - 1, 0, help="Cada comercio tiene su propio nivel de ventas")
        ix = serie_df(D, com, cat); ix = ix[D["split"][ix] >= 1]; s = D["df"].iloc[ix]; pred = np.expm1(np.clip(D["X"][ix] @ M["w"], 0, None))
        ing = D["df"].iloc[serie_df(D, com, cat)].demanda.shift(1).to_numpy()[-len(ix):]; prueba = (D["split"][ix] == 2)
        f = fig_base(470, title=f"{CAT_ES[cat]} · comercio {com}", font=FONT)
        f.add_vrect(x0="2024-07-01", x1="2026-01-01", fillcolor=VERDE, opacity=0.10, line_width=0, annotation_text="meses de prueba (nunca vistos)", annotation_position="top left")
        f.add_trace(go.Scatter(x=s.fecha, y=s.demanda, name="Lo que se vendió", line=dict(color=AZUL, width=3), mode="lines+markers"))
        f.add_trace(go.Scatter(x=s.fecha, y=pred, name="Lo que predijo el modelo", line=dict(color=NARANJA, width=3), mode="lines+markers"))
        f.add_trace(go.Scatter(x=s.fecha, y=ing, name="«Copiar el mes pasado»", line=dict(color=GRIS, dash="dash"))); f.update_layout(hovermode="x unified"); st.plotly_chart(f, width="stretch")
        err = float(np.abs(pred[prueba] - s.demanda.to_numpy()[prueba]).mean()); media = float(s.demanda.to_numpy()[prueba].mean()); err_ing = float(np.nanmean(np.abs(ing[prueba] - s.demanda.to_numpy()[prueba])))
        glob_m = M["mae"]; glob_i = mae_ingenuo_global(tam, D); c = st.columns(3)
        c[0].metric("Error del modelo en esta serie", f"{err:.0f} unidades/mes", f"{100*err/media:.0f} % de la venta media", delta_color="off")
        c[1].metric("Copiar el mes pasado", f"{err_ing:.0f} unidades/mes"); c[2].metric("Error medio en TODAS las series", f"{glob_m:.0f} u. (modelo) vs {glob_i:.0f} u. (copiar)")
        if glob_m < glob_i: st.success("En promedio el modelo acierta **mejor** que simplemente copiar el mes pasado.")
        else: st.info("Con honestidad: **copiar el mes pasado acierta un poco más** que este modelo, que es simple a propósito. El objetivo del proyecto es medir cuánto acelera la computación paralela su entrenamiento; mejorar la precisión es el siguiente paso.")

    # ============================================================ 5 · ¿QUÉ PASARÁ?
    elif i == 5:
        st.markdown("## 5 · ¿Qué pasará? Pronóstico con escenarios")
        _grande("Juega con el futuro: **¿qué pasa con las ventas si sube el dólar o si cierran la frontera?** (Los escenarios son supuestos tuyos).")
        c = st.columns(4); h = c[0].slider("Meses a pronosticar", 3, 12, 6); dtrm = c[1].slider("Cambio del dólar (%)", -20, 30, 0); dpre = c[2].slider("Cambio de precios (%)", -20, 20, 0); ev = c[3].toggle("Evento en la frontera", value=False)
        opcion = st.pills("¿De qué quieres el pronóstico?", ["Todo"] + CATS, default="Todo", selection_mode="single", format_func=lambda c_: "Todo el sistema" if c_ == "Todo" else CAT_ES[c_], key="p5") or "Todo"
        R = pronosticar(tam, D, modelo_pronostico(tam, D), h, dtrm, dpre, ev); P, real, catv = R["P"], R["real"], R["catv"]; m_ = np.ones(len(catv), bool) if opcion == "Todo" else catv == opcion
        real_s, pred_s = real[m_].sum(0), P[m_].sum(0); fh = list(R["hist_fechas"][-36:]); yh = real_s[-36:]; fut = R["fechas"]; prev = real_s[-12:][:h]
        f = fig_base(470, title=f"Pronóstico · {'todo el sistema' if opcion == 'Todo' else CAT_ES[opcion]}", font=FONT)
        f.add_trace(go.Scatter(x=fh, y=yh, name="Historia real", line=dict(color=AZUL, width=3)))
        f.add_trace(go.Scatter(x=fut, y=prev, name="Mismo período del año pasado", line=dict(color=GRIS, width=2, dash="dot")))
        f.add_trace(go.Scatter(x=[fh[-1]] + fut, y=[yh[-1]] + list(pred_s), name="Pronóstico", line=dict(color=NARANJA, width=4, dash="dash"), mode="lines+markers")); f.update_layout(hovermode="x unified"); st.plotly_chart(f, width="stretch")
        tot, tot_prev = float(pred_s.sum()), float(prev.sum()); k = st.columns(3)
        k[0].metric(f"Unidades pronosticadas ({h} meses)", f"{tot:,.0f}"); k[1].metric("Mismo período del año pasado", f"{tot_prev:,.0f}"); k[2].metric("Diferencia", f"{100*(tot/tot_prev-1):+.1f} %")
        st.caption("Para pronosticar el futuro, el modelo se **reentrena con todo el historial (2016–2025)**: es lo habitual después de validarlo con meses nunca vistos (estación 4). Es una regresión lineal básica; los escenarios son supuestos, no predicciones del dólar ni de la frontera.")

    # ============================================================ 6 · COMPUTACIÓN PARALELA
    elif i == 6:
        st.markdown("## 6 · ¿Por qué computación paralela?")
        _grande("Entrenar significa recorrer **todas las filas, 300 veces**. Con 1 millón de filas son cientos de millones de cuentas. **Paralelizar es repartir ese trabajo entre varios «trabajadores»** que lo hacen a la vez.")
        inf = info_dataset(tam); ntr = max(inf["ntr"], 1)
        st.markdown("#### La idea: repartir las filas")
        nw = st.slider("¿Cuántos trabajadores (núcleos o procesos)?", 1, 16, 4)
        f = fig_base(230, title=f"{ntr:,} filas de entrenamiento repartidas entre {nw} trabajador{'es' if nw > 1 else ''}", font=FONT); f.update_layout(barmode="stack", margin=dict(l=10, r=10, t=60, b=10), showlegend=False)
        paleta = ["#1f4e79", "#e08a00", "#2e8b57", "#8e44ad", "#d9534f", "#17a2b8", "#7f8c8d", "#b8860b"]
        for k in range(nw): f.add_trace(go.Bar(y=[""], x=[ntr / nw], orientation="h", marker_color=paleta[k % len(paleta)], text=f"T{k+1}", textposition="inside", hovertemplate=f"Trabajador {k+1}: {ntr // nw:,} filas<extra></extra>"))
        f.update_xaxes(visible=False); st.plotly_chart(f, width="stretch")
        st.markdown(f"Cada trabajador procesa **{ntr // nw:,} filas**, calcula su *parte del error* y al final todos **suman** sus resultados (28 números). En teoría el tiempo se divide por {nw}; en la práctica depende del hardware, y eso es lo que medimos:")
        R = resultados()
        if R is None: st.warning("No hay mediciones guardadas (logs/fase5_resumen.csv)."); st.stop()
        ds = st.radio("Tamaño de datos medido", ["10k", "100k", "1M"], index=2, horizontal=True, format_func=lambda x: {"10k": "10 mil", "100k": "100 mil", "1M": "1 millón"}[x]); S = R[R.dataset == ds]
        def pick(df_, rotulo):
            r = df_.sort_values("t_media").iloc[0]; return dict(modo=rotulo.format(r.config.split()[0] if r.config.split()[0].isdigit() else ""), t=float(r.t_media), sp=float(r["speedup_vs_C_seq(media)"]), tipo=r.modo)
        cands = [pick(S[S.modo.str.startswith("Python")], "Python (1 núcleo)"), pick(S[S.modo == "C secuencial (gcc)"], "Programa en C (1 núcleo)"), pick(S[S.modo == "OpenMP (gcc)"], "OpenMP ({} hilos)"),
                 pick(S[S.modo.str.startswith("MPI")], "MPI ({} procesos)"), pick(S[(S.modo.str.startswith("CUDA")) & (S.config == "todo en VRAM")], "Tarjeta gráfica (datos en la GPU)"),
                 pick(S[(S.modo.str.startswith("CUDA")) & (S.config != "todo en VRAM")], "Tarjeta gráfica (datos por lotes)")]
        T = pd.DataFrame(cands); col = [GRIS, AZUL, "#3b78b0", NARANJA, VERDE, "#7fbf9a"]
        f = fig_base(400, title="Cuántas veces más rápido que el programa en C de 1 núcleo (1× = referencia)", font=FONT); f.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        f.add_trace(go.Bar(x=T.sp, y=T.modo, orientation="h", marker_color=col, text=[f"{a:.2f}×  ({b:.2f} s)" for a, b in zip(T.sp, T.t)], textposition="outside")); f.add_vline(x=1, line_dash="dash", line_color=GRIS)
        f.update_yaxes(autorange="reversed"); f.update_xaxes(range=[0, max(5.8, float(T.sp.max()) * 1.35)]); st.plotly_chart(f, width="stretch")
        maes = S.MAE.round(1).unique()
        st.success(f"**La precisión es idéntica** en todos los modos (error promedio {maes[0]:.1f} unidades): paralelizar cambia el **tiempo**, no la respuesta." if len(maes) == 1 else "Las precisiones coinciden en todos los modos (diferencias < 0,1 unidades).")
        st.caption("Medido en el computador del laboratorio (Intel i5-12600KF de 10 núcleos + tarjeta NVIDIA RTX 2060 de 6 GB). Los tiempos de CPU varían hasta ±30 % entre corridas; los de la GPU son muy estables.")
        if cuda_ok():
            if st.button("▶ Repetir la medición ahora en este computador (secuencial vs OpenMP vs GPU)", key="live"):
                with st.spinner("Midiendo en vivo…"):
                    res = []
                    for m_, h_ in [("C secuencial", 1), ("OpenMP", 6), ("CUDA (todo en VRAM)", 1)]:
                        _, t_, mae_, _, _ = entrenar_exe(m_, tam, inf["p"], h_, 65536); res.append((m_ + (f" ({h_} hilos)" if m_ == "OpenMP" else ""), t_, mae_))
                ref = res[0][1]; kk = st.columns(3)
                for c_, (nm, t_, mae_) in zip(kk, res): c_.metric(nm, f"{t_:.3f} s", f"{ref/t_:.2f}× · error {mae_:.1f}", delta_color="off")
        else: st.info("Este computador no tiene tarjeta NVIDIA/CUDA: aquí mostramos las mediciones reales del laboratorio. Todo lo demás del recorrido se calcula en vivo.")
        with st.expander("¿Por qué la tarjeta gráfica solo gana si los datos están dentro de ella?"):
            cu = S[S.modo.str.startswith("CUDA")]
            if len(cu) >= 2:
                a_ = cu[cu.config == "todo en VRAM"].iloc[0]; b_ = cu[cu.config != "todo en VRAM"].sort_values("t_media").iloc[0]
                st.markdown(f"La GPU calcula muy rápido, pero **los datos viajan por un cable (bus PCIe)** desde la memoria del computador. Con los datos ya dentro de la GPU entrena en **{a_.t_media:.2f} s**; si hay que enviarlos por lotes en cada época tarda **{b_.t_media:.2f} s**: el viaje se come la ganancia.")

    # ============================================================ 7 · CONCLUSIONES
    else:
        st.markdown("## 7 · Conclusiones")
        R = resultados(); S = R[R.dataset == "1M"] if R is not None else None
        pts = ["**El sistema aprende** de la historia de ventas y pronostica la demanda por categoría y comercio, con escenarios de dólar y frontera."]
        if S is not None and len(S):
            cu = S[S.config == "todo en VRAM"]["speedup_vs_C_seq(media)"].max(); om = S[S.modo == "OpenMP (gcc)"]["speedup_vs_C_seq(media)"].max(); mp = S[S.modo.str.startswith("MPI")]["speedup_vs_C_seq(media)"].max()
            pts += [f"**La precisión no cambia al paralelizar:** el mismo error en los cinco modos (con 1 millón de filas: {S.MAE.round(1).iloc[0]:.1f} unidades).",
                    f"**La tarjeta gráfica entrenó {cu:.1f} veces más rápido** que un solo núcleo, siempre que los datos quepan en su memoria.",
                    f"**En CPU la ganancia es modesta:** hasta {om:.1f}× con OpenMP y {mp:.1f}× con MPI. Más núcleos no siempre acelera: la memoria se convierte en el cuello de botella."]
        pts += ["**Lo que sigue:** un modelo más preciso, probar MPI en varios computadores y datasets de decenas de millones de registros."]
        for k, t in enumerate(pts, 1): st.markdown(f"<div style='font-size:1.3rem;margin:.6rem 0'><b>{k}.</b> {_html(t)}</div>", unsafe_allow_html=True)
        st.warning("**Honestidad científica:** los datos son sintéticos (el dólar sí es real); el modelo es simple y hoy no supera a «copiar el mes pasado»; los tiempos de CPU varían entre corridas.")
        with st.expander("❓ Preguntas frecuentes del público"):
            st.markdown("""
- **¿Los datos son reales?** No: son simulados con reglas realistas (temporada, eventos, dólar). El dólar es la serie oficial. No existe un registro público de ventas de La Parada.
- **¿Por qué un modelo tan sencillo?** Para tener un cálculo claro y repetible con el que medir el efecto del paralelismo. Mejorar el modelo es el siguiente paso.
- **¿Paralelizar cambia el resultado?** No: el error es el mismo en todos los modos.
- **¿Qué diferencia hay entre OpenMP, MPI y CUDA?** OpenMP: hilos que comparten memoria. MPI: procesos que se envían mensajes (pueden estar en computadores distintos). CUDA: miles de hilos en la tarjeta gráfica.
- **¿Por qué más núcleos no aceleran proporcionalmente?** Porque el cálculo recorre muchos datos y la memoria no da abasto (hipótesis coherente con las mediciones).
- **¿Podría entrenar con más de un millón de registros?** Sí con cambios menores; en este equipo el límite práctico estimado es de unos 10 millones (sin medir). La GPU de 6 GB admitiría ~46 millones en teoría.
- **¿Funciona sin GPU?** Sí: todo el recorrido se ejecuta en cualquier computador; la GPU solo se usó para las mediciones de velocidad.
""")

    # ------------------------------------------------------------ navegación
    st.divider(); b1, _, b2 = st.columns([2, 5, 2])
    b1.button("◀ Anterior", on_click=_mov, args=(-1,), disabled=i == 0, key="prev"); b2.button("Siguiente ▶", on_click=_mov, args=(1,), disabled=i == len(ESTS) - 1, type="primary", key="next")
