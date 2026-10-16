"""Modo experto: todas las herramientas del sistema (pipeline, datos, entrenamiento, predicciones, pronostico, rendimiento)."""
import time
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nucleo import *

PAGINAS = ["Sistema completo", "Datos", "Entrenamiento en vivo", "Predicciones", "Pronostico", "Rendimiento paralelo", "Laboratorio paralelo"]


def render():
    st.sidebar.title("📈 Demanda La Parada")
    modo_tam = st.sidebar.radio("Tamano del dataset", ["10k", "100k", "1M", "Personalizado"], index=1, horizontal=True)
    if modo_tam == "Personalizado": tam = f"c{st.sidebar.slider('Comercios (x1584 filas c/u)', 3, 631, 20)}"
    else: tam = modo_tam
    f_ = filas_de(tam); st.sidebar.caption(f"**{f_:,} filas** · {f_ // 1584} comercios × 12 categorias × 132 meses")
    estado = st.sidebar.empty()

    def render_estado():
        m_ = st.session_state.get(f"modelo_{tam}")
        with estado.container():
            st.markdown("**Estado del sistema**")
            st.markdown(("✅" if tiene_datos(tam) else "⬜") + " Datos generados")
            st.markdown(("✅" if tiene_feats(tam) else "⬜") + " Variables construidas")
            st.markdown(("✅" if m_ else "⬜") + " Modelo entrenado" + (f" · MAE {m_['mae']:.1f}" if m_ else ""))
    render_estado(); st.sidebar.divider()
    pagina = st.sidebar.radio("Ir a", PAGINAS, label_visibility="collapsed")
    M = st.session_state.get(f"modelo_{tam}")

    def requiere(modelo=False):
        if not tiene_datos(tam):
            st.warning(f"El dataset **{tam}** aun no existe. Ve a **Sistema completo** y pulsa *Ejecutar todo el sistema*."); st.stop()
        if modelo and not M:
            st.warning(f"Entrena un modelo para **{tam}** en **Sistema completo** o **Entrenamiento en vivo**."); st.stop()

    # =========================================================== SISTEMA COMPLETO
    if pagina == "Sistema completo":
        st.title("Sistema completo de prediccion")
        st.caption(f"Dataset seleccionado: **{tam}** · {f_:,} filas. Puedes ejecutar cada paso o todo de una vez.")
        motores = motores_disponibles()
        cfg = st.columns([2, 1, 1, 1])
        motor = cfg[0].selectbox("Motor de entrenamiento", motores, help="Python permite ver el progreso y usar una fraccion de datos; los demas usan los ejecutables compilados (solo PC del laboratorio).")
        hilos = cfg[1].select_slider("Hilos/procesos", [1, 2, 4, 6, 8, 10, 12, 16], value=6, disabled=motor not in ("OpenMP", "MPI"))
        lote = cfg[2].select_slider("Lote CUDA (filas)", [4096, 16384, 65536, 262144], value=65536, disabled=motor != "CUDA (por lotes)")
        frac = cfg[3].slider("% de entrenamiento", 10, 100, 100, 10, disabled=not motor.startswith("Python")) / 100
        if motor.startswith("Python"): epocas = st.slider("Epocas", 50, 600, EPOCHS_DEF, 50)
        else: epocas = EPOCHS_DEF; st.caption("Los motores compilados usan 300 epocas, lr 0,05 (iguales a Python).")
        if len(motores) == 1: st.info("Este equipo no tiene los motores compilados (C/OpenMP/MPI/CUDA): se usa Python. Los resultados medidos con esos motores estan en **Rendimiento paralelo**.")

        def paso_datos(log):
            if tiene_datos(tam): log("Dataset ya existente: se reutiliza."); return 0.0
            t = asegurar_datos(tam); log(f"{f_:,} filas generadas con la TRM oficial."); return t

        def paso_features(log):
            if not tiene_datos(tam): raise RuntimeError("Primero genera los datos.")
            t0 = time.perf_counter(); D = get_D(tam); inf = info_dataset(tam)
            log(f"{inf['ntr']:,} train · {inf['nva']:,} val · {inf['nte']:,} test · {inf['p']} variables (con sesgo)."); return time.perf_counter() - t0

        def paso_entrenar(log):
            if not tiene_datos(tam): raise RuntimeError("Primero genera los datos.")
            D = get_D(tam)
            if motor.startswith("Python"):
                w, t, n = entrenar_numpy(D, epocas, LR_DEF, frac); mae, rmse = guardar_modelo(tam, w, motor, t, D, epocas=epocas, frac=frac, n=n)
                log(f"{n:,} filas · {epocas} epocas · {t:.2f} s · MAE {mae:.1f} · RMSE {rmse:.1f}"); return t
            inf = info_dataset(tam); w, t, _, _, _ = entrenar_exe(motor, tam, inf["p"], hilos, lote)
            if w is None: raise RuntimeError("El ejecutable no devolvio pesos.")
            mae, rmse = guardar_modelo(tam, w, motor + (f" ({hilos})" if motor in ("OpenMP", "MPI") else ""), t, D, epocas=EPOCHS_DEF, frac=1.0, n=inf["ntr"])
            log(f"{inf['ntr']:,} filas · {t:.3f} s · MAE {mae:.1f} · RMSE {rmse:.1f}"); return t

        def paso_evaluar(log):
            Mm = st.session_state.get(f"modelo_{tam}")
            if not Mm: raise RuntimeError("Primero entrena el modelo.")
            log(f"MAE modelo {Mm['mae']:.1f} vs ingenuo (mes anterior, test) {mae_ingenuo_global(tam, get_D(tam)):.1f}"); return 0.0

        pasos = [("1 · Generar datos", paso_datos), ("2 · Construir variables", paso_features), ("3 · Entrenar", paso_entrenar), ("4 · Evaluar", paso_evaluar)]
        ejecutar_todo = st.button("▶ Ejecutar todo el sistema", type="primary")
        cols_btn = st.columns(4); individual = None
        for i, (nombre, _) in enumerate(pasos):
            if cols_btn[i].button(nombre, key=f"b{i}"): individual = i
        if ejecutar_todo or individual is not None:
            for nombre, fn in (pasos if ejecutar_todo else [pasos[individual]]):
                with st.status(nombre, expanded=True) as s:
                    try:
                        t = fn(lambda msg: st.write(msg)); s.update(label=f"{nombre} · {t:.2f} s" if t else nombre, state="complete")
                    except Exception as ex:
                        s.update(label=f"{nombre} · ERROR", state="error"); st.error(str(ex)); break
            render_estado()
        M = st.session_state.get(f"modelo_{tam}")
        st.divider(); st.subheader("Uso del dataset y de los recursos")
        if tiene_datos(tam):
            inf = info_dataset(tam); rc = recursos(inf); c = st.columns(5)
            c[0].metric("Train / Val / Test", f"{inf['ntr']:,} / {inf['nva']:,} / {inf['nte']:,}"); c[1].metric("Matriz de variables (RAM)", f"{rc['X_mb']:.1f} MB")
            c[2].metric("VRAM necesaria (todo en GPU)", f"{rc['vram_mib']:.0f} MiB"); c[3].metric("% de los 6 GB (RTX 2060)", f"{rc['pct']:.2f} %"); c[4].metric("Filas maximas en 6 GB (teorico)", f"{rc['max_filas']/1e6:.0f} M")
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=rc["pct"], number={"suffix": " %"}, title={"text": "VRAM de la RTX 2060 que ocuparia el entrenamiento"},
                                           gauge={"axis": {"range": [0, 100]}, "bar": {"color": AZUL}, "steps": [{"range": [0, 70], "color": "#e8f1ea"}, {"range": [70, 100], "color": "#f7e3e1"}]}))
            gauge.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10), template=TPL); st.plotly_chart(gauge, width="stretch")
        else: st.info("Aun no hay datos para este dataset.")
        if M:
            st.subheader("Resultado del ultimo entrenamiento")
            c = st.columns(4); c[0].metric("Motor", M["motor"]); c[1].metric("Tiempo", f"{M['t']:.3f} s"); c[2].metric("MAE (test)", f"{M['mae']:.1f} u."); c[3].metric("RMSE (test)", f"{M['rmse']:.1f} u.")
            st.success("Modelo listo. Explora **Predicciones** y **Pronostico** en el menu lateral.")

    # =========================================================== DATOS
    elif pagina == "Datos":
        requiere(); v = ventas(tam, mtime(ruta_datos(tam)))
        st.title("Exploracion de datos")
        c = st.columns(5); c[0].metric("Filas", f"{len(v):,}"); c[1].metric("Comercios", v.comercio_id.nunique()); c[2].metric("Categorias", v.categoria.nunique())
        c[3].metric("Meses", v.fecha.nunique()); c[4].metric("Demanda media", f"{v.demanda.mean():.0f} u.")
        f1, f2 = st.columns(2)
        cats = f1.multiselect("Categorias", sorted(v.categoria.unique()), default=sorted(v.categoria.unique())); anios = f2.slider("Periodo", 2015, 2025, (2015, 2025))
        vv = v[v.categoria.isin(cats) & v.fecha.dt.year.between(*anios)]
        t1, t2, t3, t4 = st.tabs(["Serie agregada", "Estacionalidad", "Un comercio", "Distribucion"])
        with t1:
            agg = vv.groupby("fecha").agg(dem=("demanda", "sum"), trm=("trm", "first"), ev=("evento_fronterizo", "first")); tipos = vv.drop_duplicates("fecha").set_index("fecha").tipo_evento.to_dict()
            f = make_subplots(specs=[[{"secondary_y": True}]]); franjas_evento(f, agg, True, tipos)
            f.add_trace(go.Scatter(x=agg.index, y=agg.dem, name="Demanda total", line=dict(color=AZUL, width=2)), secondary_y=False)
            f.add_trace(go.Scatter(x=agg.index, y=agg.trm, name="TRM (COP/USD)", line=dict(color=NARANJA, width=1.5)), secondary_y=True)
            f.update_layout(template=TPL, height=430, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1), title="Demanda mensual total · sombreado rojo = evento fronterizo")
            f.update_yaxes(title_text="Unidades", secondary_y=False); f.update_yaxes(title_text="COP/USD", secondary_y=True); st.plotly_chart(f, width="stretch")
        with t2:
            h = vv[vv.evento_fronterizo == 0].assign(mes=lambda x: x.fecha.dt.month).groupby(["categoria", "mes"]).demanda.mean().unstack(); h = h.div(h.mean(axis=1), axis=0)
            f = go.Figure(go.Heatmap(z=h.values, x=[MESES[m - 1] for m in h.columns], y=h.index, colorscale="YlOrRd", zmin=0.6, zmax=2.0, colorbar=dict(title="Indice"), hovertemplate="%{y} · %{x}: %{z:.2f}x<extra></extra>"))
            f.update_layout(template=TPL, height=440, title="Indice estacional (1.0 = promedio anual, meses sin evento)", margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(f, width="stretch")
        with t3:
            a, b = st.columns(2); cat = a.selectbox("Categoria", sorted(v.categoria.unique()), index=7); com = b.selectbox("Comercio", sorted(v.comercio_id.unique()))
            s = v[(v.categoria == cat) & (v.comercio_id == com)].set_index("fecha"); ev = s.evento_fronterizo.to_frame("ev"); f = fig_base(400, title=f"{cat} · comercio {com} ({s.zona.iloc[0]})"); franjas_evento(f, ev)
            f.add_trace(go.Scatter(x=s.index, y=s.demanda, name="Demanda", line=dict(color=AZUL), hovertemplate="%{x|%b %Y}: %{y} u.<extra></extra>"))
            f.add_trace(go.Scatter(x=s.index, y=s.precio_cop, name="Precio (COP)", yaxis="y2", line=dict(color=GRIS, dash="dot"))); f.update_layout(yaxis2=dict(overlaying="y", side="right", title="COP", showgrid=False)); st.plotly_chart(f, width="stretch")
        with t4:
            a, b = st.columns(2)
            f = fig_base(340, title="Distribucion de la demanda"); f.add_trace(go.Histogram(x=vv.demanda, nbinsx=80, marker_color=AZUL)); a.plotly_chart(f, width="stretch")
            f = fig_base(340, title="Demanda media por categoria"); m = vv.groupby("categoria").demanda.mean().sort_values(); f.add_trace(go.Bar(x=m.values, y=m.index, orientation="h", marker_color=VERDE)); b.plotly_chart(f, width="stretch")

    # =========================================================== ENTRENAMIENTO EN VIVO
    elif pagina == "Entrenamiento en vivo":
        requiere(); D = get_D(tam)
        st.title("Entrenamiento en vivo")
        st.caption("Regresion lineal por descenso de gradiente (lote completo). Cada epoca recorre todas las filas de entrenamiento.")
        c = st.columns(5)
        epocas = c[0].slider("Epocas", 10, 600, EPOCHS_DEF, 10); lr = c[1].select_slider("Tasa de aprendizaje", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], value=LR_DEF)
        frac = c[2].slider("% del dataset de entrenamiento", 5, 100, 100, 5) / 100; pausa = c[3].slider("Pausa/epoca (ms)", 0, 100, 10)
        c[4].metric("Filas que usara", f"{int((D['split'] == 0).sum() * frac):,}", help="De las filas de entrenamiento (2016-2022)")
        tab_live, tab_curva = st.tabs(["Entrenar", "Curva de aprendizaje (cuantos datos necesito)"])
        with tab_live:
            if st.button("▶ Entrenar", type="primary"):
                barra = st.progress(0.0); info = st.empty(); g1, g2 = st.columns([3, 2]); ch1, ch2 = g1.empty(), g2.empty(); paso = max(1, epocas // 100)

                def cb(e, w, hist):
                    if not (e % paso == 0 or e == epocas - 1): return
                    h = pd.DataFrame(hist, columns=["epoca", "tr", "va"])
                    f = fig_base(380, title="Error por epoca (MSE en log-demanda)"); f.add_trace(go.Scatter(x=h.epoca, y=h.tr, name="Entrenamiento", line=dict(color=AZUL)))
                    f.add_trace(go.Scatter(x=h.epoca, y=h.va, name="Validacion", line=dict(color=NARANJA))); f.update_yaxes(type="log"); ch1.plotly_chart(f, width="stretch")
                    s = pd.Series(w, index=D["cols"]).drop("bias").sort_values(); fb = fig_base(380, title="Peso de cada variable")
                    fb.add_trace(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[AZUL if x >= 0 else ROJO for x in s.values])); ch2.plotly_chart(fb, width="stretch")
                    barra.progress((e + 1) / epocas); info.write(f"Epoca **{e+1}/{epocas}** · MSE entrenamiento {h.tr.iloc[-1]:.4f} · validacion {h.va.iloc[-1]:.4f}")
                    if pausa: time.sleep(pausa / 1000)
                w, t, n = entrenar_numpy(D, epocas, lr, frac, cb); mae, rmse = guardar_modelo(tam, w, "Python (numpy, 1 hilo)", t, D, epocas=epocas, frac=frac, n=n)
                m = st.columns(4); m[0].metric("Filas usadas", f"{n:,}"); m[1].metric("MAE test", f"{mae:.1f} u."); m[2].metric("RMSE test", f"{rmse:.1f} u."); m[3].metric("Tiempo (con graficar)", f"{t:.1f} s")
                st.success("Modelo guardado. Ve a **Predicciones** o **Pronostico**.")
        with tab_curva:
            st.write("Entrena con 5%, 10%, 25%, 50% y 100% de las filas y mide error y tiempo: muestra cuantos datos hacen falta.")
            if st.button("Calcular curva de aprendizaje"):
                rows = []; pb = st.progress(0.0)
                for i, fr in enumerate([0.05, 0.1, 0.25, 0.5, 1.0]):
                    w, t, n = entrenar_numpy(D, EPOCHS_DEF, LR_DEF, fr); te = D["split"] == 2; mae, rmse = mae_rmse(D["X"][te] @ w, D["d"][te])
                    rows.append(dict(fraccion=f"{int(fr*100)}%", filas=n, MAE=mae, RMSE=rmse, tiempo_s=t)); pb.progress((i + 1) / 5)
                T = pd.DataFrame(rows); f = make_subplots(specs=[[{"secondary_y": True}]])
                f.add_trace(go.Scatter(x=T.filas, y=T.MAE, name="MAE test", mode="lines+markers", line=dict(color=AZUL)), secondary_y=False)
                f.add_trace(go.Scatter(x=T.filas, y=T.tiempo_s, name="Tiempo (s)", mode="lines+markers", line=dict(color=NARANJA)), secondary_y=True)
                f.update_xaxes(type="log", title="Filas de entrenamiento"); f.update_yaxes(title_text="MAE (unidades)", secondary_y=False); f.update_yaxes(title_text="Segundos", secondary_y=True)
                f.update_layout(template=TPL, height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1)); st.plotly_chart(f, width="stretch"); st.dataframe(T.round(3), width="stretch")

    # =========================================================== PREDICCIONES
    elif pagina == "Predicciones":
        requiere(modelo=True); D = get_D(tam); df = D["df"].copy()
        df["pred"] = np.expm1(np.clip(D["X"] @ M["w"], 0, None)); df["split"] = D["split"]; df["ingenuo"] = df.groupby(["comercio_id", "categoria"], sort=False).demanda.shift(1)
        st.title("Predicciones vs realidad")
        st.caption(f"Modelo: **{M['motor']}** entrenado en {M['t']:.2f} s con {M.get('n', 0):,} filas · MAE test {M['mae']:.1f} · RMSE {M['rmse']:.1f}")
        T = df[df.split == 2]; e = T.pred - T.demanda; ing_ok = T.ingenuo.notna()
        c = st.columns(4); c[0].metric("MAE test (modelo)", f"{e.abs().mean():.1f}"); c[1].metric("RMSE test", f"{np.sqrt((e**2).mean()):.1f}")
        c[2].metric("MAE ingenuo (mes anterior)", f"{(T.ingenuo[ing_ok] - T.demanda[ing_ok]).abs().mean():.1f}"); c[3].metric("Error relativo", f"{100*e.abs().mean()/T.demanda.mean():.1f} %")
        t1, t2, t3 = st.tabs(["Serie: real vs modelo", "Calidad global", "Donde falla"])
        with t1:
            a, b, cx = st.columns([1, 1, 1]); cat = a.selectbox("Categoria", sorted(df.categoria.unique()), index=7); com = b.selectbox("Comercio", sorted(df.comercio_id.unique()))
            solo = cx.radio("Mostrar", ["Validacion + prueba", "Todo el historial"], horizontal=True)
            s = df[(df.categoria == cat) & (df.comercio_id == com)]; s = s if solo == "Todo el historial" else s[s.split >= 1]; f = fig_base(460, title=f"{cat} · comercio {com}")
            f.add_vrect(x0="2023-01-01", x1="2024-07-01", fillcolor=GRIS, opacity=0.10, line_width=0, annotation_text="validacion", annotation_position="top left")
            f.add_vrect(x0="2024-07-01", x1="2026-01-01", fillcolor=VERDE, opacity=0.10, line_width=0, annotation_text="prueba", annotation_position="top left")
            f.add_trace(go.Scatter(x=s.fecha, y=s.demanda, name="Real", line=dict(color=AZUL, width=2.5), mode="lines+markers", marker=dict(size=5)))
            f.add_trace(go.Scatter(x=s.fecha, y=s.pred, name="Modelo", line=dict(color=NARANJA, width=2.5), mode="lines+markers", marker=dict(size=5)))
            f.add_trace(go.Scatter(x=s.fecha, y=s.ingenuo, name="Ingenuo (mes anterior)", line=dict(color=GRIS, dash="dash"))); f.update_xaxes(rangeslider_visible=True); f.update_layout(hovermode="x unified"); st.plotly_chart(f, width="stretch")
            st_ = s[s.split == 2]; st.caption(f"En la prueba de esta serie: MAE modelo {(st_.pred-st_.demanda).abs().mean():.1f} u. · demanda media {st_.demanda.mean():.1f} u.")
        with t2:
            a, b = st.columns(2); smp = T.sample(min(6000, len(T)), random_state=1)
            f = fig_base(420, title="Real vs predicho (muestra de la prueba)"); mx = float(max(smp.demanda.max(), smp.pred.max()))
            f.add_trace(go.Scattergl(x=smp.demanda, y=smp.pred, mode="markers", marker=dict(size=4, opacity=0.35, color=AZUL), name="Filas"))
            f.add_trace(go.Scatter(x=[1, mx], y=[1, mx], mode="lines", line=dict(color=ROJO, dash="dash"), name="Perfecto")); f.update_xaxes(type="log", title="Real"); f.update_yaxes(type="log", title="Predicho"); a.plotly_chart(f, width="stretch")
            f = fig_base(420, title="Distribucion del error (predicho − real)"); f.add_trace(go.Histogram(x=e.clip(-800, 800), nbinsx=80, marker_color=NARANJA)); f.add_vline(x=0, line_color=ROJO); b.plotly_chart(f, width="stretch")
        with t3:
            a, b = st.columns(2); pc = T.assign(ae=e.abs()).groupby("categoria").agg(MAE=("ae", "mean"), media=("demanda", "mean")); pc["rel"] = 100 * pc.MAE / pc.media; pc = pc.sort_values("rel")
            f = fig_base(400, title="Error relativo por categoria (%)"); f.add_trace(go.Bar(x=pc.rel, y=pc.index, orientation="h", marker_color=AZUL)); a.plotly_chart(f, width="stretch")
            pm = T.assign(ae=e.abs(), mes=T.fecha.dt.strftime("%Y-%m")).groupby("mes").ae.mean(); f = fig_base(400, title="MAE por mes (prueba)"); f.add_trace(go.Bar(x=pm.index, y=pm.values, marker_color=VERDE)); b.plotly_chart(f, width="stretch")
            peor = T.assign(ae=e.abs()).groupby(["comercio_id", "categoria"]).agg(MAE=("ae", "mean"), demanda_media=("demanda", "mean")).sort_values("MAE", ascending=False).head(10)
            st.subheader("10 series con mayor error"); st.dataframe(peor.round(1), width="stretch")

    # =========================================================== PRONOSTICO
    elif pagina == "Pronostico":
        requiere(modelo=True); D = get_D(tam)
        st.title("Pronostico de demanda con escenarios")
        st.caption("Proyeccion recursiva: cada mes pronosticado alimenta los rezagos del siguiente. Los escenarios fijan TRM, precio y evento fronterizo para los meses futuros.")
        c = st.columns(5)
        h = c[0].slider("Horizonte (meses)", 1, 12, 6); dtrm = c[1].slider("TRM vs ultimo dato (%)", -20, 30, 0); dpre = c[2].slider("Precio (%)", -20, 20, 0)
        ev = c[3].toggle("Evento fronterizo activo", value=False); vista = c[4].selectbox("Vista", ["Total del sistema", "Una categoria", "Una serie (comercio+categoria)"])
        R = pronosticar(tam, D, modelo_pronostico(tam, D), h, dtrm, dpre, ev); P, real, catv, first, fechas = R["P"], R["real"], R["catv"], R["first"], R["fechas"]; sig = M["sigma"]; z80 = 1.2816
        if vista == "Total del sistema": real_s, pred_s, titulo = real.sum(0), P.sum(0), "Demanda total del sistema"
        elif vista == "Una categoria":
            cat = st.selectbox("Categoria", CATS, index=7); m_ = catv == cat; real_s, pred_s, titulo = real[m_].sum(0), P[m_].sum(0), f"Demanda total · {cat}"
        else:
            a, b = st.columns(2); cat = a.selectbox("Categoria", CATS, index=7); com = b.selectbox("Comercio", sorted(first.comercio_id.unique()))
            idx = int(np.where((first.comercio_id.to_numpy() == com) & (catv == cat))[0][0]); real_s, pred_s, titulo = real[idx], P[idx], f"{cat} · comercio {com}"
        fechas_h = list(R["hist_fechas"][-36:]); y_h = real_s[-36:]
        f = fig_base(470, title=titulo + f" · {h} meses · TRM {dtrm:+d}% · precio {dpre:+d}% · evento {'si' if ev else 'no'}")
        f.add_trace(go.Scatter(x=fechas_h, y=y_h, name="Historico", line=dict(color=AZUL, width=2.5)))
        xs = [fechas_h[-1]] + fechas; ys = [y_h[-1]] + list(pred_s); lo = [y_h[-1]] + [x * np.exp(-z80 * sig) for x in pred_s]; hi = [y_h[-1]] + [x * np.exp(z80 * sig) for x in pred_s]
        if vista == "Una serie (comercio+categoria)": f.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(224,138,0,0.18)", line=dict(width=0), name="Banda ~80 %", hoverinfo="skip"))
        f.add_trace(go.Scatter(x=xs, y=ys, name="Pronostico", line=dict(color=NARANJA, width=3, dash="dash"), mode="lines+markers")); f.update_layout(hovermode="x unified"); st.plotly_chart(f, width="stretch")
        st.caption("La banda (solo en una serie) usa la dispersion del error de validacion en escala log y NO acumula incertidumbre entre meses: es orientativa. Los escenarios son supuestos tuyos. Para pronosticar el futuro el modelo se reentrena con TODO el historial (el modelo entrenado solo hasta 2022 sobrepredice ~19 % en 2024-25 por el cambio de nivel posterior). Es una regresion lineal basica.")
        a, b = st.columns([1, 2]); a.dataframe(pd.DataFrame({"mes": [x.strftime("%Y-%m") for x in fechas], "pronostico": pred_s.round(1)}), width="stretch", hide_index=True)
        por_cat = pd.DataFrame(P, columns=[x.strftime("%Y-%m") for x in fechas]); por_cat["categoria"] = catv; pc = por_cat.groupby("categoria").sum().round(0)
        fb = fig_base(360, title="Pronostico por categoria (todo el sistema)", barmode="stack")
        for col in pc.columns: fb.add_trace(go.Bar(x=pc.index, y=pc[col], name=col))
        b.plotly_chart(fb, width="stretch")

    # =========================================================== RENDIMIENTO
    elif pagina == "Rendimiento paralelo":
        st.title("Rendimiento: secuencial vs OpenMP vs MPI vs CUDA")
        R = resultados()
        if R is None: st.warning("No hay resultados consolidados (logs/fase5_resumen.csv)."); st.stop()
        st.caption("Mediciones hechas en el PC del laboratorio (i5-12600KF + RTX 2060). Este equipo no necesita esos motores para mostrarlas.")
        ds = st.radio("Dataset del benchmark", ["10k", "100k", "1M"], index=2, horizontal=True); S = R[R.dataset == ds].copy()
        S["etiqueta"] = S.modo.str.split(" ").str[0] + " · " + S.config
        t1, t2, t3 = st.tabs(["Speedup vs hilos/procesos", "Tiempos y escalabilidad", "Constructos de OpenMP"])
        with t1:
            f = fig_base(440, title=f"Speedup respecto a C secuencial · {ds}")
            for modo, col in (("OpenMP (gcc)", AZUL), ("MPI (MSVC+MS-MPI)", NARANJA)):
                s = S[S.modo == modo].copy(); s["x"] = s.config.str.split().str[0].astype(int)
                f.add_trace(go.Scatter(x=s.x, y=s["speedup_vs_C_seq(media)"], name=modo, mode="lines+markers", line=dict(color=col), error_y=dict(type="data", array=(s["speedup_vs_C_seq(media)"] * s["cv_%"] / 100).values)))
            cu = S[(S.modo == "CUDA (RTX 2060)") & (S.config == "todo en VRAM")]
            if len(cu): f.add_hline(y=float(cu["speedup_vs_C_seq(media)"].iloc[0]), line_color=VERDE, line_dash="dash", annotation_text="CUDA (datos en VRAM)")
            f.add_trace(go.Scatter(x=[1, 16], y=[1, 16], name="Ideal", line=dict(color=GRIS, dash="dot"))); f.update_xaxes(title="Hilos / procesos"); f.update_yaxes(title="Speedup"); st.plotly_chart(f, width="stretch")
            st.info("El speedup en CPU se satura (hipotesis: cuello de botella de memoria, no medido directamente) y varia entre corridas hasta ~±30 %.")
        with t2:
            a, b = st.columns(2)
            f = fig_base(480, title="Tiempo de entrenamiento (s)"); s = S.sort_values("t_media"); f.add_trace(go.Bar(x=s.t_media, y=s.etiqueta, orientation="h", marker_color=AZUL, error_x=dict(array=s.t_std))); f.update_yaxes(autorange="reversed"); a.plotly_chart(f, width="stretch")
            f = fig_base(480, title="Escalabilidad frente al tamano (mejor config de cada modo)"); filas_n = {"10k": 7056, "100k": 63504, "1M": 636048}
            for modo in R.modo.unique():
                b_ = R[R.modo == modo].sort_values("t_media").groupby("dataset").head(1).set_index("dataset").reindex(["10k", "100k", "1M"]); f.add_trace(go.Scatter(x=[filas_n[k] for k in b_.index], y=b_.t_media, name=modo.split(" (")[0], mode="lines+markers"))
            f.update_xaxes(type="log", title="Filas de entrenamiento"); f.update_yaxes(type="log", title="Segundos"); b.plotly_chart(f, width="stretch")
            st.dataframe(S[["modo", "config", "t_media", "t_std", "speedup_vs_C_seq(media)", "MAE", "RMSE"]].round(4), width="stretch", height=300)
            st.success("MAE y RMSE son iguales en todos los modos: paralelizar cambia el tiempo, no la precision.")
        with t3:
            import os
            if not os.path.exists("logs/fase6_omp_constructos.csv"): st.info("Resultados de constructos no disponibles."); st.stop()
            C = pd.read_csv("logs/fase6_omp_constructos.csv")
            sub = st.selectbox("Constructo", ["data_sharing", "schedule_uniforme", "schedule_irregular", "single_region", "sections", "tasks"],
                               format_func=lambda x: {"data_sharing": "parallel + data-sharing (reduction / critical / atomic / carrera)", "schedule_uniforme": "for + schedule (carga uniforme)", "schedule_irregular": "for + schedule (carga irregular)",
                                                      "single_region": "single (region unica) vs parallel for por epoca", "sections": "sections", "tasks": "task"}[x])
            s = C[C.demo == sub]; f = fig_base(420, title=sub)
            for hh, ss in s.groupby("hilos"): f.add_trace(go.Bar(x=ss.variante, y=ss.t_media * 1000, name=f"{hh} hilos", error_y=dict(array=ss.t_std * 1000)))
            f.update_yaxes(title="ms por pasada" if sub != "single_region" else "ms por entrenamiento (x1000)", type="log" if sub == "data_sharing" else "linear"); st.plotly_chart(f, width="stretch")
            if sub == "data_sharing":
                r = s[s.variante == "carrera_sin_proteccion"].error.max(); st.error(f"Sin proteccion, el gradiente calculado difiere hasta un **{100*r:.0f} %** del correcto (carrera de datos). reduction/critical/atomic dan error ≈ 1e-14.")
            st.dataframe(s.drop(columns=["demo"]).round(6), width="stretch")

    # =========================================================== LABORATORIO PARALELO
    else:
        st.title("Laboratorio paralelo: ejecuta cada motor en vivo")
        if not exes_ok():
            st.info("Este equipo no tiene los motores compilados (OpenMP/MPI/CUDA); esa parte se ejecuta en el PC del laboratorio. "
                    "Las mediciones reales estan en **Rendimiento paralelo**."); st.stop()
        requiere(); inf = info_dataset(tam)
        st.caption("Lanza los ejecutables compilados (1 repeticion). Para tiempos fiables, cierra programas pesados. El speedup se calcula contra una corrida secuencial hecha aqui mismo.")
        mot = [m for m in motores_disponibles() if not m.startswith("Python")]; c = st.columns(3)
        motor = c[0].selectbox("Motor", mot); hilos = c[1].select_slider("Hilos/procesos", [1, 2, 4, 6, 8, 10, 12, 16], value=6, disabled=motor not in ("OpenMP", "MPI"))
        lote = c[2].select_slider("Lote CUDA", [4096, 16384, 65536, 262144], value=65536, disabled=motor != "CUDA (por lotes)"); a, b = st.columns(2)
        if a.button("▶ Ejecutar este motor", type="primary"):
            with st.spinner("Entrenando..."):
                try:
                    _, t, mae, rmse, out = entrenar_exe(motor, tam, inf["p"], hilos, lote); _, ts, _, _, _ = entrenar_exe("C secuencial", tam, inf["p"], 1, 0)
                    k = st.columns(4); k[0].metric("Tiempo", f"{t:.3f} s"); k[1].metric("Speedup vs C secuencial", f"{ts/t:.2f}x", help=f"secuencial medido ahora: {ts:.3f} s"); k[2].metric("MAE test", f"{mae:.2f}"); k[3].metric("RMSE test", f"{rmse:.2f}")
                except Exception as ex: st.error(str(ex))
        if b.button("Comparar todos los motores"):
            res = []; pb = st.progress(0.0); conf = [("C secuencial", 1), ("OpenMP", 6)] + ([("MPI", 6), ("CUDA (todo en VRAM)", 1), ("CUDA (por lotes)", 1)] if cuda_ok() else [])
            for i, (m_, h_) in enumerate(conf):
                try: _, t, mae, rmse, _ = entrenar_exe(m_, tam, inf["p"], h_, 65536); res.append(dict(motor=m_ + (f" ({h_})" if m_ in ("OpenMP", "MPI") else ""), t=t, MAE=mae))
                except Exception as ex: st.error(f"{m_}: {ex}")
                pb.progress((i + 1) / len(conf))
            if res:
                T = pd.DataFrame(res); T["speedup"] = T.t.iloc[0] / T.t; f = make_subplots(rows=1, cols=2, subplot_titles=("Tiempo (s)", "Speedup vs C secuencial"))
                f.add_trace(go.Bar(x=T.motor, y=T.t, marker_color=AZUL), 1, 1); f.add_trace(go.Bar(x=T.motor, y=T.speedup, marker_color=VERDE), 1, 2)
                f.update_layout(template=TPL, height=400, showlegend=False, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(f, width="stretch"); st.dataframe(T.round(4), width="stretch")
