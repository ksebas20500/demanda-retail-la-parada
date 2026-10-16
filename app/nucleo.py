"""Nucleo compartido de la interfaz (datos, entrenamiento, pronostico, motores). Funciona en Windows, macOS y Linux:
los motores compilados (C/OpenMP/MPI/CUDA) solo existen en el PC del laboratorio; sin ellos todo sigue funcionando con Python."""
import os, sys, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(RAIZ, "src")); os.chdir(RAIZ)
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from features import construir, exportar, CATS
from generar_datos import generar

EPOCHS_DEF, LR_DEF, L2 = 300, 0.05, 1e-4
GCC = r"C:\msys64\ucrt64\bin"; ENVBAT = os.path.abspath("build/env.bat")
VRAM_MIB = 6144
AZUL, NARANJA, GRIS, VERDE, ROJO = "#1f4e79", "#e08a00", "#7f8c8d", "#2e8b57", "#d9534f"
TPL = "plotly_white"
PRESETS = {"10k": 7, "100k": 63, "1M": 631}            # comercios; filas = comercios * 1584
TRM_CSV = "data/externos/trm_diaria.csv"
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
NOMBRES_VAR = {"lag1": "Demanda del mes pasado", "lag2": "Demanda de hace 2 meses", "lag3": "Demanda de hace 3 meses", "lag6": "Demanda de hace 6 meses",
               "lag12": "Demanda del mismo mes del año pasado", "roll3": "Promedio de los últimos 3 meses", "roll12": "Promedio del último año",
               "ev_lag1": "Hubo evento de frontera el mes pasado", "log_trm": "Nivel del dólar (TRM)", "dlog_trm": "Cambio del dólar vs mes anterior",
               "log_precio": "Precio del producto", "evento": "Evento de frontera este mes", "mes_sin": "Época del año (seno)", "mes_cos": "Época del año (coseno)",
               "zona_vdr": "Zona: Villa del Rosario", "bias": "Valor base"}
EVENTOS_ES = {"cierre_frontera_2015": "Cierre 2015", "cierre_puentes_2019": "Cierre 2019", "pandemia_covid": "Pandemia", "reapertura_2022": "Reapertura", "paro_bloqueo": "Paro"}


def nombre_var(c): return NOMBRES_VAR.get(c) or ("Categoría: " + c[4:].replace("_", " ") if c.startswith("cat_") else c)


# ------------------------------------------------------------------ disponibilidad de motores compilados
def exes_ok():
    if os.environ.get("DEMO_SIN_MOTORES") == "1" or sys.platform != "win32": return False
    return all(os.path.exists(f"build/{e}") for e in ("gd_seq.exe", "gd_omp.exe"))


def cuda_ok():
    return exes_ok() and os.path.exists("build/gd_cuda.exe") and os.path.exists("build/gd_mpi.exe")


def motores_disponibles():
    m = ["Python (numpy, 1 hilo)"]
    if exes_ok(): m += ["C secuencial", "OpenMP"]
    if cuda_ok(): m += ["MPI", "CUDA (todo en VRAM)", "CUDA (por lotes)"]
    return m


# ------------------------------------------------------------------ datos
def ruta_datos(tam): return f"data/ventas_{tam}.csv"
def tiene_datos(tam): return os.path.exists(ruta_datos(tam))
def tiene_feats(tam): return tiene_datos(tam) and st.session_state.get(f"feats_{tam}", False)
def mtime(path): return os.path.getmtime(path) if os.path.exists(path) else 0
def filas_de(tam):
    if tam in PRESETS: return PRESETS[tam] * 1584
    return int(tam[1:]) * 1584 if tam.startswith("c") else None


def asegurar_datos(tam):
    """Genera el dataset si no existe (TRM oficial, semilla 42). Devuelve segundos usados."""
    if tiene_datos(tam): return 0.0
    t0 = time.perf_counter(); os.makedirs("data", exist_ok=True)
    n = PRESETS.get(tam) or int(tam[1:])
    df, _ = generar(n, 42, TRM_CSV if os.path.exists(TRM_CSV) else None, 42); df.to_csv(ruta_datos(tam), index=False)
    return time.perf_counter() - t0


@st.cache_resource(show_spinner="Cargando ventas...")
def ventas(tam, _mt):
    return pd.read_csv(ruta_datos(tam), parse_dates=["fecha"])


@st.cache_resource(show_spinner="Preparando variables (meses anteriores, dólar, eventos...)")
def _cargar(tam, _mt):
    X, y, d, split, cols, df, mu, sd = construir(tam, devolver_escala=True)
    return dict(X=X, y=y, d=d, split=split, cols=cols, df=df, mu=mu, sd=sd)


def get_D(tam):
    D = _cargar(tam, mtime(ruta_datos(tam))); st.session_state[f"feats_{tam}"] = True; return D


@st.cache_resource(show_spinner=False)
def _matrices_serie(tam, _mt):
    v = ventas(tam, _mt); g = v.sort_values(["comercio_id", "categoria", "fecha"]); T = g.fecha.nunique(); S = len(g) // T
    first = g.iloc[::T]
    return dict(ld=np.log1p(g.demanda.to_numpy(float)).reshape(S, T), pr=g.precio_cop.to_numpy(float).reshape(S, T)[:, -1],
                evl=g.evento_fronterizo.to_numpy(float).reshape(S, T)[:, -1], trm_last=float(g.trm.iloc[-1]),
                zona=(first.zona.to_numpy() == "Villa del Rosario").astype(float), catv=first.categoria.to_numpy(), first=first.reset_index(drop=True),
                fechas=g.fecha.unique(), S=S, T=T, last=g.fecha.max())


def mae_rmse(pred_ld, d):
    e = np.expm1(np.clip(pred_ld, 0, None)) - d
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def info_dataset(tam):
    f = filas_de(tam)
    if not tiene_datos(tam): return dict(filas=f, comercios=f // 1584 if f else None, ntr=0, nva=0, nte=0, p=0)
    D = get_D(tam); s = D["split"]
    return dict(filas=f, comercios=f // 1584 if f else None, ntr=int((s == 0).sum()), nva=int((s == 1).sum()), nte=int((s == 2).sum()), p=len(D["cols"]))


def recursos(inf):
    ntr, p = inf["ntr"], inf["p"] or 28
    return dict(X_mb=ntr * p * 4 / 1048576, vram_mib=ntr * (p * 4 + 4) / 1048576, pct=100 * ntr * (p * 4 + 4) / 1048576 / VRAM_MIB,
                max_filas=int(VRAM_MIB * 0.9 * 1048576 / (p * 4 + 4)))


def resultados():
    p = "logs/fase5_resumen.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


# ------------------------------------------------------------------ graficos
def fig_base(h=420, **kw):
    f = go.Figure(); f.update_layout(template=TPL, height=h, margin=dict(l=10, r=10, t=90, b=10), legend=dict(orientation="h", y=1.02, yanchor="bottom"), **kw); return f


def franjas_evento(f, agg, etiquetas=False, tipos=None):
    run = None; ini_t = None
    for fecha, ev in zip(agg.index, agg.ev):
        if ev and run is None: run = fecha
        if not ev and run is not None:
            f.add_vrect(x0=run, x1=fecha, fillcolor=ROJO, opacity=0.12, line_width=0)
            if etiquetas and tipos is not None: f.add_annotation(x=run, y=1.0, yref="paper", text=EVENTOS_ES.get(tipos.get(run, ""), "Evento"), showarrow=False, xanchor="left", font=dict(size=11, color=ROJO))
            run = None
    if run is not None: f.add_vrect(x0=run, x1=agg.index[-1], fillcolor=ROJO, opacity=0.12, line_width=0)


# ------------------------------------------------------------------ entrenamiento
def entrenar_numpy(D, epocas, lr, frac, callback=None, seed=42):
    tr = D["split"] == 0; Xtr, ytr = D["X"][tr], D["y"][tr]
    if frac < 1.0:
        idx = np.random.default_rng(seed).choice(len(Xtr), int(len(Xtr) * frac), replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    va = D["split"] == 1; Xva, yva = D["X"][va], D["y"][va]
    n, p = Xtr.shape; w = np.zeros(p, np.float32); t0 = time.perf_counter(); hist = []
    for e in range(epocas):
        r = Xtr @ w - ytr
        w -= lr * ((Xtr.T @ r) / n + L2 * w)
        if callback:
            hist.append((e + 1, float((r ** 2).mean() / 2), float(((Xva @ w - yva) ** 2).mean() / 2))); callback(e, w, hist)
    return w, time.perf_counter() - t0, len(Xtr)


def entrenar_exe(motor, tam, p, hilos, lote):
    if not tiene_feats(tam) or not os.path.exists(f"data/proc/{tam}_meta.json"): exportar(tam)
    os.makedirs("data/modelos", exist_ok=True); wf = os.path.abspath(f"data/modelos/w_{tam}.bin")
    if os.path.exists(wf): os.remove(wf)
    env = dict(os.environ, PATH=GCC + os.pathsep + os.environ["PATH"])
    if motor == "C secuencial": cmd = f'build\\gd_seq.exe {tam} {p} 1 1 "{wf}"'
    elif motor == "OpenMP": cmd = f'build\\gd_omp.exe {tam} {p} {hilos} 1 "{wf}"'
    elif motor == "MPI": cmd = f'call "{ENVBAT}" && mpiexec -n {hilos} build\\gd_mpi.exe {tam} {p} 1 "{wf}"'
    elif motor == "CUDA (todo en VRAM)": cmd = f'build\\gd_cuda.exe {tam} {p} 0 1 "{wf}"'
    else: cmd = f'build\\gd_cuda.exe {tam} {p} {lote} 1 "{wf}"'
    r = subprocess.run(f'cmd /c "{cmd}"', capture_output=True, text=True, shell=True, env=env)
    if r.returncode or not r.stdout.strip(): raise RuntimeError((r.stderr or r.stdout)[:500])
    f = r.stdout.strip().splitlines()[-1].split(",")
    w = np.fromfile(wf, dtype=np.float32) if os.path.exists(wf) else None
    return w, float(f[3]), float(f[-2]), float(f[-1]), r.stdout.strip()


def guardar_modelo(tam, w, motor, t, D, **extra):
    te = D["split"] == 2; va = D["split"] == 1
    mae, rmse = mae_rmse(D["X"][te] @ w, D["d"][te])
    st.session_state[f"modelo_{tam}"] = dict(w=w, motor=motor, t=t, mae=mae, rmse=rmse, sigma=float(np.std(D["y"][va] - D["X"][va] @ w)), **extra)
    return mae, rmse


def asegurar_modelo(tam):
    """Garantiza datos + variables + un modelo entrenado (numpy, 300 epocas) para el dataset."""
    asegurar_datos(tam); D = get_D(tam)
    if f"modelo_{tam}" not in st.session_state:
        w, t, n = entrenar_numpy(D, EPOCHS_DEF, LR_DEF, 1.0); guardar_modelo(tam, w, "Python (numpy, 1 hilo)", t, D, epocas=EPOCHS_DEF, frac=1.0, n=n)
    return D


def modelo_pronostico(tam, D):
    """Modelo para pronosticar el FUTURO: se reentrena con TODO el historial (2016-2025), practica habitual tras validar.
    La evaluacion honesta (meses nunca vistos) usa el modelo entrenado solo hasta 2022; este no se usa para medir precision."""
    k = f"modelo_full_{tam}"
    if k not in st.session_state:
        w = np.zeros(D["X"].shape[1], np.float32); X, y = D["X"], D["y"]; n = len(X)
        for _ in range(EPOCHS_DEF): w -= LR_DEF * ((X.T @ (X @ w - y)) / n + L2 * w)
        st.session_state[k] = w
    return st.session_state[k]


def mae_ingenuo_global(tam, D):
    k = f"ing_{tam}"
    if k not in st.session_state:
        df = D["df"]; te = D["split"] == 2; sub = df[te]
        ing = sub.groupby(["comercio_id", "categoria"], sort=False).demanda.shift(1).to_numpy(); ok = ~np.isnan(ing)
        st.session_state[k] = float(np.abs(ing[ok] - sub.demanda.to_numpy()[ok]).mean())
    return st.session_state[k]


def serie_df(D, comercio, categoria):
    """Filas (ordenadas por fecha) de una serie con prediccion del modelo guardado se calcula afuera."""
    df = D["df"]; m = ((df.comercio_id == comercio) & (df.categoria == categoria)).to_numpy()
    return np.where(m)[0]


# ------------------------------------------------------------------ pronostico recursivo con escenarios
def pronosticar(tam, D, w, h, dtrm, dpre, ev):
    mt = mtime(ruta_datos(tam)); Sx = _matrices_serie(tam, mt); S = Sx["S"]; hist = Sx["ld"][:, -12:].copy()
    raw_cols = D["cols"][:-1]; fechas = []; preds = []; trm_last = Sx["trm_last"]; trm_prev = trm_last
    for k in range(1, h + 1):
        fecha = Sx["last"] + pd.DateOffset(months=k); fechas.append(fecha); trm_k = trm_last * (1 + dtrm / 100)
        F = {"lag1": hist[:, -1], "lag2": hist[:, -2], "lag3": hist[:, -3], "lag6": hist[:, -6], "lag12": hist[:, -12]}
        F["roll3"] = (F["lag1"] + F["lag2"] + F["lag3"]) / 3; F["roll12"] = hist.mean(1); F["ev_lag1"] = Sx["evl"] if k == 1 else np.full(S, float(ev))
        F["log_trm"] = np.full(S, np.log(trm_k)); F["dlog_trm"] = np.full(S, np.log(trm_k) - np.log(trm_prev)); F["log_precio"] = np.log(Sx["pr"] * (1 + dpre / 100))
        F["evento"] = np.full(S, float(ev)); F["mes_sin"] = np.full(S, np.sin(2 * np.pi * fecha.month / 12)); F["mes_cos"] = np.full(S, np.cos(2 * np.pi * fecha.month / 12)); F["zona_vdr"] = Sx["zona"]
        for cn in CATS: F[f"cat_{cn}"] = (Sx["catv"] == cn).astype(float)
        R = np.stack([F[c] for c in raw_cols], axis=1); Z = (R - D["mu"]) / D["sd"]; pl = np.clip(Z @ w[:-1] + w[-1], 0, None)
        preds.append(np.expm1(pl)); hist = np.hstack([hist[:, 1:], pl[:, None]]); trm_prev = trm_k
    return dict(P=np.stack(preds, axis=1), real=np.expm1(Sx["ld"]), catv=Sx["catv"], first=Sx["first"], fechas=fechas, hist_fechas=Sx["fechas"], S=S)
