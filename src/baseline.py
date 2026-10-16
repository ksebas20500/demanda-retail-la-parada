"""Fase 2 - Linea base secuencial (1 hilo).

Modelos:
  1. Regresion lineal por descenso de gradiente (lote completo), EPOCHS/LR fijos -> es el kernel que se
     paralelizara en OpenMP/MPI/CUDA. Entrena sobre log1p(demanda); MAE/RMSE en unidades (expm1).
  2. Arbol de decision (sklearn, max_depth=12) como referencia de precision.
Referencias ingenuas: lag-12 estacional y lag-1.
Uso: python src/baseline.py [--reps 3]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"  # secuencial real
import argparse, datetime as dt, json, platform, time
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeRegressor

SEED, EPOCHS, LR, L2 = 42, 300, 0.05, 1e-4
sys_path = os.path.dirname(__file__)
import sys; sys.path.insert(0, sys_path)
from features import construir
from sin_throttle import sin_throttle
sin_throttle()


def entrenar_gd(X, y):
    n, p = X.shape
    w = np.zeros(p, np.float32)
    for _ in range(EPOCHS):
        r = X @ w - y                      # residuos
        g = (X.T @ r) / n + L2 * w         # gradiente
        w -= LR * g
    return w


def metricas(pred_ld, d):
    pred = np.expm1(np.clip(pred_ld, 0, None))
    e = pred - d
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(); np.random.seed(SEED)
    os.makedirs("logs", exist_ok=True)
    filas = []
    for tam in ("10k", "100k", "1M"):
        X, y, d, split, cols, df = construir(tam)
        tr, va, te = (split == 0), (split == 1), (split == 2)
        Xtr, ytr = np.ascontiguousarray(X[tr]), y[tr]
        # naive
        i1, i12 = cols.index("lag1"), cols.index("lag12")
        # lag en escala estandarizada -> usar valores originales desde df
        g = df.groupby(["comercio_id", "categoria"], sort=False).demanda
        for nom, ref in (("naive_lag1", g.shift(1)), ("naive_lag12", g.shift(12))):
            r = ref.to_numpy()[te]; ok = ~np.isnan(r)
            e = r[ok] - d[te][ok]
            filas.append((tam, nom, np.abs(e).mean(), np.sqrt((e ** 2).mean()), 0.0, 0.0, int(tr.sum())))
        # regresion lineal GD
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); w = entrenar_gd(Xtr, ytr); ts.append(time.perf_counter() - t0)
        mae_v, rmse_v = metricas(X[va] @ w, d[va]); mae, rmse = metricas(X[te] @ w, d[te])
        filas.append((tam, "linreg_gd_300ep", mae, rmse, float(np.mean(ts)), float(np.std(ts)), int(tr.sum())))
        # arbol
        ts = []
        for _ in range(a.reps):
            m = DecisionTreeRegressor(max_depth=12, random_state=SEED)
            t0 = time.perf_counter(); m.fit(Xtr, ytr); ts.append(time.perf_counter() - t0)
        mae, rmse = metricas(m.predict(X[te]), d[te])
        filas.append((tam, "arbol_depth12", mae, rmse, float(np.mean(ts)), float(np.std(ts)), int(tr.sum())))
        print(tam, "listo", flush=True)
    T = pd.DataFrame(filas, columns=["dataset", "modelo", "MAE_test", "RMSE_test", "t_train_s_media", "t_train_s_std", "n_train"])
    T.round(4).to_csv("logs/fase2_baseline.csv", index=False)
    print(T.round(3).to_string(index=False))
    cfg = {"reps": a.reps, "epochs": EPOCHS, "lr": LR, "l2": L2, "seed": SEED, "hilos": 1, "cpu": platform.processor()}
    pd.DataFrame([{"fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"), "fase": 2, "script": "baseline.py",
                   "config": json.dumps(cfg), "resultado": "ver logs/fase2_baseline.csv"}]).to_csv(
        "logs/experimentos.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
