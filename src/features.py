"""Construccion de features SIN fuga de informacion y division temporal.

Features de rezago usan solo meses anteriores de la misma serie (comercio, categoria).
Exogenas del mes objetivo (trm, precio, evento, mes) se asumen conocidas al pronosticar (escenario).
Division: train 2016-01..2022-12 (2015 se pierde por lag12), val 2023-01..2024-06, test 2024-07..2025-12.
Estandarizacion ajustada solo con train.
"""
import json, os
import numpy as np, pandas as pd

CATS = ["alimentos", "bebidas", "aseo_hogar", "ropa", "calzado", "electronica", "celulares",
        "juguetes", "utiles_escolares", "medicamentos", "licores", "ferreteria"]
VAL_INI, TEST_INI = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-07-01")


def construir(tam, ruta="data", devolver_escala=False, df_in=None):
    df = df_in.copy() if df_in is not None else pd.read_csv(f"{ruta}/ventas_{tam}.csv", parse_dates=["fecha"])
    df = df.sort_values(["comercio_id", "categoria", "fecha"]).reset_index(drop=True)
    df["ld"] = np.log1p(df.demanda)
    g = df.groupby(["comercio_id", "categoria"], sort=False)
    F = {}
    for l in (1, 2, 3, 6, 12):
        F[f"lag{l}"] = g.ld.shift(l)
    F["roll3"] = (F["lag1"] + F["lag2"] + F["lag3"]) / 3
    F["roll12"] = g.ld.transform(lambda s: s.shift(1).rolling(12).mean())
    F["ev_lag1"] = g.evento_fronterizo.shift(1)
    F["log_trm"] = np.log(df.trm)
    F["dlog_trm"] = np.log(df.trm) - np.log(g.trm.shift(1))
    F["log_precio"] = np.log(df.precio_cop)
    F["evento"] = df.evento_fronterizo.astype(float)
    m = df.fecha.dt.month
    F["mes_sin"], F["mes_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    F["zona_vdr"] = (df.zona == "Villa del Rosario").astype(float)
    for c in CATS:
        F[f"cat_{c}"] = (df.categoria == c).astype(float)
    X = pd.DataFrame(F)
    ok = X.notna().all(axis=1).to_numpy()
    X, df = X[ok].reset_index(drop=True), df[ok].reset_index(drop=True)
    split = np.where(df.fecha < VAL_INI, 0, np.where(df.fecha < TEST_INI, 1, 2))
    Xn = X.to_numpy(np.float64); tr = split == 0
    mu, sd = Xn[tr].mean(0), Xn[tr].std(0)
    cat_cols = [i for i, c in enumerate(X.columns) if c.startswith(("cat_", "zona_"))]
    mu[cat_cols], sd[cat_cols] = 0.0, 1.0  # dummies sin estandarizar
    sd[sd == 0] = 1.0
    Xs = ((Xn - mu) / sd).astype(np.float32)
    Xs = np.hstack([Xs, np.ones((len(Xs), 1), np.float32)])  # columna de sesgo
    out = (Xs, df.ld.to_numpy(np.float32), df.demanda.to_numpy(np.float32), split, list(X.columns) + ["bias"], df)
    return out + (mu, sd) if devolver_escala else out


def exportar(tam, ruta="data", out="data/proc"):
    """Exporta binarios float32 (fila-mayor) para las versiones C/OpenMP/MPI/CUDA."""
    os.makedirs(out, exist_ok=True)
    X, y, d, split, cols, _ = construir(tam, ruta)
    for k, nom in enumerate(["train", "val", "test"]):
        s = split == k
        X[s].tofile(f"{out}/{tam}_{nom}_X.bin"); y[s].tofile(f"{out}/{tam}_{nom}_y.bin"); d[s].tofile(f"{out}/{tam}_{nom}_d.bin")
    meta = {"p": X.shape[1], "n": [int((split == k).sum()) for k in range(3)], "cols": cols}
    json.dump(meta, open(f"{out}/{tam}_meta.json", "w"))
    return meta


if __name__ == "__main__":
    for t in ("10k", "100k", "1M"):
        print(t, {k: v for k, v in exportar(t).items() if k != "cols"})
