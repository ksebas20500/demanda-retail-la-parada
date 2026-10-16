"""Generador de datos sinteticos de ventas - La Parada (Villa del Rosario / Cucuta).

Registro = (mes, comercio, categoria). Reglas de negocio explicitas:
  log(lambda) = log(base_k) + log(escala_comercio) + estacionalidad_k(mes) + tendencia
                + gamma_k * log(TRM/3500) + beta_k * (log markup + eps_precio)
                + log(mult_evento) + ruido AR(1)
  demanda ~ Gamma-Poisson (binomial negativa) con media lambda.

Uso:  python src/generar_datos.py [--trm-csv ruta.csv] [--seed 42]
El CSV de TRM real debe tener columnas: fecha,trm (se promedia por mes).
"""
import argparse, datetime as dt, json, os, platform
import numpy as np
import pandas as pd

SEED = 42
INICIO, FIN = "2015-01", "2025-12"
TAMANOS = {"10k": 7, "100k": 63, "1M": 631}  # n comercios; filas = n * 132 * 12

# categoria: (unidades base/mes, precio base COP, gamma TRM, beta precio, pass-through TRM->precio, bumps estacionales)
# bump = (mes, amplitud log, ancho en meses)
CATS = {
    "alimentos":      (900, 6_000, 0.10, -0.6, 0.20, [(12, .18, 1.0), (4, .05, 1.0)]),
    "bebidas":        (700, 4_500, 0.12, -0.7, 0.30, [(12, .20, 1.2), (6, .10, 1.5), (8, .08, 1.0)]),
    "aseo_hogar":     (500, 9_000, 0.15, -0.8, 0.35, [(12, .08, 1.0)]),
    "ropa":           (350, 45_000, 0.22, -1.0, 0.55, [(12, .40, 1.2), (5, .30, 0.8), (6, .15, 0.8), (1, .15, 0.8)]),
    "calzado":        (200, 80_000, 0.22, -1.0, 0.55, [(12, .35, 1.2), (5, .25, 0.8), (1, .20, 0.8)]),
    "electronica":    (80, 600_000, 0.35, -1.3, 0.90, [(12, .45, 1.0), (11, .25, 0.8), (5, .20, 0.8)]),
    "celulares":      (120, 900_000, 0.40, -1.4, 0.95, [(12, .40, 1.0), (5, .25, 0.8), (6, .15, 0.8)]),
    "juguetes":       (150, 35_000, 0.18, -1.0, 0.60, [(12, .90, 1.0), (4, .25, 0.8)]),
    "utiles_escolares": (250, 8_000, 0.08, -0.7, 0.30, [(1, .90, 0.8), (2, .50, 0.8), (7, .35, 0.8)]),
    "medicamentos":   (400, 18_000, 0.25, -0.5, 0.40, [(3, .06, 1.5), (10, .06, 1.5)]),
    "licores":        (300, 40_000, 0.12, -0.9, 0.35, [(12, .55, 1.0), (6, .15, 1.0), (8, .12, 1.0)]),
    "ferreteria":     (180, 25_000, 0.10, -0.8, 0.50, [(6, .10, 1.5), (11, .10, 1.5)]),
}

# Eventos fronterizos (mes_inicio, mes_fin, tipo, multiplicador de demanda). Fechas aproximadas de hitos publicos.
EVENTOS = [
    ("2015-08", "2016-07", "cierre_frontera_2015", 0.70),
    ("2019-02", "2019-12", "cierre_puentes_2019", 0.65),
    ("2020-03", "2021-03", "pandemia_covid", 0.45),
    ("2022-09", "2022-12", "reapertura_2022", 1.35),
]
P_PARO_ALEATORIO = 0.05  # prob. mensual de paro/bloqueo (mult. 0.75), comun a todos los comercios
TRM_ANCLAS = {  # niveles aproximados (COP/USD); la serie es SIMULADA, no es la TRM oficial
    "2015-01": 2400, "2015-12": 3150, "2016-02": 3300, "2016-12": 3000, "2017-12": 2980,
    "2018-06": 2900, "2018-12": 3200, "2019-12": 3290, "2020-03": 4000, "2020-12": 3500,
    "2021-12": 3950, "2022-07": 4400, "2022-10": 4800, "2023-06": 4100, "2023-12": 3950,
    "2024-12": 4400, "2025-06": 4100, "2025-12": 3900,
}


def serie_trm(meses, rng, trm_csv=None):
    if trm_csv:
        d = pd.read_csv(trm_csv, parse_dates=["fecha"])
        m = d.groupby(d["fecha"].dt.to_period("M").dt.to_timestamp())["trm"].mean()
        return m.reindex(meses).interpolate().bfill().ffill().to_numpy(), "real_datos.gov.co_TRM"
    ancla = pd.Series(TRM_ANCLAS); ancla.index = pd.to_datetime(ancla.index)
    base = ancla.reindex(meses.union(ancla.index)).interpolate("time").reindex(meses).to_numpy()
    ruido = np.zeros(len(meses))
    for t in range(1, len(meses)):  # AR(1) multiplicativo
        ruido[t] = 0.6 * ruido[t - 1] + rng.normal(0, 0.02)
    return base * np.exp(ruido), "simulada"


def estacionalidad():
    mes = np.arange(1, 13)
    s = {}
    for k, (*_, bumps) in CATS.items():
        v = np.zeros(12)
        for (m0, amp, w) in bumps:
            d = np.minimum(np.abs(mes - m0), 12 - np.abs(mes - m0))
            v += amp * np.exp(-0.5 * (d / w) ** 2)
        s[k] = v - v.mean()
    return s


def mult_eventos(meses, rng):
    mult = np.ones(len(meses)); flag = np.zeros(len(meses), int); tipo = np.array(["ninguno"] * len(meses), dtype=object)
    for ini, fin, t, mu in EVENTOS:
        sel = (meses >= pd.Timestamp(ini)) & (meses <= pd.Timestamp(fin))
        mult[sel] *= mu; flag[sel] = 1; tipo[sel] = t
    paros = rng.random(len(meses)) < P_PARO_ALEATORIO
    paros &= flag == 0
    mult[paros] *= 0.75; flag[paros] = 1; tipo[paros] = "paro_bloqueo"
    mult[(meses >= pd.Timestamp("2023-01")) & (flag == 0)] *= 1.10  # nivel post-reapertura (no marcado como evento)
    return mult, flag, tipo


def generar(n_com, rng_global_seed, trm_csv=None, seed=SEED):
    meses = pd.date_range(INICIO, FIN, freq="MS")
    T, K = len(meses), len(CATS)
    # Estructura comun (misma para todos los tamanos => datasets anidados)
    rg = np.random.default_rng(rng_global_seed)
    trm, trm_src = serie_trm(meses, rg, trm_csv)
    mev, flag, tipo = mult_eventos(meses, rg)
    escala_all = rg.lognormal(0, 0.6, 631)
    markup_all = rg.lognormal(0, 0.10, 631)
    zona_all = np.where(rg.random(631) < 0.7, "Villa del Rosario", "Cucuta")
    sens_all = rg.normal(1.0, 0.25, 631).clip(0.4, 1.8)  # sensibilidad del comercio a TRM
    # Ruido propio de cada tamano
    rng = np.random.default_rng(seed + n_com)

    cats = list(CATS); est = estacionalidad()
    base = np.array([CATS[k][0] for k in cats]); pbase = np.array([CATS[k][1] for k in cats], float)
    gam = np.array([CATS[k][2] for k in cats]); bet = np.array([CATS[k][3] for k in cats])
    pas = np.array([CATS[k][4] for k in cats])
    S = np.stack([est[k][meses.month - 1] for k in cats], axis=1)          # T x K
    tend = 0.01 * (np.arange(T) / 12.0)                                    # +1 %/anio
    lt = np.log(trm / 3500.0)

    N = n_com
    # AR(1) de log-ruido por (comercio, categoria)
    u = np.zeros((T, N, K)); e = rng.normal(0, 0.18, (T, N, K))
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + e[t]
    eps_p = rng.normal(0, 0.05, (T, N, K))
    lmk = np.log(markup_all[:N])[None, :, None]

    loglam = (np.log(base)[None, None, :] + np.log(escala_all[:N])[None, :, None]
              + S[:, None, :] + tend[:, None, None]
              + gam[None, None, :] * sens_all[:N][None, :, None] * lt[:, None, None]
              + bet[None, None, :] * (lmk + eps_p)
              + np.log(mev)[:, None, None] + u)
    lam = np.exp(loglam)
    r = 20.0
    dem = rng.poisson(rng.gamma(r, lam / r)).astype(np.int32)
    precio = (pbase[None, None, :] * (trm / 3500.0)[:, None, None] ** pas[None, None, :]
              * np.exp(lmk + eps_p)).round(-1).astype(np.int64)

    df = pd.DataFrame({
        "fecha": np.repeat(meses.values, N * K),
        "comercio_id": np.tile(np.repeat(np.arange(N), K), T),
        "zona": np.tile(np.repeat(zona_all[:N], K), T),
        "categoria": np.tile(np.array(cats), T * N),
        "precio_cop": precio.reshape(-1),
        "trm": np.repeat(trm.round(2), N * K),
        "evento_fronterizo": np.repeat(flag, N * K).astype(np.int8),
        "tipo_evento": np.repeat(tipo, N * K),
        "demanda": dem.reshape(-1),
    })
    return df, trm_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trm-csv", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("logs", exist_ok=True)
    for nombre, n in TAMANOS.items():
        t0 = dt.datetime.now()
        df, src = generar(n, a.seed, a.trm_csv, a.seed)
        ruta = os.path.join(a.out, f"ventas_{nombre}.csv")
        df.to_csv(ruta, index=False)
        seg = (dt.datetime.now() - t0).total_seconds()
        print(f"{ruta}: {len(df):,} filas, TRM {src}, {seg:.1f}s")
        fila = pd.DataFrame([{
            "fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"), "fase": 1, "script": "generar_datos.py",
            "config": json.dumps({"tamano": nombre, "comercios": n, "seed": a.seed, "trm": src}),
            "resultado": f"filas={len(df)}; demanda_media={df.demanda.mean():.2f}; seg={seg:.1f}"}])
        fila.to_csv("logs/experimentos.csv", mode="a", header=not os.path.exists("logs/experimentos.csv"), index=False)


if __name__ == "__main__":
    main()
