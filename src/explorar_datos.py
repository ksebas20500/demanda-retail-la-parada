"""Graficas exploratorias y verificacion cuantitativa (estacionalidad/volatilidad vs ruido)."""
import datetime as dt, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs("figs", exist_ok=True)
df = pd.read_csv("data/ventas_100k.csv", parse_dates=["fecha"])
df["mes"] = df.fecha.dt.month
agg = df.groupby("fecha").agg(dem=("demanda", "sum"), trm=("trm", "first"), ev=("evento_fronterizo", "first"), tipo=("tipo_evento", "first"))

# 1. Serie agregada + TRM + eventos
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(agg.index, agg.dem, color="#1f4e79", lw=1.4, label="Demanda total mensual")
for f in agg.index[agg.ev == 1]:
    ax.axvspan(f, f + pd.offsets.MonthBegin(1), color="#d9534f", alpha=.18, lw=0)
ax.set_ylabel("Unidades"); ax.set_title("Demanda agregada (63 comercios) - sombreado: evento fronterizo")
ax2 = ax.twinx(); ax2.plot(agg.index, agg.trm, color="#e08a00", lw=1, label="TRM simulada"); ax2.set_ylabel("COP/USD")
ax.legend(loc="upper left"); ax2.legend(loc="upper right"); fig.tight_layout(); fig.savefig("figs/01_serie_agregada.png", dpi=130); plt.close(fig)

# 2. Heatmap estacionalidad mes x categoria (indice sobre media anual, solo meses sin evento)
sin = df[df.evento_fronterizo == 0]
h = sin.groupby(["categoria", "mes"]).demanda.mean().unstack()
h = h.div(h.mean(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(10, 5)); im = ax.imshow(h.values, cmap="YlOrRd", aspect="auto")
ax.set_yticks(range(len(h))); ax.set_yticklabels(h.index); ax.set_xticks(range(12)); ax.set_xticklabels(range(1, 13))
ax.set_xlabel("Mes"); ax.set_title("Indice estacional por categoria (1.0 = promedio anual)")
fig.colorbar(im); fig.tight_layout(); fig.savefig("figs/02_estacionalidad_heatmap.png", dpi=130); plt.close(fig)

# 3. Distribucion de la demanda
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].hist(df.demanda, bins=80, color="#1f4e79"); axs[0].set_title("Distribucion de demanda"); axs[0].set_xlabel("Unidades/mes")
axs[1].hist(np.log1p(df.demanda), bins=80, color="#2e8b57"); axs[1].set_title("Distribucion de log(1+demanda)")
fig.tight_layout(); fig.savefig("figs/03_distribucion.png", dpi=130); plt.close(fig)

# 4. Estacionalidad por categoria (3 categorias) y efecto TRM
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for c in ["juguetes", "utiles_escolares", "alimentos", "celulares"]:
    axs[0].plot(range(1, 13), h.loc[c], marker="o", label=c)
axs[0].set_xlabel("Mes"); axs[0].set_ylabel("Indice"); axs[0].legend(); axs[0].set_title("Perfil estacional")
s = sin[sin.categoria == "celulares"].groupby("fecha").agg(d=("demanda", "mean"), t=("trm", "first"))
axs[1].scatter(s.t, s.d, s=14, color="#8e44ad"); axs[1].set_xlabel("TRM"); axs[1].set_ylabel("Demanda media celulares"); axs[1].set_title("Demanda vs TRM (meses sin evento)")
fig.tight_layout(); fig.savefig("figs/04_estacionalidad_y_trm.png", dpi=130); plt.close(fig)

# 5. Boxplot por tipo de evento
fig, ax = plt.subplots(figsize=(10, 4))
g = df.groupby("tipo_evento").demanda.apply(lambda x: np.log1p(x).values)
ax.boxplot(g.values, tick_labels=list(g.index), showfliers=False); ax.set_ylabel("log(1+demanda)"); plt.xticks(rotation=25)
ax.set_title("Demanda segun tipo de evento fronterizo"); fig.tight_layout(); fig.savefig("figs/05_eventos.png", dpi=130); plt.close(fig)

# Verificacion cuantitativa: varianza de log-demanda (centrada por serie) explicada por mes del anio
df["ld"] = np.log1p(df.demanda)
df["ld_c"] = df.ld - df.groupby(["comercio_id", "categoria"]).ld.transform("mean")
def r2_grupo(col):
    m = df.groupby(["categoria", col]).ld_c.transform("mean")
    return 1 - ((df.ld_c - m) ** 2).sum() / (df.ld_c ** 2).sum()
r2_mes = r2_grupo("mes")
r2_ev = r2_grupo("tipo_evento")
rng = np.random.default_rng(42)
df["mes_perm"] = df.groupby(["comercio_id", "categoria"]).mes.transform(lambda x: rng.permutation(x.values))
r2_perm = r2_grupo("mes_perm")
cv = (df.groupby(["comercio_id", "categoria"]).demanda.std() / df.groupby(["comercio_id", "categoria"]).demanda.mean()).mean()
res = {"R2_mes_x_categoria": round(r2_mes, 4), "R2_mes_permutado_control": round(r2_perm, 4),
       "R2_tipo_evento": round(r2_ev, 4), "CV_medio_por_serie": round(cv, 3),
       "pico_dic_vs_media_agregada": round(float(agg[agg.index.month == 12].dem.mean() / agg.dem.mean()), 3)}
print(json.dumps(res, indent=1))
pd.DataFrame([{"fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"), "fase": 1, "script": "explorar_datos.py",
               "config": json.dumps({"dataset": "100k", "seed": 42}), "resultado": json.dumps(res)}]
             ).to_csv("logs/experimentos.csv", mode="a", header=False, index=False)
