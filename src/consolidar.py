"""Fase 5: consolida tiempos (media, desviacion, mediana) y precision de todos los modos y genera graficas finales.
Lee logs/fase*_raw.csv y logs/fase2_baseline.csv. Escribe logs/fase5_resumen.csv y figs/fase5_*.png.
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

omp = pd.read_csv("logs/fase3_openmp_raw.csv"); mpi = pd.read_csv("logs/fase4_mpi_raw.csv"); cu = pd.read_csv("logs/fase4_cuda_raw.csv")
py = pd.read_csv("logs/fase2_baseline.csv").query("modelo=='linreg_gd_300ep'")

def agg(d, keys):
    return d.groupby(keys).agg(t_media=("t", "mean"), t_std=("t", "std"), t_mediana=("t", "median"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), n=("t", "size")).reset_index()

rows = []
for _, r in py.iterrows():
    rows.append(dict(dataset=r.dataset, modo="Python secuencial (1 hilo)", config="numpy 1 hilo", t_media=r.t_train_s_media, t_std=r.t_train_s_std, t_mediana=np.nan, MAE=r.MAE_test, RMSE=r.RMSE_test, n=5))
for _, r in agg(omp[omp.modo == "C_secuencial"], ["dataset"]).iterrows():
    rows.append(dict(r) | dict(modo="C secuencial (gcc)", config="1 hilo, sin OpenMP"))
for _, r in agg(omp[omp.modo == "openmp"], ["dataset", "hilos"]).iterrows():
    rows.append(dict(r) | dict(modo="OpenMP (gcc)", config=f"{int(r.hilos)} hilos"))
for _, r in agg(mpi, ["dataset", "procesos"]).iterrows():
    rows.append(dict(r) | dict(modo="MPI (MSVC+MS-MPI)", config=f"{int(r.procesos)} procesos"))
for _, r in agg(cu, ["dataset", "batch"]).iterrows():
    rows.append(dict(r) | dict(modo="CUDA (RTX 2060)", config="todo en VRAM" if r.batch == 0 else f"lote {int(r.batch)} filas"))
R = pd.DataFrame(rows)[["dataset", "modo", "config", "t_media", "t_std", "t_mediana", "MAE", "RMSE", "n"]]
ref = R[R.modo == "C secuencial (gcc)"].set_index("dataset").t_media
R["speedup_vs_C_seq(media)"] = R.dataset.map(ref) / R.t_media
R["speedup_vs_C_seq(mediana)"] = R.dataset.map(R[R.modo == "C secuencial (gcc)"].set_index("dataset").t_mediana) / R.t_mediana
R["cv_%"] = 100 * R.t_std / R.t_media
R.round(5).to_csv("logs/fase5_resumen.csv", index=False)
print(R.round(3).to_string(index=False))

orden = ["10k", "100k", "1M"]
# Fig 1: speedup vs hilos/procesos (media con barras = propagacion simple de std)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for k, (modo, titulo, x) in enumerate([("OpenMP (gcc)", "OpenMP: speedup vs hilos", "hilos"), ("MPI (MSVC+MS-MPI)", "MPI: speedup vs procesos", "procesos")]):
    for tam in orden:
        s = R[(R.dataset == tam) & (R.modo == modo)].copy(); s["x"] = s.config.str.split().str[0].astype(int)
        sp = s["speedup_vs_C_seq(media)"]; err = sp * s["cv_%"] / 100
        ax[k].errorbar(s.x, sp, yerr=err, marker="o", capsize=3, label=tam)
    ax[k].plot([1, 16], [1, 16], "k--", lw=.8, label="ideal"); ax[k].set_xlabel(x.capitalize()); ax[k].set_title(titulo); ax[k].legend(); ax[k].grid(alpha=.3)
ax[0].set_ylabel("Speedup vs C secuencial (gcc)"); fig.tight_layout(); fig.savefig("figs/fase5_speedup_hilos_procesos.png", dpi=130)

# Fig 2: escalabilidad vs tamano de dataset (mejor config de cada modo)
fig, ax = plt.subplots(figsize=(7, 4.5)); filas_n = {"10k": 7056, "100k": 63504, "1M": 636048}
for modo in ["Python secuencial (1 hilo)", "C secuencial (gcc)", "OpenMP (gcc)", "MPI (MSVC+MS-MPI)", "CUDA (RTX 2060)"]:
    best = R[R.modo == modo].sort_values("t_media").groupby("dataset").head(1).set_index("dataset").loc[orden]
    ax.plot([filas_n[t] for t in orden], best.t_media, marker="o", label=modo)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Filas de entrenamiento"); ax.set_ylabel("Tiempo de entrenamiento (s)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Escalabilidad frente al tamano de datos (mejor config)"); fig.tight_layout(); fig.savefig("figs/fase5_escalabilidad_datos.png", dpi=130)

# Fig 3: MAE/RMSE por modo (1M)
m = R[(R.dataset == "1M")].groupby("modo").agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean")).reindex(["Python secuencial (1 hilo)", "C secuencial (gcc)", "OpenMP (gcc)", "MPI (MSVC+MS-MPI)", "CUDA (RTX 2060)"])
fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(m)); ax.bar(x - .2, m.MAE, .4, label="MAE"); ax.bar(x + .2, m.RMSE, .4, label="RMSE")
ax.set_xticks(x); ax.set_xticklabels([s.split(" (")[0] for s in m.index], rotation=15); ax.set_ylabel("Unidades"); ax.set_ylim(0, m.RMSE.max() * 1.25); ax.legend(); ax.set_title("Error en test por modo de ejecucion (dataset 1M)")
for i, (a, b) in enumerate(zip(m.MAE, m.RMSE)): ax.text(i - .2, a + 3, f"{a:.1f}", ha="center", fontsize=8); ax.text(i + .2, b + 3, f"{b:.1f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig("figs/fase5_precision_por_modo.png", dpi=130)
