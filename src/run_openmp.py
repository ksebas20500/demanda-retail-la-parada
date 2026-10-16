"""Fase 3: compila gd_omp.c (gcc) y mide secuencial C vs OpenMP con varios hilos.
Uso: python src/run_openmp.py [--reps 5]
"""
import argparse, datetime as dt, json, os, subprocess, io
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

GCC_DIR = r"C:\msys64\ucrt64\bin"
ENV = dict(os.environ, PATH=GCC_DIR + os.pathsep + os.environ["PATH"])
HILOS = [1, 2, 4, 6, 8, 10, 12, 16]  # 16 = maximo logico (10 nucleos fisicos: 6P con HT + 4E)
def pcols(tam):
    return json.load(open(f'data/proc/{tam}_meta.json'))['p']


def run(cmd):
    return subprocess.run(cmd, env=ENV, capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=5); a = ap.parse_args()
    os.makedirs("build", exist_ok=True)
    flags = ["-O3", "-march=native"]
    subprocess.run([os.path.join(GCC_DIR, "gcc.exe"), *flags, "-o", os.path.abspath("build/gd_seq.exe"), "src/gd_omp.c", "-lm"], env=ENV, check=True)
    subprocess.run([os.path.join(GCC_DIR, "gcc.exe"), *flags, "-fopenmp", "-o", os.path.abspath("build/gd_omp.exe"), "src/gd_omp.c", "-lm"], env=ENV, check=True)
    rows = []
    for tam in ("10k", "100k", "1M"):
        out = run([os.path.abspath("build/gd_seq.exe"), tam, str(pcols(tam)), "1", str(a.reps)])
        rows += [dict(zip(["dataset", "hilos", "rep", "t", "MAE", "RMSE"], l.split(","))) | {"modo": "C_secuencial"} for l in out.split()]
        for h in HILOS:
            out = run([os.path.abspath("build/gd_omp.exe"), tam, str(pcols(tam)), str(h), str(a.reps)])
            rows += [dict(zip(["dataset", "hilos", "rep", "t", "MAE", "RMSE"], l.split(","))) | {"modo": "openmp"} for l in out.split()]
        print(tam, "ok", flush=True)
    d = pd.DataFrame(rows).astype({"hilos": int, "rep": int, "t": float, "MAE": float, "RMSE": float})
    d.to_csv("logs/fase3_openmp_raw.csv", index=False)
    seq = d[d.modo == "C_secuencial"].groupby("dataset").t.mean()
    py = pd.read_csv("logs/fase2_baseline.csv").query("modelo=='linreg_gd_300ep'").set_index("dataset").t_train_s_media
    g = d[d.modo == "openmp"].groupby(["dataset", "hilos"]).agg(t_media=("t", "mean"), t_std=("t", "std"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean")).reset_index()
    g["t_seq_C"] = g.dataset.map(seq); g["t_seq_python"] = g.dataset.map(py)
    g["speedup_vs_C_seq"] = g.t_seq_C / g.t_media; g["speedup_vs_python"] = g.t_seq_python / g.t_media
    g["eficiencia"] = g.speedup_vs_C_seq / g.hilos
    g.round(5).to_csv("logs/fase3_openmp.csv", index=False)
    print(g.round(4).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for tam, s in g.groupby("dataset"):
        ax[0].plot(s.hilos, s.speedup_vs_C_seq, marker="o", label=tam)
    ax[0].plot(HILOS, HILOS, "k--", lw=.8, label="ideal"); ax[0].set_xlabel("Hilos"); ax[0].set_ylabel("Speedup vs C secuencial"); ax[0].legend(); ax[0].set_title("OpenMP: speedup vs hilos")
    for h in (1, 4, 16):
        s = g[g.hilos == h]; ax[1].plot(s.dataset, s.t_media, marker="o", label=f"{h} hilos")
    ax[1].set_yscale("log"); ax[1].set_ylabel("Tiempo de entrenamiento (s)"); ax[1].set_xlabel("Tamano de dataset"); ax[1].legend(); ax[1].set_title("Escalabilidad vs tamano")
    fig.tight_layout(); fig.savefig("figs/fase3_openmp.png", dpi=130)
    cfg = {"reps": a.reps, "hilos": HILOS, "compilador": "gcc 16.1 -O3 -march=native -fopenmp", "cpu": "i5-12600KF 10c/16t"}
    pd.DataFrame([{"fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"), "fase": 3, "script": "run_openmp.py",
                   "config": json.dumps(cfg), "resultado": "ver logs/fase3_openmp.csv"}]).to_csv("logs/experimentos.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
