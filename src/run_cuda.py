"""Fase 4b: compila gd_cuda.cu (nvcc sm_75) y mide GPU vs C secuencial para varios tamanos de lote.
Uso: python src/run_cuda.py [--reps 5]
"""
import argparse, datetime as dt, json, os, subprocess
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ENVBAT = os.path.abspath("build/env.bat")
BATCHES = {"10k": [0, 2048], "100k": [0, 16384, 4096], "1M": [0, 262144, 65536, 16384]}  # 0 = todo residente en VRAM


def cmd(c):
    r = subprocess.run(f'cmd /c ""{ENVBAT}" && {c}"', capture_output=True, text=True, shell=True)
    if r.returncode: raise RuntimeError(r.stdout + r.stderr)
    return r.stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=5); a = ap.parse_args()
    cmd("nvcc -O3 -arch=sm_75 -o build\\gd_cuda.exe src\\gd_cuda.cu")
    rows = []
    for tam, bs in BATCHES.items():
        p = json.load(open(f"data/proc/{tam}_meta.json"))["p"]
        for b in bs:
            out = cmd(f"build\\gd_cuda.exe {tam} {p} {b} {a.reps}")
            rows += [dict(zip(["dataset", "batch", "rep", "t", "t_h2d", "vram_MiB", "MAE", "RMSE"], l.split(","))) for l in out.split()]
        print(tam, "ok", flush=True)
    d = pd.DataFrame(rows).astype({"batch": int, "rep": int, "t": float, "t_h2d": float, "vram_MiB": float, "MAE": float, "RMSE": float})
    d.to_csv("logs/fase4_cuda_raw.csv", index=False)
    g = d.groupby(["dataset", "batch"]).agg(t_media=("t", "mean"), t_mediana=("t", "median"), t_std=("t", "std"), t_h2d=("t_h2d", "mean"),
                                           vram_MiB=("vram_MiB", "mean"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean")).reset_index()
    seq = pd.read_csv("logs/fase3_openmp.csv").drop_duplicates("dataset").set_index("dataset").t_seq_C
    g["t_seq_C"] = g.dataset.map(seq); g["speedup_vs_C_seq"] = g.t_seq_C / g.t_media
    g.round(5).to_csv("logs/fase4_cuda.csv", index=False)
    print(g.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tam, s in g.groupby("dataset"):
        ax.plot(s.batch.replace(0, s.batch.max() * 2 if s.batch.max() else 1).astype(str), s.speedup_vs_C_seq, marker="o", label=tam)
    ax.set_xlabel("Tamano de lote (filas; mayor = todo residente)"); ax.set_ylabel("Speedup vs C secuencial"); ax.set_yscale("log"); ax.legend(); ax.set_title("CUDA RTX 2060: speedup vs lote")
    fig.tight_layout(); fig.savefig("figs/fase4_cuda.png", dpi=130)
    pd.DataFrame([{"fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"), "fase": 4, "script": "run_cuda.py",
                   "config": json.dumps({"reps": a.reps, "batches": BATCHES, "nvcc": "13.4 -O3 sm_75", "gpu": "RTX 2060 6GB"}),
                   "resultado": "ver logs/fase4_cuda.csv"}]).to_csv("logs/experimentos.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
