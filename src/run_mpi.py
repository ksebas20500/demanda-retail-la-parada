"""Fase 4a: compila gd_mpi.c con MSVC + MS-MPI y mide speedup vs procesos (1 maquina).
Uso: python src/run_mpi.py [--reps 5]
"""
import argparse, datetime as dt, json, os, subprocess
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROCS = [1, 2, 4, 6, 8, 10, 12, 16]
ENVBAT = os.path.abspath("build/env.bat")
SDK = r"C:\Program Files (x86)\Microsoft SDKs\MPI"


def cmd(c):
    r = subprocess.run(f'cmd /c ""{ENVBAT}" && {c}"', capture_output=True, text=True, shell=True)
    if r.returncode: raise RuntimeError(r.stdout + r.stderr)
    return r.stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=5); a = ap.parse_args()
    cmd(f'cl /nologo /O2 /arch:AVX2 /fp:fast /Fo:build\\ /Fe:build\\gd_mpi.exe src\\gd_mpi.c /I"{SDK}\\Include" /link /LIBPATH:"{SDK}\\Lib\\x64" msmpi.lib')
    rows = []
    for tam in ("10k", "100k", "1M"):
        p = json.load(open(f"data/proc/{tam}_meta.json"))["p"]
        for n in PROCS:
            out = cmd(f"mpiexec -n {n} build\\gd_mpi.exe {tam} {p} {a.reps}")
            rows += [dict(zip(["dataset", "procesos", "rep", "t", "MAE", "RMSE"], l.split(","))) for l in out.split()]
        print(tam, "ok", flush=True)
    d = pd.DataFrame(rows).astype({"procesos": int, "rep": int, "t": float, "MAE": float, "RMSE": float})
    d.to_csv("logs/fase4_mpi_raw.csv", index=False)
    g = d.groupby(["dataset", "procesos"]).agg(t_media=("t", "mean"), t_std=("t", "std"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean")).reset_index()
    base = g[g.procesos == 1].set_index("dataset").t_media
    g["t_1proc"] = g.dataset.map(base); g["speedup"] = g.t_1proc / g.t_media; g["eficiencia"] = g.speedup / g.procesos
    gc = pd.read_csv("logs/fase3_openmp.csv"); seq = gc.drop_duplicates("dataset").set_index("dataset").t_seq_C
    g["speedup_vs_C_gcc_seq"] = g.dataset.map(seq) / g.t_media
    g.round(5).to_csv("logs/fase4_mpi.csv", index=False)
    print(g.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tam, s in g.groupby("dataset"): ax.plot(s.procesos, s.speedup, marker="o", label=tam)
    ax.plot(PROCS, PROCS, "k--", lw=.8, label="ideal"); ax.set_xlabel("Procesos MPI"); ax.set_ylabel("Speedup vs 1 proceso MPI"); ax.legend(); ax.set_title("MPI (1 maquina): speedup")
    fig.tight_layout(); fig.savefig("figs/fase4_mpi.png", dpi=130)
    pd.DataFrame([{"fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"), "fase": 4, "script": "run_mpi.py",
                   "config": json.dumps({"reps": a.reps, "procesos": PROCS, "compilador": "MSVC /O2 /arch:AVX2 /fp:fast + MS-MPI 10.1"}),
                   "resultado": "ver logs/fase4_mpi.csv"}]).to_csv("logs/experimentos.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
