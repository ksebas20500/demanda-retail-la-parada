"""Unidad 2: compila y ejecuta omp_constructos.c, agrega resultados (media/std) y grafica.
Uso: python src/run_omp_constructos.py [--tam 1M] [--reps 5] [--pasadas 20]
"""
import argparse, datetime as dt, json, os, subprocess
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

GCC = r"C:\msys64\ucrt64\bin"
ENV = dict(os.environ, PATH=GCC + os.pathsep + os.environ["PATH"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tam", default="1M"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--pasadas", type=int, default=20)
    a = ap.parse_args()
    exe = os.path.abspath("build/omp_constructos.exe")
    subprocess.run([os.path.join(GCC, "gcc.exe"), "-O3", "-march=native", "-fopenmp", "-o", exe, "src/omp_constructos.c", "-lm"], env=ENV, check=True)
    p = json.load(open(f"data/proc/{a.tam}_meta.json"))["p"]
    r = subprocess.run([exe, a.tam, str(p), str(a.reps), str(a.pasadas)], env=ENV, capture_output=True, text=True, check=True)
    d = pd.read_csv(pd.io.common.StringIO(r.stdout), names=["demo", "variante", "hilos", "rep", "t_s", "error", "nota"])
    d.to_csv("logs/fase6_omp_constructos_raw.csv", index=False)
    g = d.groupby(["demo", "variante", "hilos"], sort=False).agg(t_media=("t_s", "mean"), t_std=("t_s", "std"), error=("error", "max"), nota=("nota", "first")).reset_index()
    g.round(6).to_csv("logs/fase6_omp_constructos.csv", index=False)
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    demos = ["data_sharing", "schedule_uniforme", "schedule_irregular", "single_region", "tasks", "sections"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for ax, dm in zip(axs.ravel(), demos):
        s = g[g.demo == dm]
        for h, ss in s.groupby("hilos"):
            ax.bar([f"{v}\n({h}h)" for v in ss.variante], ss.t_media * 1000, yerr=ss.t_std * 1000, capsize=2, label=f"{h} hilos")
        ax.set_title(dm); ax.set_ylabel("ms"); ax.tick_params(axis="x", labelsize=6, rotation=70)
        ax.set_yscale("log") if dm == "data_sharing" else None
    fig.suptitle(f"Constructos OpenMP sobre el gradiente ({a.tam})"); fig.tight_layout(); fig.savefig("figs/fase6_omp_constructos.png", dpi=120)
    pd.DataFrame([{"fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"), "fase": 6, "script": "run_omp_constructos.py",
                   "config": json.dumps({"tam": a.tam, "reps": a.reps, "pasadas": a.pasadas, "gcc": "-O3 -march=native -fopenmp"}),
                   "resultado": "ver logs/fase6_omp_constructos.csv"}]).to_csv("logs/experimentos.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
