"""Crea dist/demanda_la_parada_demo.zip: paquete portatil (macOS/Windows/Linux) sin motores compilados ni datasets pesados.
Los datasets se generan solos al abrir la interfaz. Uso: python empaquetar_demo.py
"""
import os, zipfile

RAIZ = os.path.dirname(os.path.abspath(__file__)); SALIDA = os.path.join(RAIZ, "dist"); TOP = "demanda_la_parada_demo"
ARCHIVOS = ["app/app.py", "app/nucleo.py", "app/experto.py", "app/recorrido.py",
            "src/generar_datos.py", "src/features.py", "src/sin_throttle.py",
            "data/externos/trm_diaria.csv", "logs/fase5_resumen.csv", "logs/fase2_baseline.csv", "logs/fase6_omp_constructos.csv",
            ".streamlit/config.toml", "requirements_demo.txt", "iniciar_mac.command", "iniciar_windows.bat", "abrir_interfaz_red.bat", "README_DEMO.md",
            "docs/guion_presentacion.md", "docs/fundamentos_teoria.md", "docs/fase5_desarrollo_conclusiones.md", "docs/fase6_constructos_resultados.md"]
EJECUTABLES = {"iniciar_mac.command"}


def main():
    os.makedirs(SALIDA, exist_ok=True); destino = os.path.join(SALIDA, "demanda_la_parada_demo.zip")
    faltan = [a for a in ARCHIVOS if not os.path.exists(os.path.join(RAIZ, a))]
    if faltan: raise SystemExit(f"Faltan archivos: {faltan}")
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for a in ARCHIVOS:
            data = open(os.path.join(RAIZ, a), "rb").read()
            if a.endswith((".command", ".bat", ".py", ".md", ".toml", ".txt")): data = data.replace(b"\r\n", b"\n") if a.endswith((".command",)) else data
            zi = zipfile.ZipInfo(f"{TOP}/{a}"); zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if a in EJECUTABLES else 0o644) << 16
            z.writestr(zi, data)
    print(f"{destino}  ({os.path.getsize(destino)/1024:.0f} KB, {len(ARCHIVOS)} archivos)")


if __name__ == "__main__":
    main()
