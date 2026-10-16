#!/bin/bash
# Inicia la interfaz en macOS / Linux. Doble clic en Finder (si no abre: chmod +x iniciar_mac.command) o: bash iniciar_mac.command
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "No se encontro Python 3. Instalalo desde https://www.python.org/downloads/ (version 3.11 o superior) y vuelve a abrir este archivo."; read -p "Enter para salir"; exit 1
fi
if [ ! -d ".venv" ]; then
  echo "Primera vez: creando entorno e instalando paquetes (necesita internet, ~1-2 minutos)..."
  python3 -m venv .venv || { echo "No se pudo crear el entorno"; read -p "Enter para salir"; exit 1; }
  source .venv/bin/activate
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements_demo.txt || { echo "Fallo la instalacion (revisa internet)"; read -p "Enter para salir"; exit 1; }
else
  source .venv/bin/activate
fi
echo "Abriendo la interfaz en http://localhost:8501  (Ctrl+C para cerrar)"
python -m streamlit run app/app.py
