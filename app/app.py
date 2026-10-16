"""Sistema de prediccion de demanda (La Parada). Dos modos: Recorrido guiado (feria/presentacion) y Modo experto.
Ejecutar:  python -m streamlit run app/app.py     (funciona en Windows, macOS y Linux; sin GPU tambien)
"""
import os
# Un solo hilo de BLAS: tiempos reproducibles y comparables entre equipos (debe fijarse ANTES de importar numpy).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import streamlit as st
st.set_page_config(page_title="Predicción de demanda · La Parada", layout="wide", page_icon="📈", initial_sidebar_state="auto")
from sin_throttle import sin_throttle
sin_throttle()
import nucleo, experto, recorrido

CSS_BASE = """<style>
.block-container{padding-top:3.2rem;max-width:1400px}
div[data-testid="stMetric"]{background:#f4f6f9;border-radius:12px;padding:12px 16px;border:1px solid #e3e7ee}
@media (prefers-color-scheme: dark){div[data-testid="stMetric"]{background:#1d2330;border-color:#2c3446}}
</style>"""
CSS_FERIA = """<style>
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"]{display:none !important}
html{font-size:18px}
h1{font-size:2.6rem !important} h2{font-size:2.1rem !important}
div[data-testid="stMetricValue"]{font-size:2rem}
.stButton>button{font-size:1.2rem;padding:.6rem 1.6rem;border-radius:12px}
div[data-testid="stRadio"] label p{font-size:.95rem}
</style>"""

st.markdown(CSS_BASE, unsafe_allow_html=True)
top = st.columns([3, 3])
top[0].caption("Universidad de Pamplona · Sistemas Inteligentes · Predicción de demanda con computación paralela")
modo = top[1].radio("Modo", ["🎪 Recorrido guiado", "🔬 Modo experto"], horizontal=True, label_visibility="collapsed", key="modo_ui")
if modo.startswith("🎪"):
    st.markdown(CSS_FERIA, unsafe_allow_html=True); recorrido.render()
else:
    experto.render()
