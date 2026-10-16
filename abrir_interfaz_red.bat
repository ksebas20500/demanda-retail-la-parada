@echo off
REM Sirve la interfaz a OTROS equipos de la misma red (p. ej. el MacBook abre http://IP-DE-ESTE-PC:8501)
cd /d "%~dp0"
echo Direcciones IP de este PC (usa la de tu red WiFi, tipo 192.168.x.x):
ipconfig | findstr /i "IPv4"
echo.
echo En el otro equipo abre:  http://ESA-IP:8501      (si no carga, permite Python en el Firewall de Windows)
python -m streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
