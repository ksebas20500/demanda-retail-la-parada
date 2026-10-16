@echo off
call "%~dp0env.bat"
cl 2>&1 | findstr /i "Version"
nvcc --version | findstr release
mpiexec -n 2 hostname
