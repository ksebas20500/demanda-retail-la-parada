@echo off
call "%~dp0env.bat"
cd /d "%~dp0"
cl /nologo /O2 /openmp t_omp.c >nul && "%~dp0t_omp.exe"
cl /nologo /O2 t_mpi.c /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib >nul && mpiexec -n 2 "%~dp0t_mpi.exe"
nvcc -arch=sm_75 t_cuda.cu -o t_cuda.exe && "%~dp0t_cuda.exe"
