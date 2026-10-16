@echo off
call "%~dp0env.bat"
cd /d "%~dp0.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
gcc -O3 -march=native -o build\gd_seq.exe src\gd_omp.c -lm || exit /b 1
gcc -O3 -march=native -fopenmp -o build\gd_omp.exe src\gd_omp.c -lm || exit /b 1
cl /nologo /O2 /arch:AVX2 /fp:fast /Fo:build\ /Fe:build\gd_mpi.exe src\gd_mpi.c /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib >nul || exit /b 1
nvcc -O3 -arch=sm_75 -o build\gd_cuda.exe src\gd_cuda.cu || exit /b 1
echo COMPILADO_OK
