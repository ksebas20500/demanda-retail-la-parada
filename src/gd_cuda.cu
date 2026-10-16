// Regresion lineal por GD (lote completo) en CUDA, sm_75. Mismo algoritmo/hiperparametros que gd_omp.c.
// El gradiente se calcula por tiles de 256 filas en memoria compartida y se acumula con atomicAdd(double).
// Si batch < n, los datos viven en memoria host "pinned" y se transmiten por lotes de `batch` filas en cada
// epoca (doble buffer, 2 streams): el resultado es identico a lote completo, solo cambia el uso de VRAM.
// Uso: gd_cuda <tam> <p> <batch|0=todo en GPU> <reps>
// Salida: tam,batch,rep,t_train_s,t_h2d_s,vram_MiB,MAE_test,RMSE_test
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define EPOCHS 300
#define LR 0.05f
#define L2 1e-4f
#define TILE 256
#define PMAX 32
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

static float *leer(const char *tam, const char *split, const char *suf, long *n, int cols, bool pinned) {
    char f[256]; snprintf(f, sizeof f, "data/proc/%s_%s_%s.bin", tam, split, suf);
    FILE *fp = fopen(f, "rb"); if (!fp) { fprintf(stderr, "no existe %s\n", f); exit(1); }
    fseek(fp, 0, SEEK_END); long b = ftell(fp); rewind(fp); *n = b / (4L * cols);
    float *a; if (pinned) CK(cudaMallocHost(&a, b)); else a = (float *)malloc(b);
    if (fread(a, 1, b, fp) != (size_t)b) { fprintf(stderr, "lectura\n"); exit(1); }
    fclose(fp); return a;
}

__constant__ float cw[PMAX];

__global__ void grad_kernel(const float *X, const float *y, long n, int p, double *g) {
    __shared__ float tile[TILE][PMAX + 1];     // stride impar => sin conflictos de banco
    __shared__ float res[TILE];
    __shared__ double part[8][PMAX];
    long base = (long)blockIdx.x * TILE; int t = threadIdx.x;
    long rows = min((long)TILE, n - base);
    for (int k = t; k < TILE * p; k += blockDim.x) {          // carga coalescida
        int r = k / p, c = k % p;
        tile[r][c] = (r < rows) ? X[(base + r) * p + c] : 0.f;
    }
    __syncthreads();
    float s = 0.f;
    if (t < rows) { for (int j = 0; j < p; j++) s += tile[t][j] * cw[j]; res[t] = s - y[base + t]; } else res[t] = 0.f;
    __syncthreads();
    if (t < 8 * p) {
        int j = t % p, q = t / p; double acc = 0;
        for (int r = q; r < TILE; r += 8) acc += (double)res[r] * tile[r][j];
        part[q][j] = acc;
    }
    __syncthreads();
    if (t < p) { double a = 0; for (int q = 0; q < 8; q++) a += part[q][t]; atomicAdd(&g[t], a); }
}

__global__ void update_kernel(float *w, double *g, long n, int p) {
    int j = threadIdx.x; if (j < p) { w[j] -= LR * ((float)(g[j] / n) + L2 * w[j]); g[j] = 0.0; }
}

int main(int argc, char **argv) {
#ifdef _WIN32   /* evita que Windows 11 relegue el proceso a nucleos eficientes (EcoQoS) y falsee las mediciones */
    { PROCESS_POWER_THROTTLING_STATE st_ = {PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
      SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st_, sizeof st_); }
#endif

    if (argc < 5) { fprintf(stderr, "uso: gd_cuda tam p batch reps\n"); return 1; }
    const char *tam = argv[1]; int p = atoi(argv[2]); long batch = atol(argv[3]); int reps = atoi(argv[4]);
    if (p > PMAX) { fprintf(stderr, "p>PMAX\n"); return 1; }
    long n, t1, nte, t2;
    float *hX = leer(tam, "train", "X", &n, p, true), *hy = leer(tam, "train", "y", &t1, 1, true);
    float *Xte = leer(tam, "test", "X", &nte, p, false), *dte = leer(tam, "test", "d", &t2, 1, false);
    bool resident = (batch <= 0 || batch >= n); if (resident) batch = n;
    size_t free0, tot; CK(cudaMemGetInfo(&free0, &tot));
    float *dX[2], *dy[2], *dw; double *dg; cudaStream_t st[2];
    int nb = resident ? 1 : 2;
    for (int i = 0; i < nb; i++) { CK(cudaMalloc(&dX[i], (size_t)batch * p * 4)); CK(cudaMalloc(&dy[i], (size_t)batch * 4)); CK(cudaStreamCreate(&st[i])); }
    CK(cudaMalloc(&dw, 4 * PMAX)); CK(cudaMalloc(&dg, 8 * PMAX));
    double h2d = 0;
    if (resident) {
        auto a = std::chrono::steady_clock::now();
        CK(cudaMemcpy(dX[0], hX, (size_t)n * p * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dy[0], hy, (size_t)n * 4, cudaMemcpyHostToDevice));
        h2d = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    }
    size_t free1; CK(cudaMemGetInfo(&free1, &tot)); double vram = (free0 - free1) / 1048576.0;
    float w[PMAX];
    for (int r = -1; r < reps; r++) {            // r=-1: calentamiento (no se reporta)
        for (int j = 0; j < PMAX; j++) w[j] = 0.f;
        CK(cudaMemcpy(dw, w, 4 * PMAX, cudaMemcpyHostToDevice)); CK(cudaMemset(dg, 0, 8 * PMAX)); CK(cudaMemcpyToSymbol(cw, w, 4 * PMAX));
        CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        for (int e = 0; e < EPOCHS; e++) {
            // los pesos van a __constant__ desde dw (copia D2D)
            CK(cudaMemcpyToSymbol(cw, dw, 4 * PMAX, 0, cudaMemcpyDeviceToDevice));
            int c = 0;
            for (long off = 0; off < n; off += batch, c++) {
                long m = min(batch, n - off);
                if (resident) { grad_kernel<<<(m + TILE - 1) / TILE, TILE>>>(dX[0], dy[0], m, p, dg); }
                else {
                    int s = c & 1;
                    CK(cudaMemcpyAsync(dX[s], hX + off * p, (size_t)m * p * 4, cudaMemcpyHostToDevice, st[s]));
                    CK(cudaMemcpyAsync(dy[s], hy + off, (size_t)m * 4, cudaMemcpyHostToDevice, st[s]));
                    grad_kernel<<<(m + TILE - 1) / TILE, TILE, 0, st[s]>>>(dX[s], dy[s], m, p, dg);
                }
            }
            CK(cudaDeviceSynchronize());
            update_kernel<<<1, 32>>>(dw, dg, n, p);
        }
        CK(cudaDeviceSynchronize());
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r < 0) continue;
        CK(cudaMemcpy(w, dw, 4 * PMAX, cudaMemcpyDeviceToHost));
        double ae = 0, se = 0;
        for (long i = 0; i < nte; i++) {
            float s = 0.f; for (int j = 0; j < p; j++) s += Xte[i * p + j] * w[j];
            if (s < 0) s = 0; double e = expm1((double)s) - dte[i]; ae += fabs(e); se += e * e;
        }
        printf("%s,%ld,%d,%.6f,%.6f,%.1f,%.4f,%.4f\n", tam, resident ? 0L : batch, r, t, h2d, vram, ae / nte, sqrt(se / nte));
    }
    if (argc > 5) { FILE *fw = fopen(argv[5], "wb"); if (fw) { fwrite(w, 4, p, fw); fclose(fw); } }   // opcional: guarda pesos
    return 0;
}
