/* Regresion lineal por GD, paralelismo de datos con MPI (MS-MPI). Cada proceso lee su bloque de filas de
 * entrenamiento; el gradiente parcial se suma con MPI_Allreduce cada epoca. Cada proceso es monohilo.
 * Uso: mpiexec -n N gd_mpi.exe <tam> <p> <reps>
 * Salida (rank 0): tam,procesos,rep,t_train_s,MAE_test,RMSE_test
 */
#include <mpi.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPOCHS 300
#define LR 0.05f
#define L2 1e-4f

static float *leer_trozo(const char *tam, const char *split, const char *suf, long fila0, long nfilas, int cols) {
    char f[256]; snprintf(f, sizeof f, "data/proc/%s_%s_%s.bin", tam, split, suf);
    FILE *fp = fopen(f, "rb"); if (!fp) { fprintf(stderr, "no existe %s\n", f); MPI_Abort(MPI_COMM_WORLD, 1); }
    _fseeki64(fp, (__int64)fila0 * cols * 4, SEEK_SET);
    float *a = malloc((size_t)nfilas * cols * 4);
    if (fread(a, 4, (size_t)nfilas * cols, fp) != (size_t)nfilas * cols) { fprintf(stderr, "lectura\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    fclose(fp); return a;
}

static long filas_total(const char *tam, const char *split, const char *suf, int cols) {
    char f[256]; snprintf(f, sizeof f, "data/proc/%s_%s_%s.bin", tam, split, suf);
    FILE *fp = fopen(f, "rb"); _fseeki64(fp, 0, SEEK_END); long n = (long)(_ftelli64(fp) / (4 * cols)); fclose(fp); return n;
}

int main(int argc, char **argv) {
#ifdef _WIN32   /* evita que Windows 11 relegue el proceso a nucleos eficientes (EcoQoS) y falsee las mediciones */
    { PROCESS_POWER_THROTTLING_STATE st_ = {PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
      SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st_, sizeof st_); }
#endif

    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 4) { if (!rank) fprintf(stderr, "uso: gd_mpi tam p reps\n"); MPI_Finalize(); return 1; }
    const char *tam = argv[1]; int p = atoi(argv[2]), reps = atoi(argv[3]);
    long n = filas_total(tam, "train", "X", p);
    long a = n * rank / size, b = n * (rank + 1) / size, nl = b - a;
    float *X = leer_trozo(tam, "train", "X", a, nl, p), *y = leer_trozo(tam, "train", "y", a, nl, 1);
    float *w = malloc(4 * p); double *g = malloc(8 * p);
    long nte = 0; float *Xte = NULL, *dte = NULL;
    if (!rank) { nte = filas_total(tam, "test", "X", p); Xte = leer_trozo(tam, "test", "X", 0, nte, p); dte = leer_trozo(tam, "test", "d", 0, nte, 1); }

    for (int r = 0; r < reps; r++) {
        for (int j = 0; j < p; j++) w[j] = 0.f;
        MPI_Barrier(MPI_COMM_WORLD); double t0 = MPI_Wtime();
        for (int e = 0; e < EPOCHS; e++) {
            for (int j = 0; j < p; j++) g[j] = 0.0;
            for (long i = 0; i < nl; i++) {
                const float *x = X + i * p; float s = 0.f;
                for (int j = 0; j < p; j++) s += x[j] * w[j];
                float res = s - y[i];
                for (int j = 0; j < p; j++) g[j] += (double)res * x[j];
            }
            MPI_Allreduce(MPI_IN_PLACE, g, p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            for (int j = 0; j < p; j++) w[j] -= LR * ((float)(g[j] / n) + L2 * w[j]);
        }
        MPI_Barrier(MPI_COMM_WORLD); double t = MPI_Wtime() - t0;
        if (!rank) {
            double ae = 0, se = 0;
            for (long i = 0; i < nte; i++) {
                float s = 0.f; for (int j = 0; j < p; j++) s += Xte[i * p + j] * w[j];
                if (s < 0) s = 0; double e = expm1((double)s) - dte[i]; ae += fabs(e); se += e * e;
            }
            printf("%s,%d,%d,%.6f,%.4f,%.4f\n", tam, size, r, t, ae / nte, sqrt(se / nte)); fflush(stdout);
        }
    }
    if (!rank && argc > 4) { FILE *fw = fopen(argv[4], "wb"); if (fw) { fwrite(w, 4, p, fw); fclose(fw); } }   /* opcional: guarda pesos */
    MPI_Finalize(); return 0;
}
