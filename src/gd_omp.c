/* Regresion lineal por descenso de gradiente (lote completo), identica a baseline.py.
 * Compilar secuencial:  gcc -O3 -march=native -o gd_seq.exe gd_omp.c -lm
 * Compilar OpenMP:      gcc -O3 -march=native -fopenmp -o gd_omp.exe gd_omp.c -lm
 * Uso: gd_omp <tam> <p> <hilos> <reps> [archivo_pesos]   (lee data/proc/<tam>_{train,test}_{X,y,d}.bin)
 * Salida: una linea CSV  tam,hilos,rep,t_train_s,MAE_test,RMSE_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

#define EPOCHS 300
#define LR 0.05f
#define L2 1e-4f

static double now(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec t; timespec_get(&t, TIME_UTC); return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
}

static float *leer(const char *tam, const char *split, const char *suf, long *n, int cols) {
    char f[256]; snprintf(f, sizeof f, "data/proc/%s_%s_%s.bin", tam, split, suf);
    FILE *fp = fopen(f, "rb"); if (!fp) { fprintf(stderr, "no existe %s\n", f); exit(1); }
    fseek(fp, 0, SEEK_END); long b = ftell(fp); rewind(fp);
    *n = b / (4L * cols); float *a = malloc(b);
    if (fread(a, 1, b, fp) != (size_t)b) { fprintf(stderr, "lectura fallida\n"); exit(1); }
    fclose(fp); return a;
}

static void entrenar(const float *X, const float *y, long n, int p, float *w) {
    double *g = malloc(sizeof(double) * p);
    for (int j = 0; j < p; j++) w[j] = 0.f;
    for (int e = 0; e < EPOCHS; e++) {
        for (int j = 0; j < p; j++) g[j] = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:g[:p])
#endif
        for (long i = 0; i < n; i++) {
            const float *x = X + i * p; float s = 0.f;
            for (int j = 0; j < p; j++) s += x[j] * w[j];
            float r = s - y[i];
            for (int j = 0; j < p; j++) g[j] += (double)r * x[j];
        }
        for (int j = 0; j < p; j++) w[j] -= LR * ((float)(g[j] / n) + L2 * w[j]);
    }
    free(g);
}

int main(int argc, char **argv) {
#ifdef _WIN32   /* evita que Windows 11 relegue el proceso a nucleos eficientes (EcoQoS) y falsee las mediciones */
    { PROCESS_POWER_THROTTLING_STATE st_ = {PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
      SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st_, sizeof st_); }
#endif

    if (argc < 5) { fprintf(stderr, "uso: %s tam p hilos reps\n", argv[0]); return 1; }
    const char *tam = argv[1]; int p = atoi(argv[2]), hilos = atoi(argv[3]), reps = atoi(argv[4]);
#ifdef _OPENMP
    omp_set_num_threads(hilos);
#else
    hilos = 1;
#endif
    long ntr, nte, t1, t2;
    float *Xtr = leer(tam, "train", "X", &ntr, p), *ytr = leer(tam, "train", "y", &t1, 1);
    float *Xte = leer(tam, "test", "X", &nte, p), *dte = leer(tam, "test", "d", &t2, 1);
    float *w = malloc(sizeof(float) * p);
    for (int r = 0; r < reps; r++) {
        double t0 = now(); entrenar(Xtr, ytr, ntr, p, w); double t = now() - t0;
        double ae = 0, se = 0;
        for (long i = 0; i < nte; i++) {
            float s = 0.f; for (int j = 0; j < p; j++) s += Xte[i * p + j] * w[j];
            if (s < 0) s = 0;
            double e = expm1((double)s) - dte[i]; ae += fabs(e); se += e * e;
        }
        printf("%s,%d,%d,%.6f,%.4f,%.4f\n", tam, hilos, r, t, ae / nte, sqrt(se / nte));
    }
    if (argc > 5) { FILE *fw = fopen(argv[5], "wb"); if (fw) { fwrite(w, 4, p, fw); fclose(fw); } }   /* opcional: guarda pesos */
    return 0;
}
