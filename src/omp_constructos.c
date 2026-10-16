/* Constructos de OpenMP (Unidad 2) aplicados al gradiente de la regresion lineal del proyecto.
 *  demo "data_sharing" : parallel + default(none)/shared/private, reduction vs critical vs atomic vs CARRERA (sin proteccion)
 *  demo "schedule_*"   : for + schedule(static|dynamic|guided, chunk) con carga uniforme e irregular
 *  demo "sections"     : evaluar val / test / train en paralelo con sections
 *  demo "single_region": una sola region parallel + single (actualiza pesos) vs parallel for por epoca (fork/join)
 *  demo "tasks"        : parallel + single + task por bloque de filas
 * Compilar: gcc -O3 -march=native -fopenmp -o omp_constructos.exe omp_constructos.c -lm
 * Uso: omp_constructos <tam> <p> <reps> <pasadas>
 * Salida CSV: demo,variante,hilos,rep,t_s,error,nota
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define EPOCHS 300
#define LR 0.05f
#define L2 1e-4f
#define KMAX 300

static float *leer(const char *tam, const char *split, const char *suf, long *n, int cols) {
    char f[256]; snprintf(f, sizeof f, "data/proc/%s_%s_%s.bin", tam, split, suf);
    FILE *fp = fopen(f, "rb"); if (!fp) { fprintf(stderr, "no existe %s\n", f); exit(1); }
    fseek(fp, 0, SEEK_END); long b = ftell(fp); rewind(fp); *n = b / (4L * cols);
    float *a = malloc(b); if (fread(a, 1, b, fp) != (size_t)b) exit(1); fclose(fp); return a;
}

static inline float extra(const float *x, long i, long n) {   /* trabajo extra creciente con i (carga irregular) */
    int k = 1 + (int)(KMAX * (double)i / (double)n); float a = x[0];
    for (int t = 0; t < k; t++) a = a * 1.0000001f + x[t & 7];
    return a;
}

static double rel(const double *g, const double *r, int p) {
    double a = 0, b = 0; for (int j = 0; j < p; j++) { a += (g[j] - r[j]) * (g[j] - r[j]); b += r[j] * r[j]; }
    return sqrt(a / b);
}

#define ROW(i, irr)  const float *x = X + (i) * p; float s = 0.f; \
    for (int j = 0; j < p; j++) s += x[j] * w[j]; \
    float r = s - y[i]; if (irr) r += extra(x, (i), n) * 1e-30f;

static void g_seq(const float *X, const float *y, const float *w, long n, int p, double *g, int irr) {
    for (int j = 0; j < p; j++) g[j] = 0;
    for (long i = 0; i < n; i++) { ROW(i, irr); for (int j = 0; j < p; j++) g[j] += (double)r * x[j]; }
}
static void g_reduction(const float *X, const float *y, const float *w, long n, int p, double *g, int irr, int nth) {
    for (int j = 0; j < p; j++) g[j] = 0;
#pragma omp parallel for num_threads(nth) schedule(runtime) reduction(+:g[:p])
    for (long i = 0; i < n; i++) { ROW(i, irr); for (int j = 0; j < p; j++) g[j] += (double)r * x[j]; }
}
static void g_critical(const float *X, const float *y, const float *w, long n, int p, double *g, int nth) {
    for (int j = 0; j < p; j++) g[j] = 0;
#pragma omp parallel num_threads(nth) default(none) shared(X, y, w, n, p, g)
    {
        double *loc = calloc(p, sizeof(double));            /* privado: declarado dentro de la region */
#pragma omp for schedule(static) nowait
        for (long i = 0; i < n; i++) { ROW(i, 0); for (int j = 0; j < p; j++) loc[j] += (double)r * x[j]; }
#pragma omp critical(fusion)
        for (int j = 0; j < p; j++) g[j] += loc[j];
        free(loc);
    }
}
static void g_atomic(const float *X, const float *y, const float *w, long n, int p, double *g, int nth) {
    for (int j = 0; j < p; j++) g[j] = 0;
#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(X, y, w, n, p, g)
    for (long i = 0; i < n; i++) {
        ROW(i, 0);
        for (int j = 0; j < p; j++) {
#pragma omp atomic
            g[j] += (double)r * x[j];
        }
    }
}
static void g_carrera(const float *X, const float *y, const float *w, long n, int p, double *g, int nth) {
    for (int j = 0; j < p; j++) g[j] = 0;
#pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(X, y, w, n, p, g)
    for (long i = 0; i < n; i++) { ROW(i, 0); for (int j = 0; j < p; j++) g[j] += (double)r * x[j]; }   /* CARRERA DE DATOS a proposito */
}
static void g_tasks(const float *X, const float *y, const float *w, long n, int p, double *g, long B, int nth) {
    for (int j = 0; j < p; j++) g[j] = 0;
#pragma omp parallel num_threads(nth) default(none) shared(X, y, w, n, p, g, B)
    {
#pragma omp single
        for (long b0 = 0; b0 < n; b0 += B) {
            long b1 = b0 + B < n ? b0 + B : n;
#pragma omp task default(none) firstprivate(b0, b1) shared(X, y, w, n, p, g)
            {
                double *loc = calloc(p, sizeof(double));
                for (long i = b0; i < b1; i++) { ROW(i, 0); for (int j = 0; j < p; j++) loc[j] += (double)r * x[j]; }
#pragma omp critical(fusion_t)
                for (int j = 0; j < p; j++) g[j] += loc[j];
                free(loc);
            }
        }
    }
}

static void train_forkjoin(const float *X, const float *y, long n, int p, float *w, int nth) {
    double *g = malloc(p * sizeof(double)); for (int j = 0; j < p; j++) w[j] = 0;
    for (int e = 0; e < EPOCHS; e++) {
        for (int j = 0; j < p; j++) g[j] = 0;
#pragma omp parallel for num_threads(nth) schedule(static) reduction(+:g[:p])
        for (long i = 0; i < n; i++) { ROW(i, 0); for (int j = 0; j < p; j++) g[j] += (double)r * x[j]; }
        for (int j = 0; j < p; j++) w[j] -= LR * ((float)(g[j] / n) + L2 * w[j]);
    }
    free(g);
}
static void train_region(const float *X, const float *y, long n, int p, float *w, int nth) {
    double *g = calloc(p, sizeof(double)); for (int j = 0; j < p; j++) w[j] = 0;
#pragma omp parallel num_threads(nth) default(none) shared(X, y, n, p, w, g)
    {
        for (int e = 0; e < EPOCHS; e++) {
#pragma omp for schedule(static) reduction(+:g[:p])
            for (long i = 0; i < n; i++) { ROW(i, 0); for (int j = 0; j < p; j++) g[j] += (double)r * x[j]; }
#pragma omp single
            {   /* un solo hilo actualiza los pesos y registra el progreso; barrera implicita al final */
                for (int j = 0; j < p; j++) { w[j] -= LR * ((float)(g[j] / n) + L2 * w[j]); g[j] = 0; }
                if (e % 100 == 0) fprintf(stderr, "  [single] epoca %d hilo %d\n", e, omp_get_thread_num());
            }
        }
    }
    free(g);
}

static void metr(const float *Xs, const float *ds, long m, int p, const float *w, double *mae, double *rmse) {
    double ae = 0, se = 0;
    for (long i = 0; i < m; i++) { float s = 0; for (int j = 0; j < p; j++) s += Xs[i * p + j] * w[j]; if (s < 0) s = 0;
        double e = expm1((double)s) - ds[i]; ae += fabs(e); se += e * e; }
    *mae = ae / m; *rmse = sqrt(se / m);
}
static double mse_log(const float *X, const float *y, long m, int p, const float *w) {
    double a = 0; for (long i = 0; i < m; i++) { float s = 0; for (int j = 0; j < p; j++) s += X[i * p + j] * w[j]; a += (double)(s - y[i]) * (s - y[i]); }
    return a / m;
}

static void emit(const char *demo, const char *var, int th, int rep, double t, double err, const char *nota) {
    printf("%s,%s,%d,%d,%.6f,%.3e,%s\n", demo, var, th, rep, t, err, nota); fflush(stdout);
}

int main(int argc, char **argv) {
#ifdef _WIN32   /* evita que Windows 11 relegue el proceso a nucleos eficientes (EcoQoS) y falsee las mediciones */
    { PROCESS_POWER_THROTTLING_STATE st_ = {PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
      SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st_, sizeof st_); }
#endif

    if (argc < 5) { fprintf(stderr, "uso: omp_constructos tam p reps pasadas\n"); return 1; }
    const char *tam = argv[1]; int p = atoi(argv[2]), reps = atoi(argv[3]), P = atoi(argv[4]);
    long n, t1, nv, nte, t2, t3;
    float *X = leer(tam, "train", "X", &n, p), *y = leer(tam, "train", "y", &t1, 1);
    float *Xv = leer(tam, "val", "X", &nv, p), *dv = leer(tam, "val", "d", &t3, 1);
    float *Xt = leer(tam, "test", "X", &nte, p), *dt = leer(tam, "test", "d", &t2, 1);
    float *w = malloc(4 * p); for (int j = 0; j < p; j++) w[j] = 0.01f * sinf(j + 1.f);
    double *g = malloc(8 * p), *gr = malloc(8 * p), *gri = malloc(8 * p);
    g_seq(X, y, w, n, p, gr, 0); g_seq(X, y, w, n, p, gri, 1);

    /* 1. data sharing y sincronizacion */
    int HS[] = {2, 6, 16};
    for (int hi = 0; hi < 3; hi++) { int th = HS[hi];
        omp_set_schedule(omp_sched_static, 0);
        for (int v = 0; v < 4; v++) {
            const char *nm[] = {"reduction", "critical", "atomic", "carrera_sin_proteccion"};
            for (int r = -1; r < reps; r++) {
                double t0 = omp_get_wtime();
                for (int k = 0; k < P; k++) {
                    if (v == 0) g_reduction(X, y, w, n, p, g, 0, th); else if (v == 1) g_critical(X, y, w, n, p, g, th);
                    else if (v == 2) g_atomic(X, y, w, n, p, g, th); else g_carrera(X, y, w, n, p, g, th);
                }
                double t = omp_get_wtime() - t0; if (r >= 0) emit("data_sharing", nm[v], th, r, t / P, rel(g, gr, p), "error=||g-g_seq||/||g_seq||");
            }
        }
    }
    /* 2. schedule: uniforme (irr=0) e irregular (irr=1) */
    struct { omp_sched_t k; int c; const char *nm; } S[] = {
        {omp_sched_static, 0, "static"}, {omp_sched_static, 1, "static-1"}, {omp_sched_static, 64, "static-64"},
        {omp_sched_dynamic, 64, "dynamic-64"}, {omp_sched_dynamic, 1024, "dynamic-1024"}, {omp_sched_dynamic, 16384, "dynamic-16384"},
        {omp_sched_guided, 64, "guided-64"}};
    int HS2[] = {6, 16};
    for (int irr = 0; irr < 2; irr++) for (int hi = 0; hi < 2; hi++) for (int s = 0; s < 7; s++) {
        omp_set_schedule(S[s].k, S[s].c);
        for (int r = -1; r < reps; r++) {
            double t0 = omp_get_wtime();
            for (int k = 0; k < (irr ? 5 : P); k++) g_reduction(X, y, w, n, p, g, irr, HS2[hi]);
            double t = (omp_get_wtime() - t0) / (irr ? 5 : P);
            if (r >= 0) emit(irr ? "schedule_irregular" : "schedule_uniforme", S[s].nm, HS2[hi], r, t, rel(g, irr ? gri : gr, p), irr ? "carga creciente con la fila" : "carga uniforme");
        }
    }
    omp_set_schedule(omp_sched_static, 0);
    /* 3. sections */
    for (int r = -1; r < reps; r++) {
        double mv, rv, mt, rt, tr, mv2, rv2, mt2, rt2, tr2; double t0 = omp_get_wtime();
        for (int k = 0; k < P; k++) { metr(Xv, dv, nv, p, w, &mv, &rv); metr(Xt, dt, nte, p, w, &mt, &rt); tr = mse_log(X, y, n, p, w); }
        double ts = (omp_get_wtime() - t0) / P; t0 = omp_get_wtime();
        for (int k = 0; k < P; k++) {
#pragma omp parallel sections num_threads(3)
            {
#pragma omp section
                metr(Xv, dv, nv, p, w, &mv2, &rv2);
#pragma omp section
                metr(Xt, dt, nte, p, w, &mt2, &rt2);
#pragma omp section
                tr2 = mse_log(X, y, n, p, w);
            }
        }
        double tp = (omp_get_wtime() - t0) / P;
        if (r >= 0) { emit("sections", "secuencial", 1, r, ts, 0, "val+test+train"); emit("sections", "parallel_sections", 3, r, tp, fabs(mv - mv2) + fabs(mt - mt2) + fabs(tr - tr2), "3 sections"); }
    }
    /* 4. single: region unica vs fork/join por epoca */
    float *wa = malloc(4 * p), *wb = malloc(4 * p), *wc = malloc(4 * p);
    train_forkjoin(X, y, n, p, wc, 1);
    int HS3[] = {2, 6, 16};
    for (int hi = 0; hi < 3; hi++) for (int r = -1; r < reps; r++) {
        double t0 = omp_get_wtime(); train_forkjoin(X, y, n, p, wa, HS3[hi]); double ta = omp_get_wtime() - t0;
        t0 = omp_get_wtime(); train_region(X, y, n, p, wb, HS3[hi]); double tb = omp_get_wtime() - t0;
        double da = 0, db = 0; for (int j = 0; j < p; j++) { da = fmax(da, fabs(wa[j] - wc[j])); db = fmax(db, fabs(wb[j] - wc[j])); }
        if (r >= 0) { emit("single_region", "parallel_for_por_epoca", HS3[hi], r, ta, da, "error=max|w-w_1hilo|"); emit("single_region", "region_unica+single", HS3[hi], r, tb, db, "error=max|w-w_1hilo|"); }
    }
    /* 5. tasks */
    long BS[] = {1024, 16384, 65536};
    for (int b = 0; b < 3; b++) for (int r = -1; r < reps; r++) {
        double t0 = omp_get_wtime(); for (int k = 0; k < P; k++) g_tasks(X, y, w, n, p, g, BS[b], 6);
        double t = (omp_get_wtime() - t0) / P; char nm[64]; snprintf(nm, sizeof nm, "task_bloque_%ld", BS[b]);
        if (r >= 0) emit("tasks", nm, 6, r, t, rel(g, gr, p), "parallel+single+task");
    }
    return 0;
}
