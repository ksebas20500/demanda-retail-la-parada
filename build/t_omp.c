#include <omp.h>
#include <stdio.h>
int main(){
#pragma omp parallel
{
#pragma omp single
printf("OpenMP hilos=%d\n", omp_get_num_threads());
}
return 0;}
