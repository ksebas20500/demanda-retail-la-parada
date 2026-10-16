#include <mpi.h>
#include <stdio.h>
int main(int c,char**v){MPI_Init(&c,&v);int r,s;MPI_Comm_rank(MPI_COMM_WORLD,&r);MPI_Comm_size(MPI_COMM_WORLD,&s);printf("MPI rank %d/%d\n",r,s);MPI_Finalize();return 0;}
