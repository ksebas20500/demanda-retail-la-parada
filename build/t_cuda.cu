#include <cstdio>
__global__ void k(int*a){a[threadIdx.x]=threadIdx.x*2;}
int main(){int*d,h[8];cudaMalloc(&d,32);k<<<1,8>>>(d);cudaMemcpy(h,d,32,cudaMemcpyDeviceToHost);
cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf("CUDA %s cc%d.%d h[7]=%d\n",p.name,p.major,p.minor,h[7]);return 0;}
