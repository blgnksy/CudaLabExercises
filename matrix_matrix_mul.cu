#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>



#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}

__global__ void MatrixMulKernel(float *d_x, float *d_y, float *d_z, int Width) {

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    float kernelSum = 0;

    for(int k = 0; k < Width ; ++k) {
        kernelSum += d_x[idy*Width + k]*d_y[k*Width + idx];
    }

    d_z[idy*Width + idx] = kernelSum;
}

int main() {

    int Width=5;
    int size = Width*Width*sizeof(float);
    float h_x[Width*Width], h_y[Width*Width], h_z[Width*Width];
    float *d_x, *d_y, *d_z;

    for(int i = 0; i < (Width*Width) ; i++) {
        h_x[i] = i;
        h_y[i] = i;
        h_z[i] = 0;
    }




    cudaEvent_t start, stop;
    float elapsed_time_ms;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );


    cudaMalloc((void**)&d_x, size);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_y, size);
    cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_z,size);


    dim3 dimBlock(Width,Width);
    dim3 dimGrid(1,1);

    MatrixMulKernel<<<dimGrid,dimBlock>>>(d_x,d_y,d_z,Width);

    cudaMemcpy(h_z,d_z,size,cudaMemcpyDeviceToHost);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsed_time_ms, start, stop );
    printf("Time to calculate results(GPU Time): %f ms.\n", elapsed_time_ms);

    for(int i = 0; i < (Width*Width) ; i++) {
        printf("%0.2f \n", h_z[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}
