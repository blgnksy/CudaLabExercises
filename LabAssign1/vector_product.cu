#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>


__global__ void vecProduct(int *d_x, int *d_y, int *d_z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_z[idx] = d_x[idx] * d_y[idx];
    }
}

int main() {
    int N=0;

    printf("%s","Enter the size of vector : ");

    if( scanf( "%d", &N) == 0 )
    {
        fprintf( stderr, "Expected a positive number as input\n");
        exit(1);
    }
    int size = N * sizeof(int);
    int h_x[N], h_y[N], h_z[N], *d_x, *d_y, *d_z;
    int i = 0;

    int total=0;

    //Initialize vectors
    for (i = 0; i < N; i++) {
        h_x[i] = i;
        h_y[i] = i;
        h_z[i] = 0;
    }

    cudaEvent_t startC, stopC;
    float elapsed_time_msC;
    cudaEventCreate( &startC );
    cudaEventCreate( &stopC );
    cudaEventRecord( startC, 0 );
    for (i = 0; i < N; i++) {
        h_z[i] =h_x[i]+h_y[i] ;
    }
    cudaEventRecord( stopC, 0 );
    cudaEventSynchronize( stopC );
    cudaEventElapsedTime( &elapsed_time_msC, startC, stopC );
    printf("Time to calculate results(CPU Time): %f ms.\n", elapsed_time_msC);

    cudaMalloc(&d_x, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_y, size);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_z, size);

    dim3 dimGrid(1, 1);
    dim3 dimBlock(N, 1);

    cudaEvent_t start, stop;
    float elapsed_time_ms;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    vecProduct <<<  dimGrid, dimBlock >>> (d_x, d_y, d_z, N);
    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsed_time_ms, start, stop );

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    for (i = 0; i < N; i++) {
        total+= h_z[i];
    }
    printf("Time to calculate results(GPU Time): %f ms.\n", elapsed_time_ms);
}
