#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

//You can change the dimension, program will produce two matrices.
#define M 600
#define N 800

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}

__global__ void matrixAdd(int d_x[][N], int d_y[][N], int d_z[][N]) {
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    if (idx < M && idy < N) {
        d_z[idx][idy] = d_x[idx][idy] + d_y[idx][idy];
    }
}

int main() {

    int size = (M * N) * sizeof(int);

    int h_x[M][N], h_y[M][N], h_z[M][N];
    int(*d_x)[N], (*d_y)[N], (*d_z)[N];

    int i = 0;
    int j = 0;

    //Initialize matrix
    for (i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_x[i][j] = M;
            h_y[i][j] = N;
            h_z[i][j] = 0;
        }
    }

    cudaEvent_t startC, stopC;
    float elapsed_time_msC;
    cudaEventCreate( &startC );
    cudaEventCreate( &stopC );
    cudaEventRecord( startC, 0 );
    for (i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_z[i][j] =h_x[i][j] + h_y[i][j] ;
        }
    }
    cudaEventRecord( stopC, 0 );
    cudaEventSynchronize( stopC );
    cudaEventElapsedTime( &elapsed_time_msC, startC, stopC );
    printf("Time to calculate results(CPU Time): %f ms.\n", elapsed_time_msC);

    CUDA_CALL(cudaMalloc(&d_x, size));
    CUDA_CALL(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_y, size));
    CUDA_CALL(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_z, size));


    dim3 dimGrid(1, 1);
    dim3 dimBlock(M, N);

    cudaEvent_t start, stop;
    float elapsed_time_ms;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    matrixAdd <<< dimGrid, dimBlock >>> (d_x, d_y, d_z);
    CUDA_CALL(cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost));

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsed_time_ms, start, stop );
    printf("Time to calculate results(GPU Time): %f ms.\n", elapsed_time_ms);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    printf("Output of Summation\n");
//    for (i = 0; i<M; i++) {
//        for (j = 0; j<N; j++) {
//            printf("%d\t", h_z[i][j]);
//        }
//        printf("\n");
//    }
    printf("\n");
}
