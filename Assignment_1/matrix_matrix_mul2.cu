#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}

__global__ void MatrixMulKernel(int *d_x, int *d_y, int *d_z, int Block_Width, int M , int N) {

    int row = blockIdx.y*blockDim.y+ threadIdx.y;
    int col = blockIdx.x*blockDim.x+ threadIdx.x;

    int kernelSum = 0;
    if ((row<N) && (col<N)) {
        for (int i = 0; i < Block_Width ; ++i) {
            kernelSum+=d_x[col * Block_Width + i] * d_y[i * Block_Width + row];
        }
    }
    d_z[row * Block_Width +col] = kernelSum;
}

int main(void) {

    int M = 5;
    int N = 5;
    int Block_Width=M;
    int h_x[M][N], h_y[N][M], h_z[M][M];
    int *d_x, *d_y, *d_z;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; ++j) {
            h_x[i][j] = 1;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; ++j) {
            h_y[i][j] = 1;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; ++j) {
            h_z[i][j] = 1;
        }
    }
    cudaEvent_t start, stop;
    float elapsed_time_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CUDA_CALL(cudaMalloc((void **) &d_x, (M * N) * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_x, h_x, (M * N) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &d_y, (M * N) * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_y, h_y, (M * N) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &d_z, (M * M) * sizeof(int)));

    dim3 dimBlock(M,N);
    dim3 dimGrid(1, 1);

    MatrixMulKernel << < dimGrid, dimBlock >> > (d_x, d_y, d_z, Block_Width, M, N);

    CUDA_CALL(cudaMemcpy(h_z, d_z, (M * M) * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("Time to calculate results(GPU Time): %f ms.\n", elapsed_time_ms);

    for (int i = 0; i < (M * M); i++) {
        printf("%d\t", *h_z[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}
