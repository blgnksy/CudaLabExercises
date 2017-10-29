#include <ctime>
#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define M 12
#define N 6

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
	printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
	printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
	exit(EXIT_FAILURE); }}

void matrixSumGPU(float* h_z, const float* h_x, const float* h_y);
void matrixSumCPU(float h_z[M][N], float h_x[M][N], float h_y[M][N]);

__global__ void matrixSumKernel(float d_z[][N], float d_x[][N], float d_y[][N])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	d_z[idx][idy] = d_x[idx][idy] + d_y[idx][idy];

}

void matrixSumCPU(float h_z[M][N], float h_x[M][N], float h_y[M][N])
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			h_z[i][j] = h_x[i][j] + h_y[i][j];
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%0.2f ms in CPU.\n", elapsedTime);
}

void matrixSumGPU(float* h_z, const float* h_x, const float* h_y)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float(*d_x)[N], (*d_y)[N], (*d_z)[N];

	CUDA_CALL(cudaMalloc((void**)&d_x, (M*N) * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_y, (M*N) * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_z, (M*N) * sizeof(int)));

	CUDA_CALL(cudaMemcpy(d_x, h_x, (M*N) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_y, h_y, (M*N) * sizeof(int), cudaMemcpyHostToDevice));

	int numBlocks = 1;
	dim3 threadsPerBlock(M, N);
	matrixSumKernel << <numBlocks, threadsPerBlock >> >(d_z, d_x, d_y);

	CUDA_CALL(cudaMemcpy(h_z, d_z, (M*N) * sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(d_x));
	CUDA_CALL(cudaFree(d_y));
	CUDA_CALL(cudaFree(d_z));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%0.2f ms in GPU.\n", elapsedTime);
}

int main() {

	float h_x[M][N], h_y[M][N], h_z[M][N];

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			h_z[i][j] = 0;
			h_x[i][j] = 1;
			h_y[i][j] = 2;
		}
	}


	matrixSumGPU(*h_z, *h_x, *h_y);

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f\t", h_z[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	matrixSumCPU(h_z, h_x, h_y);

	printf("Press enter key to return...");
	getchar();
	return 0;
}
