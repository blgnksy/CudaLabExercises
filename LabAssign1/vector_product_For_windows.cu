#include <ctime>
#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
	printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
	printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
	exit(EXIT_FAILURE); }}

void vecProductGPU(float *h_z, const float *h_x, const float *h_y, unsigned int N);

void vecProductCPU(float *h_z, const float *h_x, const float *h_y, unsigned int N);

__global__ void vecProductKernel(float *d_z, const float *d_x, const float *d_y, unsigned int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		d_z[idx] = d_x[idx] * d_y[idx];
	}
}

int main()
{
	const int N = 700;
	int i = 0;
	float h_x[N], h_y[N], h_z[N];
	for (i = 0; i < N; i++)
	{
		h_x[i] = float(i + 0.5f);
		h_y[i] = float(i + 1);
		h_z[i] = float(0);
	}

	vecProductGPU(h_z, h_x, h_y, N);
	
	for (i = 0; i < N; i++)
	{
		printf("%.2f\t", h_z[i]);
	}
	printf("\n");

	vecProductCPU(h_z, h_x, h_y, N);

	for (i = 0; i < N; i++)
	{
		printf("%.2f\t", h_z[i]);
	}
	printf("\n");
	
	printf("Press enter key to return...");
	getchar();
	return 0;
}

void vecProductGPU(float *h_z, const float *h_x, const float *h_y, unsigned int N)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float *d_x, *d_y, *d_z;
	CUDA_CALL(cudaMalloc((void**)&d_x, N * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_y, N * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_z, N * sizeof(float)));

	CUDA_CALL(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

	vecProductKernel << <1, N >> >(d_z, d_x, d_y, N);
	
	CUDA_CALL(cudaMemcpy(h_z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(d_x));
	CUDA_CALL(cudaFree(d_y));
	CUDA_CALL(cudaFree(d_z));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%0.2f ms in GPU.\n", elapsedTime);
}

void vecProductCPU(float *h_z, const float *h_x, const float *h_y, unsigned int N)
{
	clock_t begin = clock();

	for (int i = 0; i < N; i++)
	{
		h_z[i] = h_x[i] * h_y[i];
	}

	clock_t end = clock();
	double elapsedTime = double(end - begin) / CLOCKS_PER_SEC;
	printf("%0.2f ms in CPU.\n", elapsedTime);
}
