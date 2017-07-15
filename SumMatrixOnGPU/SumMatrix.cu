#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#define CHECK(call)										\
{														\
	const cudaError_t error = call;						\
	if (error != cudaSuccess)							\
	{													\
		printf("error: %s:%d", __FILE__, __LINE__);		\
		exit(-10*error);								\
	}													\
}														\

void initialInt(int *ip, int size)
{
	for (int i = 0; i < size; i++)
		ip[i] = i;
}

void printMatrix(int *C, const int nx, const int ny)
{
	int *ic = C;
	printf("\nMatrix:(%d, %d)\n", nx, ny);
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%3d", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
		"global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
		blockIdx.y, ix, iy, idx, A[idx]);
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	if(ix < nx && iy < ny)
		MatC[idx] = MatA[idx] + MatB[idx];
}

void sumMatrixOnHost(int* h_A, int* h_B, int *hostRef, int nx, int ny)
{
	for (int ix = 0; ix < nx; ix++)
	{
		for (int iy = 0; iy < ny; iy++)
		{
			int ind = iy * nx + iy;
			hostRef[ind] = h_A[ind] + h_B[ind];
		}
	}
}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
	double epsilon = 1.0e-8;
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i] > epsilon)) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match, \n\n");
}

int main(int argc, char**argv)
{
	printf("%s Starting...\n", argv[0]);

	// get device information
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set matrix dimension
	int nx = 1<<14;
	int ny = 1<<14;
	int nxy = nx*ny;
	int nBytes = nxy * sizeof(float);

	// malloc host memory
	int *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (int *)malloc(nBytes);
	h_B = (int *)malloc(nBytes);
	hostRef = (int *)malloc(nBytes);
	gpuRef = (int *)malloc(nBytes);

	// initial data
	initialInt(h_A, nxy);
	initialInt(h_B, nxy);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// add matrix at host side
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

	// malloc device global memory
	float *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void**)&d_MatA, nBytes);
	cudaMalloc((void**)&d_MatB, nBytes);	
	cudaMalloc((void**)&d_MatC, nBytes);
	
	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
	// invoke kernel
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	sumMatrixOnGPU2D << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
	cudaDeviceSynchronize();

	cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

	// check
	checkResult(hostRef, gpuRef, nxy);

	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	cudaDeviceReset();
	

	return (0);
}