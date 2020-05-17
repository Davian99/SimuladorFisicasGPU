#include <stdio.h>
#include <cuda.h>

#include "gpu_routines.h"

#define CUDA_CALLABLE __device__ __host__

#define BLOCK 32
#define N_COL 20

class deviceCircle {
	public:
		float px, py, radius;
		float vx, vy;
		float force_x, force_y;
		float angularVelocity;
		float mass, inv_mass;
		float inertia, inv_inertia;

		CUDA_CALLABLE deviceCircle(){};
		CUDA_CALLABLE deviceCircle(int px){
			this->px = px;
		};

		CUDA_CALLABLE void applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy){
		    this->vx += this->inv_mass * impulse_x;
		    this->vy += this->inv_mass * impulse_y;
		    this->angularVelocity += this->inv_inertia * ((cvx * impulse_y) - (cvy * impulse_x));
		}
};

__global__ void kernel(deviceCircle * o, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < n){
		printf("I am thread %d, mi circle has px = %f\n", i, o[i].px);
		o[i].px +=  1;
	}
	return;
}

void GPU::initializeContext(){
	//Initialize cuda context
	cudaFree(0);
	printf("Cuda context initialized!\n");
}

GPU::GPU(){
	
	;
	/*
	int n = 10;
	deviceCircle *v_gpu, *v_host;

	//v_host = (deviceCircle *)malloc(sizeof(deviceCircle)*n);

	for(int i = 0; i < n; ++i)
		v_host[i] = deviceCircle(i);

	printf("v_host[3].px = %f\n", v_host[3].px);

	

	//Mallocs GPU
	cudaMalloc((void **) &v_gpu, sizeof(deviceCircle) * n);

	//CPU -> GPU
	cudaMemcpy(v_gpu, v_host, sizeof(deviceCircle) * n, cudaMemcpyHostToDevice);

	//Number of blocks and threads per block
	dim3 dimGrid(1);
	dim3 dimBlock(n);
	//Kernel call
	kernel<<<dimGrid,dimBlock>>>(v_gpu, n);
	cudaDeviceSynchronize();

	//GPU->CPU
	cudaMemcpy(v_host, v_gpu, sizeof(deviceCircle) * n, cudaMemcpyDeviceToHost);

	printf("deviceCircle.px = %f\n", v_host[7].px);

	//Free GPU
	cudaFree(v_gpu);

	delete[] v_host;
	*/
}

