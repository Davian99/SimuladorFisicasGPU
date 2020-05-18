#include <stdio.h>

#include "gpu_routines.h"

#include <cuda.h>

#define BLOCK 32
#define N_COL 20

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void kernel(Circle * o, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	printf("I am thread %d\n", i);
	if(i < n){
		o[i].px += 1;
		printf("I am thread %d, mi circle has mass = %f\n", i, o[i].mass);
	}
	return;
}

__global__ void print_circle_kernel(Circle * o, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < n){
		printf("Circle[%d] in (%f, %f)\n", i, o[i].px, o[i].py);
	}
}

__global__ void init_context_kernel() {
	printf("Cuda context initialized!\n");
}

void GPU::initializeContext(){
	//Initialize cuda contex
	init_context_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
}

void GPU::update_mem(){
	if(true || this->lro->size() > this->N_GPU_obj){
		//Hay objetos que faltan en GPU
		int init = 0;//this->N_GPU_obj;
		int num_copy = this->lro->size();// - this->N_GPU_obj;
		if(this->lro->size() > this->MAX_GPU_obj){ //Si no caben en GPU
			init = 0;
			num_copy = this->lro->size();
			this->MAX_GPU_obj = 2 * this->lro->size();

			cudaFree(circles_GPU); //Liberamos la memoria existente
			cudaMalloc((void **) &circles_GPU, sizeof(Circle) * this->MAX_GPU_obj);
		}

		//CPU -> GPU
		//printf("COPIADO desde %d con n = %d\n", init, num_copy);
		cudaMemcpy(&circles_GPU[init], &(this->lro->vro[init]), 
			sizeof(Circle) * num_copy, cudaMemcpyHostToDevice);

		this->N_GPU_obj = this->lro->size();
	}

	cudaMemcpy(&(this->lro->vro[0]), &circles_GPU[0], 
			sizeof(Circle) * this->lro->size(), cudaMemcpyDeviceToHost);

	/*
	print_circle_kernel<<<1, this->N_GPU_obj>>>(this->circles_GPU, this->N_GPU_obj);
	cudaDeviceSynchronize();
	for (auto p : this->lro->vro){
		printf("Circle in (%f, %f)\n", p.px, p.py);
	}
	printf("FIN CIRCLES\n");
	*/
}

GPU::GPU(){
	this->N_GPU_obj = 0;
	this->MAX_GPU_obj = 0;
}

GPU::GPU(ListCircles * list){
	this->lro = list;
	this->N_GPU_obj = 0;
	this->MAX_GPU_obj = 0;
	/*
	int n = 10;
	Circle *v_gpu, *v_host;
	v_host = (Circle *)malloc(sizeof(Circle)*n);
	for(int i = 0; i < n; ++i)
		v_host[i] = Circle(100 * i, 100, 100, false);
	printf("v_host[3].mass = %f\n", v_host[3].mass);

	
	//Mallocs GPU
	cudaMalloc((void **) &v_gpu, sizeof(Circle) * n);
	//CPU -> GPU
	cudaMemcpy(v_gpu, v_host, sizeof(Circle) * n, cudaMemcpyHostToDevice);
	//Number of blocks and threads per block
	dim3 dimGrid(1);
	dim3 dimBlock(n);
	//Kernel call
	kernel<<<dimGrid,dimBlock>>>(v_gpu, n);
	cudaDeviceSynchronize();
	//GPU->CPU
	cudaMemcpy(v_host, v_gpu, sizeof(Circle) * n, cudaMemcpyDeviceToHost);
	printf("Circle.mass = %f\n", v_host[7].mass);
	//Free GPU
	cudaFree(v_gpu);
	//Free host
	free(v_host);
	*/
}