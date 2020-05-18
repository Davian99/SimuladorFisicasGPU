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

__global__ void calculateContacs_Kernel(Circle * circles, int n, 
	Collision * colls, int * n_cols) 
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < n){
		Circle ri = circles[i];
		for (unsigned int j = i + 1; j < n; ++j){
			Circle rj = circles[j];
			if(ri.inv_mass == 0.0f && rj.inv_mass == 0.0f)
				continue;
			Collision c;
			c.A = i;
			c.B = j;
			c.normal_x = rj.px - ri.px;
			c.normal_y = rj.py - ri.py;
			float dist = hypot(c.normal_x, c.normal_y);
			float suma_radius = ri.radius + rj.radius;
			if(dist >= suma_radius)
				continue; //Not contact
			
			if(dist <= EPS) {
				c.penetration = ri.radius;
				c.normal_x = 1.0f;
				c.normal_y = 0.0f;
				c.contact_x = ri.px;
				c.contact_y = ri.py;
			}
			else{
				c.penetration = suma_radius - dist;
				c.normal_x /= dist;
				c.normal_y /= dist;
				c.contact_x = c.normal_x * ri.radius + ri.px;
				c.contact_y = c.normal_y * ri.radius + ri.py;
			}
			int idx = atomicAdd(n_cols, 1);
			colls[idx] = c;
		}
	}
}

__global__ void integrateVelocities_Kernel(Circle * circles, int n, float gravity, float dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < n){
		Circle c = circles[i];
		if(c.inv_mass > 0.0f){
			c.px += c.vx * dt;
			c.py += c.vy * dt;
			c.vy += gravity * (dt / 2.0f);
		}
		circles[i] = c;
	}
}

__global__ void integrateForces_Kernel(Circle * circles, int n, float gravity, float dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < n){
		Circle c = circles[i];
		if(c.inv_mass > 0.0f)
			c.vy += gravity * (dt / 2.0f);
		circles[i] = c;
	}
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

void GPU::calculateContact_GPU(vector<Collision> &contacts){
	int n_cols = 0, *n_cols_GPU; 
	cudaMalloc((void **) &n_cols_GPU, sizeof(int));
	cudaMemcpy(n_cols_GPU, &n_cols, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &colls_GPU, sizeof(Collision) * this->lro->size() * 30);

	dim3 dimGrid(ceil((float)this->lro->size() / BLOCK));
	dim3 dimBlock(BLOCK);
	calculateContacs_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), 
		colls_GPU, n_cols_GPU);
	cudaDeviceSynchronize();

	cudaMemcpy(&n_cols, n_cols_GPU, sizeof(int), cudaMemcpyDeviceToHost);
	//printf("N_COLS = %d\n", n_cols);
	contacts.resize(n_cols);
	cudaMemcpy(&(contacts[0]), &colls_GPU[0], 
			sizeof(Collision) * n_cols, cudaMemcpyDeviceToHost);

	cudaFree(colls_GPU);
	cudaFree(n_cols_GPU);
}

void GPU::integrateVelocities_GPU(){
	dim3 dimGrid(ceil((float)this->lro->size() / BLOCK));
	dim3 dimBlock(BLOCK);
	integrateVelocities_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), gravity, dt);
	cudaDeviceSynchronize();
}

void GPU::integrateForces_GPU(){
	dim3 dimGrid(ceil((float)this->lro->size() / BLOCK));
	dim3 dimBlock(BLOCK);
	integrateForces_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), gravity, dt);
	cudaDeviceSynchronize();
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
	this->copy_DeviceToHost();
}

void GPU::copy_HostToDevice(){
	cudaFree(circles_GPU);
	cudaMalloc((void **) &circles_GPU, sizeof(Circle) * this->lro->size());
	cudaMemcpy(&circles_GPU[0], &(this->lro->vro[0]), 
			sizeof(Circle) * this->lro->size(), cudaMemcpyHostToDevice);
}

void GPU::copy_DeviceToHost(){
	cudaMemcpy(&(this->lro->vro[0]), &circles_GPU[0], 
			sizeof(Circle) * this->lro->size(), cudaMemcpyDeviceToHost);
}

GPU::GPU(){
	this->N_GPU_obj = 0;
	this->MAX_GPU_obj = 0;
}

GPU::GPU(ListCircles * list){
	this->lro = list;
	this->N_GPU_obj = 0;
	this->MAX_GPU_obj = 0;
}