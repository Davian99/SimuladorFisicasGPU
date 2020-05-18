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

__global__ void positionalCorrection_Kernel(Circle * circles, int n, 
	Collision * colls, int * n_cols) 
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < (*n_cols)){
		Collision c = colls[idx];
		Circle A = circles[c.A];
		Circle B = circles[c.B];

		float k_slop = 0.05f; // Penetration allowance
		float percent = 0.4f; // Penetration percentage to correct
		float corr_x = (max(c.penetration - k_slop, 0.0f) / (A.inv_mass + B.inv_mass)) * c.normal_x * percent;
		float corr_y = (max(c.penetration - k_slop, 0.0f) / (A.inv_mass + B.inv_mass)) * c.normal_y * percent;

		atomicAdd(&circles[c.A].px, (-1.0f) * (corr_x * A.inv_mass));
		atomicAdd(&circles[c.A].py, (-1.0f) * (corr_y * A.inv_mass));
		atomicAdd(&circles[c.B].px, (corr_x * B.inv_mass));
		atomicAdd(&circles[c.B].py, (corr_y * B.inv_mass));
	}
}

__global__ void solveCollisions_Kernel(Circle * circles, int n, 
	Collision * colls, int * n_cols, int iterations, float gravity, float dt) 
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < (*n_cols)){
		Collision c = colls[idx];
		Circle A = circles[c.A];
		Circle B = circles[c.B];
		float Avx = A.vx, Avy = A.vy, AvA = A.angularVelocity;
		float Bvx = B.vx, Bvy = B.vy, BvA = B.angularVelocity;

		if(A.mass + B.mass < EPS) {
	    	A.vx = 0.0f;
	    	A.vy = 0.0f;
	    	B.vx = 0.0f;
	    	B.vy = 0.0f;
	    	return;
	  	}

	  	float rax = c.contact_x - A.px;
	  	float ray = c.contact_y - A.py;
	  	float rbx = c.contact_x - B.px;
	  	float rby = c.contact_y - B.py;

	  	float rvx = Bvx - (BvA * rby) - Avx + (AvA * ray); //ERROR POSIBLE
	  	float rvy = Bvy + (BvA * rbx) - Avy - (AvA * rax);

	  	float contact_vel = rvx * c.normal_x + rvy * c.normal_y;
	    if(contact_vel > 0.0f)
	    	return;

	  	float raCrossN = (rax * c.normal_y) - (ray * c.normal_x);
	  	float rbCrossN = (rbx * c.normal_y) - (rby * c.normal_x);

	  	float invMassSum = A.inv_mass + B.inv_mass + raCrossN*raCrossN * A.inv_inertia + rbCrossN*rbCrossN * B.inv_inertia; 
	  	
	  	float e = 0.2f;
	  	if((rvx * rvx + rvy * rvy) < ((dt * gravity * dt * gravity) + EPS))
	  		e = 0.0f;

	  	float j = -(1.0f + e) * contact_vel;
	  	j /= invMassSum;

	  	float impulse_x = c.normal_x * j;
	  	float impulse_y = c.normal_y * j;

	  	Avx += A.inv_mass * (-impulse_x);
		Avy += A.inv_mass * (-impulse_y);
		AvA += A.inv_inertia * ((rax * (-impulse_y)) - (ray * (-impulse_x)));

	  	Bvx += B.inv_mass * (impulse_x);
		Bvy += B.inv_mass * (impulse_y);
		BvA += B.inv_inertia * ((rbx * (impulse_y)) - (rby * (impulse_x)));

		atomicAdd(&circles[c.A].vx, (Avx - A.vx));
		atomicAdd(&circles[c.A].vy, (Avy - A.vy));
		atomicAdd(&circles[c.A].angularVelocity, (AvA - A.angularVelocity));

		atomicAdd(&circles[c.B].vx, (Bvx - B.vx));
		atomicAdd(&circles[c.B].vy, (Bvy - B.vy));
		atomicAdd(&circles[c.B].angularVelocity, (BvA - B.angularVelocity));
	}
}

__global__ void calculateContacs_Kernel(Circle * circles, int n, 
	Collision * colls, int * n_cols) 
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int bloc = blockDim.x * blockIdx.x;

	__shared__ Circle sh_Circles[BLOCK];
	Circle ri;
	if(i < n)
		ri = circles[i];

	for (unsigned int j = bloc; j < n; j += BLOCK){
		if(j + threadIdx.x < n)
			sh_Circles[threadIdx.x] = circles[j + threadIdx.x];

		__syncthreads();

		for(int k = 0; k < BLOCK && j + k < n; ++k){
			if(j + k <= i)
				continue;
			Circle rj = sh_Circles[k];
			if(ri.inv_mass == 0.0f && rj.inv_mass == 0.0f)
				continue;
			Collision c;
			c.A = i;
			c.B = j + k;
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

void GPU::positionalCorrection_GPU(){
	dim3 dimGrid(ceil((float)this->n_cols / BLOCK));
	dim3 dimBlock(BLOCK);

	positionalCorrection_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), 
		colls_GPU, this->n_cols_GPU);
	cudaDeviceSynchronize();

	cudaFree(colls_GPU);
	cudaFree(n_cols_GPU);
}

void GPU::solveCollisions_GPU(vector<Collision> &contacts){

	dim3 dimGrid(ceil((float)this->n_cols / BLOCK));
	dim3 dimBlock(BLOCK);
	for(int i = 0; i < iterations; ++i){
		solveCollisions_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), 
			colls_GPU, this->n_cols_GPU, iterations, gravity, dt);
		cudaDeviceSynchronize();
	}
	/*
	contacts.resize(n_cols);
	cudaMemcpy(&(contacts[0]), &colls_GPU[0], 
			sizeof(Collision) * this->n_cols, cudaMemcpyDeviceToHost);

	cudaFree(colls_GPU);
	cudaFree(n_cols_GPU);
	*/
}

void GPU::calculateContact_GPU(vector<Collision> &contacts){
	this->n_cols = 0; 
	cudaMalloc((void **) &this->n_cols_GPU, sizeof(int));
	cudaMemcpy(this->n_cols_GPU, &this->n_cols, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &colls_GPU, sizeof(Collision) * this->lro->size() * 30);

	dim3 dimGrid(ceil((float)this->lro->size() / BLOCK));
	dim3 dimBlock(BLOCK);
	calculateContacs_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), 
		colls_GPU, this->n_cols_GPU);
	cudaDeviceSynchronize();

	cudaMemcpy(&this->n_cols, this->n_cols_GPU, sizeof(int), cudaMemcpyDeviceToHost);
	//printf("N_COLS = %d\n", n_cols);
	
	contacts.resize(n_cols);
	cudaMemcpy(&(contacts[0]), &colls_GPU[0], 
			sizeof(Collision) * this->n_cols, cudaMemcpyDeviceToHost);
	/*
	cudaFree(colls_GPU);
	cudaFree(n_cols_GPU);
	*/
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