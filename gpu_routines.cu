#include <stdio.h>

#include "gpu_routines.h"

#include <cuda.h>

#define BLOCK 32
#define N_COL 30

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void positionalCorrection_Kernel(Circle * circles, int n, 
	Collision * colls, unsigned int * n_cols) 
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < (*n_cols)){
		Collision c = colls[idx];
		Circle A = circles[c.A];
		Circle B = circles[c.B];

		float k_slop = 0.05f; // Penetration allowance
		//float percent = 0.4f; // Penetration percentage to correct
		float corr_aux = 0.4f * (max(c.penetration - k_slop, 0.0f) / (A.inv_mass + B.inv_mass));
		float corr_x = corr_aux * c.normal_x;
		float corr_y = corr_aux * c.normal_y;

		atomicAdd(&circles[c.A].px, (-1.0f) * (corr_x * A.inv_mass));
		atomicAdd(&circles[c.A].py, (-1.0f) * (corr_y * A.inv_mass));
		atomicAdd(&circles[c.B].px, (corr_x * B.inv_mass));
		atomicAdd(&circles[c.B].py, (corr_y * B.inv_mass));
	}
}

__global__ void solveCollisions_Kernel(Circle * circles, int n, 
	Collision * colls, unsigned int * n_cols, int iterations, float gravity, float dt) 
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < (*n_cols)){
		Collision c = colls[idx];
		Circle A = circles[c.A];
		Circle B = circles[c.B];
		float Avx = A.vx, Avy = A.vy, AvA = A.angularVelocity;
		float Bvx = B.vx, Bvy = B.vy, BvA = B.angularVelocity;

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
	Collision * colls, unsigned int * n_cols) 
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int bloc = blockDim.x * blockIdx.x;

	__shared__ Circle sh_Circles[BLOCK];
	//Circle ri;
	Collision c;
	float ripx, ripy, riinv_mass, riradius;
	float rjpx, rjpy, rjinv_mass, rjradius;
	if(i < n){
		//ri = circles[i];
		ripx = circles[i].px;
		ripy = circles[i].py;
		riinv_mass = circles[i].inv_mass;
		riradius = circles[i].radius;
	}

	for (unsigned int j = bloc; j < n; j += BLOCK){
		if(j + tid < n)
			sh_Circles[tid] = circles[j + tid];

		//__syncthreads();

		for(int k = 0; k < BLOCK && j + k < n; ++k){
			if(j + k <= i)
				continue;
			//Circle rj = sh_Circles[k];
			rjinv_mass = sh_Circles[k].inv_mass;
			rjpx = sh_Circles[k].px;
			rjpy = sh_Circles[k].py;
			rjradius = sh_Circles[k].radius;

			if(riinv_mass == 0.0f && rjinv_mass == 0.0f)
				continue;
			
			c.A = i;
			c.B = j + k;
			c.normal_x = rjpx - ripx;
			c.normal_y = rjpy - ripy;
			//c.normal_x = __fsub_rd(rjpx, ripx);
			//c.normal_y = __fsub_rd(rjpy, ripy);
			float suma_radius = riradius + rjradius;
			//float suma_radius = __fadd_rd(riradius, rjradius);

			float squared_dist = c.normal_x * c.normal_x + c.normal_y * c.normal_y;

			//float suma_radius = ri.radius + rj.radius;
			if(squared_dist > suma_radius * suma_radius)
				continue; //Not contact

			float dist = sqrtf(squared_dist);
			float inv_dist = __frcp_rd(dist);
			
			if(dist < EPS) {
				c.penetration = riradius;
				c.normal_x = 1.0f;
				c.normal_y = 0.0f;
				c.contact_x = ripx;
				c.contact_y = ripy;
			}
			else{
				c.penetration = suma_radius - dist;
				//c.penetration = __fsub_rd(suma_radius, dist);
				c.normal_x *= inv_dist;
				c.normal_y *= inv_dist;
				//c.normal_x = __fmul_rd(c.normal_x, inv_dist);
				//c.normal_y = __fmul_rd(c.normal_y, inv_dist);
				c.contact_x = c.normal_x * riradius + ripx;
				c.contact_y = c.normal_y * riradius + ripy;
				//c.contact_x = __fmaf_rd(c.normal_x, riradius, ripx);
				//c.contact_y = __fmaf_rd(c.normal_y, riradius, ripy);
			}
			//int idx = *n_cols;
			//(*n_cols)++;

			int idx = atomicInc(n_cols, 1e9); //Faster than atomicAdd(n_cols, 1)??
			colls[idx] = c;
		}

		//__syncthreads();
	}
}

__global__ void integrateVelocities_Kernel(Circle * circles, int n, float gravity, float dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < n){
		//Circle c = circles[i]; its faster with more global reads??
		if(circles[i].inv_mass > 0.0f){
			circles[i].px += circles[i].vx * dt;
			circles[i].py += circles[i].vy * dt;
			circles[i].vy += gravity * (dt / 2.0f);
		}
		//circles[i] = c;
	}
}

__global__ void integrateForces_Kernel(Circle * circles, int n, float gravity, float dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < n){
		//Circle c = circles[i];
		if(circles[i].inv_mass > 0.0f)
			circles[i].vy += gravity * (dt / 2.0f);
		//circles[i] = c;
	}
}

__global__ void initialize_Kernel() {
	printf("GPU initialized\n");
}

void GPU::positionalCorrection_GPU(){
	dim3 dimGrid(ceil((float)this->n_cols / BLOCK));
	dim3 dimBlock(BLOCK);

	positionalCorrection_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), 
		colls_GPU, this->n_cols_GPU);
	cudaDeviceSynchronize();

	//cudaFree(colls_GPU);
	//cudaFree(n_cols_GPU);
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
	
	cudaMemset(this->n_cols_GPU, 0, sizeof(unsigned int));
	
	if(this->lro->size() >= this->MAX_cols_GPU){
		cudaFree(colls_GPU);
		this->MAX_cols_GPU = this->lro->size() * 2;
		cudaMalloc((void **) &colls_GPU, sizeof(Collision) * this->MAX_cols_GPU * N_COL);
	}
	//cudaMalloc((void **) &colls_GPU, sizeof(Collision) * this->lro->size() * N_COL);

	dim3 dimGrid(ceil((float)this->lro->size() / BLOCK));
	dim3 dimBlock(BLOCK);
	calculateContacs_Kernel<<<dimGrid,dimBlock>>>(circles_GPU, this->lro->size(), 
		colls_GPU, this->n_cols_GPU);
	cudaDeviceSynchronize();

	cudaMemcpy(&this->n_cols, this->n_cols_GPU, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	contacts.resize(n_cols);
	cudaMemcpy(&(contacts[0]), &colls_GPU[0], 
			sizeof(Collision) * this->n_cols, cudaMemcpyDeviceToHost);
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
	//cudaFree(0);
	initialize_Kernel<<<1,1>>>();
	
	//Create in GPU to save cudaMalloc's
	cudaMalloc((void **) &this->n_cols_GPU, sizeof(int));
	//hello<<<1,1>>>();
	//cudaDeviceSynchronize();

	int device;
	cudaGetDevice(&device);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	printf("Cuda context initialized in device: \"%s\"\n", props.name);
	cudaDeviceSynchronize();
	return;
}

void GPU::copy_HostToDevice(){
	if(this->lro->size() >= this->MAX_GPU_obj){
		cudaFree(circles_GPU);
		this->MAX_GPU_obj = 2 * this->lro->size();
		cudaMalloc((void **) &circles_GPU, sizeof(Circle) * this->MAX_GPU_obj);
	}
	cudaMemcpyAsync(&circles_GPU[0], &(this->lro->vro[0]), 
			sizeof(Circle) * this->lro->size(), cudaMemcpyHostToDevice);
	this->N_GPU_obj = this->lro->size();
}

void GPU::copy_DeviceToHost(){
	cudaMemcpyAsync(&(this->lro->vro[0]), &circles_GPU[0], 
			sizeof(Circle) * this->lro->size(), cudaMemcpyDeviceToHost);
}

int GPU::circlesInGPU(){
	return this->N_GPU_obj;
}

GPU::GPU(){
	this->N_GPU_obj = -1;
	this->MAX_GPU_obj = 0;
	this->MAX_cols_GPU = 0;
}

GPU::GPU(ListCircles * list){
	this->lro = list;
	this->N_GPU_obj = 0;
	this->MAX_GPU_obj = 0;
	this->MAX_cols_GPU = 0;
}

GPU::~GPU(){
	if(colls_GPU != 0){
		cudaFree(colls_GPU);
		cudaFree(n_cols_GPU);
		cudaFree(circles_GPU);
	}
	//printf("%d\n", n_cols_GPU);
	//printf("%d\n", circles_GPU);
	//if(n_cols_GPU != 0)	
	//	cudaFree(n_cols_GPU);
	//if(circles_GPU != 0)
	//	cudaFree(circles_GPU);

}