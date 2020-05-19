#include "main.h"
#include "list_circles.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

class Collision {
	public:
		CUDA_HOSTDEV Collision(){;}
		int A, B;
		float penetration, contact_x, contact_y;
		float normal_x, normal_y;
};

class GPU{
	private:
		int N_GPU_obj, MAX_GPU_obj;
		int MAX_cols_GPU;
		ListCircles * lro;
		Circle * circles_GPU;
		Collision * colls_GPU;
	public:
		int n_cols, *n_cols_GPU;
		GPU(ListCircles * list);
		GPU();
		~GPU();
		void initializeContext();
		void copy_HostToDevice();
		void copy_DeviceToHost();
		void integrateForces_GPU();
		void integrateVelocities_GPU();
		void calculateContact_GPU(vector<Collision> &contacts);
		void solveCollisions_GPU(vector<Collision> &contacts);
		void positionalCorrection_GPU();
};
