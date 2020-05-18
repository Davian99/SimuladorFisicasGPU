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
		ListCircles * lro;
		Circle * circles_GPU;
		Collision * colls_GPU;
	public:
		GPU(ListCircles * list);
		GPU();
		void initializeContext();
		void update_mem();
		void copy_HostToDevice();
		void copy_DeviceToHost();
		void integrateForces_GPU();
		void integrateVelocities_GPU();
		void calculateContact_GPU(vector<Collision> &contacts);
};
