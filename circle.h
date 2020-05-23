#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

class Circle {
	private:

	public:
		float px, py, radius;
		float vx, vy;
		float angularVelocity;
		float inv_mass;
		float inv_inertia;

		CUDA_HOSTDEV Circle(float x, float y, float r, bool _static);
		CUDA_HOSTDEV Circle(){};
		CUDA_HOSTDEV ~Circle(){};
		void render();
		void setStatic();
		void applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy);		
};
