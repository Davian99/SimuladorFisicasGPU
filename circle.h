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
		float force_x, force_y;
		float angularVelocity;
		float mass, inv_mass;
		float inertia, inv_inertia;

		Circle(float x, float y, float r, bool _static);
		Circle();
		CUDA_HOSTDEV ~Circle(){};
		void render();
		void setStatic();
		void applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy);		
};
