#pragma once

#include "circle.h"
#include "gpu_routines.h"

class ListCircles;

class Collision {
	public:
		Collision(){;}
		Circle *A, *B;
		float penetration, contact_x, contact_y;
		float normal_x, normal_y;
};

class Physics {
	private:
		ListCircles * lro;
		vector<Collision> contacs;
		GPU gpu;
	public:
		int n_collisions;

		Physics(ListCircles * lro);
		Physics(){;};
		void step();
		void calculateContacs();
		void integrateForces();
		void solveCollisions();
		void integrateVelocities();
		void positionalCorrection();
};

