#pragma once

#include "circle.h"
#include "gpu_routines.h"

class ListCircles;

class Physics {
	private:
		ListCircles * lro;
		vector<Collision> contacs;
		GPU gpu;
	public:
		unsigned int n_collisions;

		Physics(ListCircles * lro);
		Physics(){
			this->n_collisions = 0;
		};
		void step();
		void calculateContacs();
		void integrateForces();
		void solveCollisions();
		void integrateVelocities();
		void positionalCorrection();
		void calculateImpulse(Collision &c);
		void contactCorrection(Collision &c);
};

