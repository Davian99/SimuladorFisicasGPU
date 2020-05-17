#pragma once

#include "render_object.h"
#include "gpu_routines.h"

class Scene;

class Collision {
	public:
		Collision(){;}
		RenderObject *A, *B;
		float penetration, contact_x, contact_y;
		float normal_x, normal_y;
};

class Physics {
	private:
		Scene * scene; // TODO Cambiar por listRenderObject
		vector<Collision> contacs;
		GPU gpu;
	public:
		Physics(Scene * scene);
		Physics(){;};
		void step();
		void calculateContacs();
		void integrateForces();
		void solveCollisions();
		void integrateVelocities();
		void positionalCorrection();
};

