#include "main.h"
#include "physics.h"
#include "scene.h"

Physics::Physics(Scene * scene){
	this->scene = scene;
}

void Physics::step(){
	if(this->scene->lro.vro.size() == 0)
		return;
	this->integrateForces();
	this->integrateVelocities();
	this->clearForces();
}

void integrateForcesObject(RenderObject * ro){
	if(ro->inv_mass > 0.0f){
		//ro->modifyPos(0.0f, 1.0f);
		ro->vx += (ro->force_x * ro->inv_mass) * (dt / 2.0f);
		ro->vy += (ro->force_y * ro->inv_mass + gravity) * (dt / 2.0f);
		ro->angularVelocity = ro->torque * ro->inertia * (dt / 2.0f);
	}
}

void integrateVelocitiesObject(RenderObject * ro){
	if(ro->inv_mass > 0.0f){
		//b->position += b->velocity * dt;
		ro->px += ro->vx * dt;
		ro->py += ro->vy * dt;
		ro->orient += ro->angularVelocity * dt;
		integrateForcesObject(ro);
	}
}

void Physics::integrateVelocities(){
	for (auto p : this->scene->lro.vro){
		integrateVelocitiesObject(p);
	}
}

void Physics::integrateForces(){
	for (auto p : this->scene->lro.vro){
		integrateForcesObject(p);
	}
}

void Physics::clearForces(){
	for (auto p : this->scene->lro.vro){
		p->force_x = 0.0f;
		p->force_y = 0.0f;
    	p->torque = 0.0f;
	}
}
