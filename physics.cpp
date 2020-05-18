#include "main.h"
#include "physics.h"
#include "scene.h"

void integrateVelocitiesObject(Circle * ro);
void integrateForcesObject(Circle * ro);

Physics::Physics(ListCircles * lro){
	this->lro = lro;
	this->gpu = GPU(lro);
	this->gpu.initializeContext();
}

void Physics::step(){
	if(true){
		this->gpu.copy_HostToDevice();
		this->gpu.calculateContact_GPU(this->contacs);
		//printf("contacs.size()%d\n", contacs.size());
		//this->contacs.clear();
		//this->calculateContacs();
		//this->gpu.copy_DeviceToHost();

		//this->gpu.copy_HostToDevice();
		this->gpu.integrateForces_GPU();
		this->gpu.copy_DeviceToHost();

		//this->gpu.solveCollisions_GPU(this->contacs);
		this->solveCollisions();

		this->gpu.copy_HostToDevice();
		this->gpu.integrateVelocities_GPU();
		this->gpu.copy_DeviceToHost();

		this->positionalCorrection();
		this->n_collisions = contacs.size();
		this->contacs.clear();
	}
	else{
		this->calculateContacs();
		this->integrateForces();
		this->solveCollisions();
		this->integrateVelocities();
		this->positionalCorrection();
		this->contacs.clear();
	}	
}

bool calculateContacPoints(Circle * ri, Circle * rj, Collision &c){
	c.normal_x = rj->px - ri->px;
	c.normal_y = rj->py - ri->py;
	float dist = hypot(c.normal_x, c.normal_y);
	float suma_radius = ri->radius + rj->radius;
	if(dist >= suma_radius)
		return false; //Not contact
	
	if(dist <= EPS) {
		c.penetration = ri->radius;
		c.normal_x = 1.0f;
		c.normal_y = 0.0f;
		c.contact_x = ri->px;
		c.contact_y = ri->py;
	}
	else{
		c.penetration = suma_radius - dist;
		c.normal_x /= dist;
		c.normal_y /= dist;
		c.contact_x = c.normal_x * ri->radius + ri->px;
		c.contact_y = c.normal_y * ri->radius + ri->py;
	}
	return true;
}

void Physics::calculateContacs(){
	for (unsigned int i = 0; i < this->lro->size(); ++i){
		Circle * ri = &(this->lro->vro[i]);
		for (unsigned int j = i + 1; j < this->lro->size(); ++j){
			Circle * rj = &(this->lro->vro[j]);
			if(ri->inv_mass == 0.0f && rj->inv_mass == 0.0f)
				continue;
			Collision c;
			c.A = i;
			c.B = j;
			if(calculateContacPoints(ri, rj, c))
				this->contacs.push_back(c);
		}
	}
	this->n_collisions = this->contacs.size();
}

void integrateForcesObject(Circle * ro){
	if(ro->inv_mass > 0.0f){
		ro->vy += gravity * (dt / 2.0f);
	}
}

void Physics::integrateForces(){
	for (unsigned int i = 0; i < this->lro->size(); ++i){
		integrateForcesObject(&(this->lro->vro[i]));
	}
}

void Physics::calculateImpulse(Collision &c){
	Circle * A = &(this->lro->vro[c.A]);
	Circle * B = &(this->lro->vro[c.B]);
	if(A->mass + B->mass < EPS) {
    	A->vx = 0.0f;
    	A->vy = 0.0f;
    	B->vx = 0.0f;
    	B->vy = 0.0f;
    	return;
  	}

  	float rax = c.contact_x - A->px;
  	float ray = c.contact_y - A->py;
  	float rbx = c.contact_x - B->px;
  	float rby = c.contact_y - B->py;

  	float rvx = B->vx - (B->angularVelocity * rby) - A->vx + (A->angularVelocity * ray); //ERROR POSIBLE
  	float rvy = B->vy + (B->angularVelocity * rbx) - A->vy - (A->angularVelocity * rax);

  	float contact_vel = rvx * c.normal_x + rvy * c.normal_y;

    if(contact_vel > 0.0f)
      return;

  	float raCrossN = (rax * c.normal_y) - (ray * c.normal_x);
  	float rbCrossN = (rbx * c.normal_y) - (rby * c.normal_x);

  	float invMassSum = A->inv_mass + B->inv_mass + raCrossN*raCrossN * A->inv_inertia + rbCrossN*rbCrossN * B->inv_inertia; 
  	
  	float e = 0.2f;
  	if((rvx * rvx + rvy * rvy) < ((dt * gravity * dt * gravity) + EPS))
  		e = 0.0f;

  	float j = -(1.0f + e) * contact_vel;
  	j /= invMassSum;

  	float impulse_x = c.normal_x * j;
  	float impulse_y = c.normal_y * j;

  	A->applyImpulse(-impulse_x, -impulse_y, rax, ray);
  	B->applyImpulse(impulse_x, impulse_y, rbx, rby);
  	return;
}

void Physics::solveCollisions(){
	for (int i = 0; i < iterations; ++i){
		for (auto c : this->contacs){
			this->calculateImpulse(c);
		}
	}
}

void integrateVelocitiesObject(Circle * ro){
	if(ro->inv_mass > 0.0f){
		ro->px += ro->vx * dt;
		ro->py += ro->vy * dt;
		integrateForcesObject(ro);
	}
}

void Physics::integrateVelocities(){
	for (unsigned int i = 0; i < this->lro->size(); ++i){
		integrateVelocitiesObject(&(this->lro->vro[i]));
	}
}

void Physics::contactCorrection(Collision &c){
	Circle * A = &(this->lro->vro[c.A]);
	Circle * B = &(this->lro->vro[c.B]);
	float k_slop = 0.05f; // Penetration allowance
	float percent = 0.4f; // Penetration percentage to correct
	float corr_x = (max(c.penetration - k_slop, 0.0f) / (A->inv_mass + B->inv_mass)) * c.normal_x * percent;
	float corr_y = (max(c.penetration - k_slop, 0.0f) / (A->inv_mass + B->inv_mass)) * c.normal_y * percent;

	A->px -= corr_x * A->inv_mass;
	A->py -= corr_y * A->inv_mass;
	B->px += corr_x * B->inv_mass;
	B->py += corr_y * B->inv_mass;
}

void Physics::positionalCorrection(){
	for (auto c : this->contacs){
		this->contactCorrection(c);
	}
}
