#include "main.h"
#include "physics.h"
#include "scene.h"

void integrateVelocitiesObject(RenderObject * ro);
void integrateForcesObject(RenderObject * ro);

Physics::Physics(Scene * scene){
	this->scene = scene;
}

void Physics::step(){
	this->calculateContacs();
	this->integrateForces();
	this->solveCollisions();
	this->integrateVelocities();
	this->positionalCorrection();
	this->clearForces();
	this->contacs.clear();
}

bool calculateContacPoints(RenderObject * ri, RenderObject * rj, Collision &c){
	c.A = ri;
	c.B = rj;
	c.normal_x = rj->px - ri->px;
	c.normal_y = rj->py - ri->py;
	float dist = hypot(c.normal_x, c.normal_y);
	float suma_radius = ri->radius + rj->radius;
	if(dist >= suma_radius)
		return false; //Not contact
	
	c.e = min(ri->restitution, rj->restitution);
	c.sf = sqrt(ri->staticFriction * rj->staticFriction);
  	c.df = sqrt(ri->dynamicFriction * rj->dynamicFriction);
	
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
	//printf("Contact: (%f, %f)\n", c.contact_x, c.contact_y);
	return true;
}

void Physics::calculateContacs(){
	for (unsigned int i = 0; i < this->scene->lro.vro.size(); ++i){
		RenderObject * ri = this->scene->lro.vro[i];
		for (unsigned int j = i + 1; j < this->scene->lro.vro.size(); ++j){
			RenderObject * rj = this->scene->lro.vro[j];
			if(ri->inv_mass == 0.0f && rj->inv_mass == 0.0f)
				continue;
			Collision c;
			if(calculateContacPoints(ri, rj, c))
				this->contacs.push_back(c);
		}
	}
	this->scene->n_collisions = this->contacs.size();
}

void Physics::integrateForces(){
	for (auto p : this->scene->lro.vro){
		integrateForcesObject(p);
	}
}

void calculateImpulse(Collision &c){
	RenderObject * A = c.A;
	RenderObject * B = c.B;
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
  		
  	if((rvx * rvx + rvy * rvy) < ((dt * gravity * dt * gravity) + EPS))
  		c.e = 0.0f;

  	float j = -(1.0f + c.e) * contact_vel;
  	j /= invMassSum;

  	float impulse_x = c.normal_x * j;
  	float impulse_y = c.normal_y * j;

  	A->applyImpulse(-impulse_x, -impulse_y, rax, ray);
  	B->applyImpulse(impulse_x, impulse_y, rbx, rby);

  	float tx = rvx - (c.normal_x * (rvx * c.normal_x + rvy * c.normal_y));
  	float ty = rvy - (c.normal_y * (rvx * c.normal_x + rvy * c.normal_y));

  	float len = sqrt(tx*tx + ty*ty);
  	if(len > EPS){
  		tx /= len;
  		ty /=len;
  	}

  	float jt = -(rvx * tx + rvy * ty);
  	jt /= invMassSum;
  	//printf("jt=%d\n", jt);
  	if(fabs(jt) <= EPS)
  		return;

  	float timp_x = 0.0f, timp_y = 0.0f;
  	if(abs(jt) < j * c.sf){
  		timp_x = tx * jt;
  		timp_y = ty * jt;
  	}
  	else{
  		timp_x = tx * (-jt) * c.df;
  		timp_y = ty * (-jt) * c.df;
  		;
  	}
  	//printf("timp_x=%f, timp_y=%f\n", timp_x, timp_y);
  	//printf("jt=%f\n", jt);
  	//A->applyImpulse(-timp_x, -timp_y, rax, ray);
  	//B->applyImpulse(timp_x, timp_y, rbx, rby);

  	return;
}

void Physics::solveCollisions(){
	for (int i = 0; i < iterations; ++i){
		for (auto c : this->contacs){
			calculateImpulse(c);
		}
	}
}

void Physics::integrateVelocities(){
	for (auto p : this->scene->lro.vro){
		integrateVelocitiesObject(p);
	}
}

void contactCorrection(Collision &c){
	float k_slop = 0.05f; // Penetration allowance
	float percent = 0.4f; // Penetration percentage to correct
	float corr_x = (max(c.penetration - k_slop, 0.0f) / (c.A->inv_mass + c.B->inv_mass)) * c.normal_x * percent;
	float corr_y = (max(c.penetration - k_slop, 0.0f) / (c.A->inv_mass + c.B->inv_mass)) * c.normal_y * percent;

	c.A->px -= corr_x * c.A->inv_mass;
	c.A->py -= corr_y * c.A->inv_mass;
	c.B->px += corr_x * c.B->inv_mass;
	c.B->py += corr_y * c.B->inv_mass;
}

void Physics::positionalCorrection(){
	for (auto c : this->contacs){
		contactCorrection(c);
	}
}

void Physics::clearForces(){
	for (auto p : this->scene->lro.vro){
		p->force_x = 0.0f;
		p->force_y = 0.0f;
    	p->torque = 0.0f;
	}
}

void integrateForcesObject(RenderObject * ro){
	if(ro->inv_mass > 0.0f){
		ro->vx += (ro->force_x * ro->inv_mass) * (dt / 2.0f);
		ro->vy += (ro->force_y * ro->inv_mass + gravity) * (dt / 2.0f);
		ro->angularVelocity = ro->torque * ro->inv_inertia * (dt / 2.0f);
		//printf("AV=%f, torque=%f, inv_inertia=%f, dt=%f\n", ro->angularVelocity, ro->torque, ro->inv_inertia, dt);
	}
}

void integrateVelocitiesObject(RenderObject * ro){
	if(ro->inv_mass > 0.0f){
		ro->px += ro->vx * dt;
		ro->py += ro->vy * dt;
		float orient_change = ro->angularVelocity * dt;
		ro->orient += orient_change;
		ro->setOrient(ro->orient);
		//printf("Orient=%f\n", ro->orient);
		integrateForcesObject(ro);
	}
}
