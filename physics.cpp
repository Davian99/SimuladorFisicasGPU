#include "main.h"
#include "physics.h"

void integrateVelocitiesObject(Circle * ro);
void integrateForcesObject(Circle * ro);

Physics::Physics(ListCircles * lro){
	this->lro = lro;
	this->gpu = GPU(lro);
	this->gpu.initializeContext();
	this->n_collisions = 0;
	//this->total_collisions = 0;
	this->solve_time = 0;
}

void Physics::step(){
	//Test which is faster
	//if(random_solve_cols)
	//	random_shuffle(this->lro->vro.begin(), this->lro->vro.end()); //More natural simulation
	if(use_gpu){
		//if(this->lro->size() > gpu.circlesInGPU())
		this->gpu.copy_HostToDevice();
		this->gpu.calculateContact_GPU(this->contacs);

		this->gpu.integrateForces_GPU();
		this->gpu.copy_DeviceToHost();

		//this->gpu.solveCollisions_GPU(this->contacs);
		this->solveCollisions();

		this->gpu.copy_HostToDevice();
		this->gpu.integrateVelocities_GPU();
		
		this->gpu.positionalCorrection_GPU();
		this->gpu.copy_DeviceToHost();

		this->n_collisions = this->gpu.n_cols;
		//printf("%d\n", this->n_collisions);
		//this->contacs.clear();
	}
	else{
		this->calculateContacs();
		this->integrateForces();
		this->solveCollisions();
		this->integrateVelocities();
		this->positionalCorrection();
		this->contacs.clear();
	}
	//this->total_collisions += this->n_collisions;
}

void Physics::renderCollisions(){
	glPointSize(4.0f);

	glBegin(GL_POINTS);
    glColor3ub(255, 0, 0);
    
	for (auto c : this->contacs){
		glVertexC(c.contact_x, c.contact_y);
	}
	glEnd();
}

bool calculateContacPoints(Circle * ri, Circle * rj, Collision &c){
	c.normal_x = rj->px - ri->px;
	c.normal_y = rj->py - ri->py;

	float suma_radius = ri->radius + rj->radius;
	float squared_dist = c.normal_x * c.normal_x + c.normal_y * c.normal_y;
	
	if(squared_dist >= suma_radius * suma_radius)
		return false; //Not contact

	float dist = sqrtf(squared_dist);
	float inv_dist = 1.0f / dist;
	
	if(dist < EPS) {
		c.penetration = ri->radius;
		c.normal_x = 1.0f;
		c.normal_y = 0.0f;
		c.contact_x = ri->px;
		c.contact_y = ri->py;
	}
	else{
		c.penetration = suma_radius - dist;
		c.normal_x *= inv_dist;
		c.normal_y *= inv_dist;
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
  	if((rvx * rvx + rvy * rvy) < calcule_e)
  		e = 0.0f;

  	float j = (-(1.0f + e) * contact_vel) / invMassSum;
  	//j /= invMassSum;

  	float impulse_x = c.normal_x * j;
  	float impulse_y = c.normal_y * j;

  	A->vx += A->inv_mass * (-impulse_x);
  	A->vy += A->inv_mass * (-impulse_y);
  	A->angularVelocity += A->inv_inertia * ((rax * (-impulse_y)) - (ray * (-impulse_x)));

	B->vx += B->inv_mass * (impulse_x);
  	B->vy += B->inv_mass * (impulse_y);
  	B->angularVelocity += B->inv_inertia * ((rbx * (impulse_y)) - (rby * (impulse_x)));

  	return;
}

void Physics::solveCollisions(){
	if(random_solve_cols)
		random_shuffle(this->contacs.begin(), this->contacs.end()); //More natural simulation
	//Timing of the crucial function
	auto start = high_resolution_clock::now();
	for (int i = 0; i < iterations; ++i){
		for (int j = 0; j < this->contacs.size(); ++j){
			this->calculateImpulse(this->contacs[j]);
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	this->solve_time += duration.count();
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
