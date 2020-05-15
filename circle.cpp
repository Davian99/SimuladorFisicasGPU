#include "main.h"
#include "render_object.h"

Circle::Circle(float x, float y, float r, bool _static){
	this->px = x;
	this->py = y;
	this->radius = r;
	this->orient = 0.0F;
    this->torque = 0.0f;
    this->vx = 0.0f;
    this->vy = 0.0f;
    this->force_x = 0.0f;
    this->force_y = 0.0f;
    this->angularVelocity = 0.0f;
    this->staticFriction = 0.5f;
    this->dynamicFriction = 0.3f;
    this->restitution = 0.2f;
    if(_static)
        this->setStatic();
    else{
        this->mass = PI * r * r * density;
        this->inv_mass = 1.0f / mass;
        this->inertia = mass * r * r;
        this->inv_inertia = 1.0f / inertia;
    }
    //printf("R=%f, Mass=%f, Inertia=%f\n", radius, mass, inertia);
}

Circle::Circle(){
    this->px = 0.0f;
    this->py = 0.0f;
    this->radius = 0.0f;
}

void Circle::modifyPos(float x, float y){
    this->px += x;
    this->py += y;
}

void Circle::rotate1Deg(){
    this->orient += 1.0f;
    this->orient = (int)this->orient % 360;
}

void Circle::setOrient(float radians){
	this->orient = radians;
}

void Circle::setStatic(){
    this->mass = 0.0f;
    this->inv_mass = 0.0f;
    this->inertia = 0.0f;
    this->inv_inertia = 0.0f;
}

void Circle::applyForce(const float fx, const float fy){
    this->force_x += fx;
    this->force_y += fy;
}

void Circle::applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy){
    this->vx += this->inv_mass * impulse_x;
    this->vy += this->inv_mass * impulse_y;
    this->angularVelocity += this->inv_inertia * ((cvx * impulse_y) - (cvy * impulse_x));
  //  printf("AV=%f\n", angularVelocity);
}

void Circle::render() {
    glBegin(GL_LINE_LOOP);
   	glColor3ub(255, 255, 255);
    for (int ii = 0; ii < NUM_CIR_SEG; ii++)   {
        //float theta = 2.0f * PI * float(ii) / float(NUM_CIR_SEG);//get the current angle
        float x = this->radius * cos_table[ii];
        float y = this->radius * sin_table[ii];
        glVertexC(x + this->px, y + this->py);
    }
    glEnd();

    glBegin(GL_LINE_LOOP);

    glBegin(GL_LINES);
    glVertexC(this->px, this->py);
    float x = this->radius * cos(this->orient);
    float y = this->radius * sin(this->orient);
    glVertexC(this->px + x, this->py + y);
    glEnd();

    glColor3ub(255, 255, 255);
}
