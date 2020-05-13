#include "main.h"
#include "render_object.h"

Circle::Circle(float x, float y, float r){
	this->px = x;
	this->py = y;
    this->real_r = r;
	this->radius = r;
    this->orient = rand() % 360;
    this->torque = 0.0f;
    this->vx = 0.0f;
    this->vy = 0.0f;
    this->force_x = 0.0f;
    this->force_y = 0.0f;
    this->angularVelocity = 0.0f;

    this->mass = PI * radius * radius * density;
    this->inv_mass = 1.0f / mass;
    this->inertia = mass * radius * radius;
    this->inv_inertia = 1.0f / inertia;
}

Circle::Circle(){
    this->px = 0.0f;
    this->py = 0.0f;
    this->real_r = 0.0f;
}

void Circle::modifyPos(float x, float y){
    this->px += x;
    this->py += y;
}

void Circle::rotate1Deg(){
    this->orient += 1.0f;
    this->orient = (int)this->orient % 360;
}

void Circle::setStatic(){
    this->mass = 0.0f;
    this->inv_mass = 0.0f;
    this->inertia = 0.0f;
    this->inv_inertia = 0.0f;
}

void Circle::ApplyForce(const float fx, const float fy){
    this->force_x += fx;
    this->force_y += fy;
}

void Circle::render() {
    glBegin(GL_LINE_LOOP);
   	glColor3ub(255, 255, 255);
    for (int ii = 0; ii < NUM_CIR_SEG; ii++)   {
        float theta = 2.0f * PI * float(ii) / float(NUM_CIR_SEG);//get the current angle 
        float x = this->real_r * cos_table[ii];
        float y = this->real_r * sin_table[ii]; 
        glVertexC(x + this->px, y + this->py);
    }
    glEnd();

    glBegin(GL_LINE_LOOP);
    
    glBegin(GL_LINES);
    glVertexC(this->px, this->py);
    float x = this->real_r * cos_table[(int)this->orient];
    float y = this->real_r * sin_table[(int)this->orient];
    glVertexC(this->px + x, this->py + y);
    glEnd();

    glColor3ub(255, 255, 255);
}
