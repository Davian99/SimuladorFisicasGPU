#include "main.h"
#include "circle.h"

Circle::Circle(float x, float y, float r, bool _static){
	this->px = x;
	this->py = y;
	this->radius = r;
    this->vx = 0.0f;
    this->vy = 0.0f;
    this->angularVelocity = 0.0f;
    if(_static){
        this->inv_mass = 0.0f;
        this->inv_inertia = 0.0f;
    }
    else{
        float mass = PI * r * r * density;
        this->inv_mass = 1.0f / mass;
        float inertia = mass * r * r;
        this->inv_inertia = 1.0f / inertia;
    }
    //printf("Circle in (%d,%d), r=%f, m=%f, im=%f, inertia=%f\n", (int)px, (int)py, radius, mass, inv_mass, inertia);
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
}