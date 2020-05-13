#include "main.h"
#include "render_object.h"

Circle::Circle(float x, float y, float r){
	this->cx = x;
	this->cy = y;
	this->radius = r;
    this->orient = rand() % 360;
    cout << "orient = " << this->orient << endl;
}

Circle::Circle(){
    this->cx = 0.0f;
    this->cy = 0.0f;
    this->radius = 0.0f;
}

void Circle::modifyPos(float x, float y){
    this->cx += x;
    this->cy += y;
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
	float px, py, rf = pixelsTo(2.0f * this->radius);
	coordTo(this->cx, this->cy, px, py);

    glBegin(GL_LINE_LOOP);
   	glColor3ub(255, 255, 255);

    for (int ii = 0; ii < NUM_CIR_SEG; ii++)   {
        float theta = 2.0f * PI * float(ii) / float(NUM_CIR_SEG);//get the current angle 
        float x = this->radius * cos_table[ii];
        float y = this->radius * sin_table[ii]; 
        glVertexC(x + this->cx, y + this->cy);
    }
    glEnd();

    glBegin(GL_LINE_LOOP);

    glBegin(GL_LINES);
    glVertexC(this->cx, this->cy);
    float x = this->radius * cos_table[(int)this->orient];
    float y = this->radius * sin_table[(int)this->orient];
    glVertexC(this->cx + x, this->cy + y);
    glEnd();


    glColor3ub(255, 255, 255);
}