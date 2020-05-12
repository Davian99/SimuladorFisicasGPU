#include "main.h"
#include "render_object.h"

Circle::Circle(int x, int y, int r){
	this->cx = x;
	this->cy = y;
	this->r = r;
}

Circle::Circle(){
    this->cx = 0;
    this->cy = 0;
    this->r = 0;
}

void Circle::render() {
	float px, py, rf = pixelsTo(this->r);
	coordTo(this->cx, this->cy, px, py);

    glBegin(GL_LINE_LOOP);
   	glColor3ub(255, 255, 255);

    for (int ii = 0; ii < 180; ii++)   {
        float theta = 2.0f * PI * float(ii) / float(180);//get the current angle 
        float x = rf * cos_table[ii];
        float y = rf * sin_table[ii]; 
        glVertex2f(x + px, y + py);
    }
    glEnd();
    glColor3ub(255, 255, 255);
}