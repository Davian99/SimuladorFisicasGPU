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

void Circle::render(){
	float px, py, rf = pixels_to(this->r);
	coord_to(this->cx, this->cy, px, py);

    glBegin(GL_LINE_LOOP);
   	glColor3ub(255, 255, 255);

    for (int ii = 0; ii < 360; ii++)   {
        float theta = 2.0f * 3.1415926f * float(ii) / float(360);//get the current angle 
        float x = rf * cos(theta);//calculate the x component 
        float y = rf * sin(theta);//calculate the y component 
        glVertex2f(x + px, y + py);//output vertex 
    }
    glEnd();
    glColor3ub(255, 255, 255);
}