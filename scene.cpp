#include "scene.h"

Scene::Scene(){
	cir = Circle(100, 100, 100);
}

float pixels_to(float px){
	return px / ((WIDTH + HEIGHT) / 2);
}

void coord_to(int x, int y, float &fx, float &fy){
	fx = ((x * 2.0f) / WIDTH) - 1;
	fy = -1.0f * (((y * 2.0f) / HEIGHT) - 1);
}

void Scene::render(){
	cir.render();
}

void Scene::upddate_mouse_pos(int x, int y){
	this->mx = x;
	this->my = y;
}

void Scene::print_text(){
	unsigned char ss[] = "The quick god jumps over the lazy brown fox.";
	string sss = "The quick god jumps over the lazy brown fox.";
	int w;
	w = glutBitmapLength(GLUT_BITMAP_8_BY_13, ss);
	glRasterPos2f(0., 0.);
	float x = .5; /* Centre in the middle of the window */
	glRasterPos2f(x - (float) WIDTH / 2, 0.);
	glColor3ub(255, 0, 0);
	for (int i = 0; i < sss.size(); i++) {
    	glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ss[i]);
	}
}