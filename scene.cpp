#include "scene.h"

Scene::Scene(){
	;
}

void Scene::render(){
	this->lro.renderAll();
}

void Scene::upddate_mouse_pos(int x, int y){
	this->mx = x;
	this->my = y;
}

void Scene::addCircle(int x, int y, int r){
	this->lro.addCircle(x, y, r);
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

float pixelsTo(float px){
	return px / ((WIDTH + HEIGHT) / 2);
}

void coordTo(int x, int y, float &fx, float &fy){
	fx = ((x * 2.0f) / WIDTH) - 1;
	fy = -1.0f * (((y * 2.0f) / HEIGHT) - 1);
}

void renderString(int x, int y, string s){
	glColor3f(0.8f, 0.8f, 0.8f);
	float fx, fy;
	coordTo(x, y, fx, fy);
	glRasterPos2f(fx, fy);
	unsigned int l = s.size();
	for(unsigned int i = 0; i < l; ++i)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18 , s[i]);
}