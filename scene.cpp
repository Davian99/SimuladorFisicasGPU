#include "scene.h"

Scene::Scene(){
	this->frame_count = 0;
	this->phy = Physics(this);
}

void Scene::render(){
	this->phy.step();
	this->frame_count++;
	this->lro.renderAll();
	this->renderDefaultText();
}

void Scene::upddateMousePos(int x, int y){
	this->mx = x;
	this->my = y;
}	

void Scene::addCircle(int x, int y, int r){
	this->lro.addCircle(x, y, r);
}

void Scene::renderDefaultText(){
	string frames = "Frame: " + to_string(this->frame_count);
	renderString(10, 24, frames);
	string object_count = "Objects: " + to_string(this->lro.size());
	renderString(10, 48, object_count);
}

void Scene::reset(){
	this->frame_count = 0;
	this->lro.clear();
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

void glVertexC(float x, float y){
	float fx = ((x * 2.0f) / WIDTH) - 1;
	float fy = -1.0f * (((y * 2.0f) / HEIGHT) - 1);
	glVertex2f(fx, fy);
}

float degToRad(float deg){
	return deg * PI / 180.f;
}