#include "main.h"
#include "physics.h"
#include "scene.h"

Physics::Physics(Scene * scene){
	this->scene = scene;
}

void Physics::step(){
	for (auto p : this->scene->lro.vro){
		p->modifyPos(0.0f, 1.0f);
	}
}