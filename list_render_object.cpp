#include "main.h"
#include "list_render_object.h"

ListRenderObject::ListRenderObject(){
	;
}

ListRenderObject::~ListRenderObject(){
	for (auto p : this->vro)
    	delete p;
}


void ListRenderObject::addCircle(int x, int y, int r, bool _static){
	this->vro.push_back(new Circle(x, y, r, _static));
}

void ListRenderObject::renderAll(){
	for (auto p : this->vro)
    	p->render();
}

int ListRenderObject::size(){
	return this->vro.size();
}

void ListRenderObject::clear(){
	this->vro.clear();
}