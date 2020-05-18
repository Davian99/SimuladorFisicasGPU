#include "main.h"
#include "list_circles.h"

ListCircles::ListCircles(){
	;
}

void ListCircles::addCircle(int x, int y, int r, bool _static){
	this->vro.push_back(Circle(x, y, r, _static));
}

void ListCircles::renderAll(){
	for (auto p : this->vro)
    	p.render();
}

int ListCircles::size(){
	return this->vro.size();
}

void ListCircles::clear(){
	this->vro.clear();
}