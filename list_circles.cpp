#include "main.h"
#include "list_circles.h"

ListCircles::ListCircles(){
	this->n_non_static = 0;
}

void ListCircles::addCircle(int x, int y, int r, bool _static){
	if(!_static)
		this->n_non_static++;
	this->vro.push_back(Circle(x, y, r, _static));
}

void ListCircles::renderAll(){
	for (auto p : this->vro)
    	p.render();
}

void ListCircles::removeNonVisible(){
	int idx = 0;
	this->n_non_static = 0;
	for (auto p : this->vro){
		if(p.inv_mass == 0.0f){
			this->vro[idx] = p;
    		idx++;
    		continue;
		}
		//printf("p.py %f\n", p.py);
    	if(p.py < (HEIGHT + p.radius + 100.0f)){
    		this->n_non_static++;
    		this->vro[idx] = p;
    		idx++;
    	}
	}
	this->vro.resize(idx);
}

int ListCircles::size(){
	return this->vro.size();
}

int ListCircles::m_size(){
	return this->n_non_static;
}

void ListCircles::clear(){
	this->n_non_static = 0;
	this->vro.clear();
}