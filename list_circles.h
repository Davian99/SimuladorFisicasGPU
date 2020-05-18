#pragma once

#include "circle.h"

class ListCircles {

	private:
		
	public:
		vector<Circle> vro;
		
		ListCircles();
		void addCircle(int x, int y, int r, bool _static);
		void renderAll();
		int size();
		void clear();
};
