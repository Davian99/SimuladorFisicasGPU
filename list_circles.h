#pragma once

#include "circle.h"

class ListCircles {

	private:
		
	public:
		int n_non_static;
		vector<Circle> vro;
		
		ListCircles();
		void addCircle(int x, int y, int r, bool _static);
		void renderAll();
		int size();
		int m_size();
		void clear();
		void removeNonVisible();
};
