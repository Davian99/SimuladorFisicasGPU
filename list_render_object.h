#pragma once

#include "render_object.h"

class ListRenderObject {

	private:
		
	public:
		vector<RenderObject *> vro;
		
		ListRenderObject();
		~ListRenderObject();
		void addCircle(int x, int y, int r, bool _static);
		void renderAll();
		int size();
		void clear();
};
