#pragma once

#include "render_object.h"

class ListRenderObject {

	private:
		vector<RenderObject *> vro;
	public:
		ListRenderObject();
		~ListRenderObject();
		void addCircle(int x, int y, int r);
		void renderAll();
};
