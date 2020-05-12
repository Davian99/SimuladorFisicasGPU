#pragma once

#include "main.h"
#include "list_render_object.h"

class Scene {

	private:
		int mx, my;
		ListRenderObject lro;
		void print_text();

	public:
		Scene();
		void render();
		void print_x();
		void upddate_mouse_pos(int x, int y);
		void addCircle(int x, int y, int r);
};