#pragma once

#include "main.h"
#include "render_object.h"

class Scene {

	private:
		int mx, my;
		void print_text();
		Circle cir;

	public:
		Scene();
		void render();
		void print_x();
		void upddate_mouse_pos(int x, int y);
};