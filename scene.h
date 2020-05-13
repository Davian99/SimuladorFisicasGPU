#pragma once

#include "main.h"
#include "list_render_object.h"
#include "physics.h"

class Scene {

	private:
		int mx, my;
		int frame_count;
		Physics phy;

	public:
		ListRenderObject lro;
		bool activated_physics;
		
		Scene();
		void render();
		void upddateMousePos(int x, int y);
		void addCircle(int x, int y, int r);
		void renderDefaultText();
		void reset();
};