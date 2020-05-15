#pragma once

#include "main.h"
#include "list_render_object.h"
#include "physics.h"

class Scene {

	private:
		int mx, my;
		Physics phy;

	public:
		int frame_count, n_collisions;
		ListRenderObject lro;
		bool activated_physics, stepByStep, static_circles = false;
		
		Scene();
		void render();
		void upddateMousePos(int x, int y);
		void addCircle(int x, int y, int r, bool _static);
		void renderDefaultText();
		void reset();
};