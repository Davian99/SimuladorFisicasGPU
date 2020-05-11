#pragma once

class RenderObject {
	public:
		void draw();
};

class Circle: public RenderObject {
	private:
		int cx, cy, r;

	public:
		Circle(int x, int y, int r);
		Circle();
		void render();
};