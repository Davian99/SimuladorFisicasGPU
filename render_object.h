#pragma once

class RenderObject {
	public:
		virtual void render() {;}
};

class Circle: public RenderObject {
	private:
		int cx, cy, r;

	public:
		Circle(int x, int y, int r);
		Circle();
		~Circle();
		void render() override;
};