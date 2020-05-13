#pragma once

class Scene;

class Physics {
	private:
		Scene * scene;
	public:
		Physics(Scene * scene);
		Physics(){;};
		void step();
};