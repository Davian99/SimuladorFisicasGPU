#pragma once

#include "main.h"
#include "list_circles.h"
#include "physics.h"

using namespace std::chrono;

class Scene {

	private:
		int mx, my;
		unsigned int max_collisions;
		steady_clock::time_point begin;
		Physics phy;

	public:
		unsigned int frame_count, n_collisions;
		int bench_frames;
		ListCircles lro;
		float frames_per_second;
		bool activated_physics, stepByStep, static_circles = false;
		bool benchmarking, render_extra;
		
		Scene();
		void render();
		void upddateMousePos(int x, int y);
		void addCircle(int x, int y, int r, bool _static);
		void renderDefaultText();
		void reset();
		void addWalls();
		void normalDistribution();
		void benchmark(int sep);
		void elapsedTime();
		void no_ogl();
};