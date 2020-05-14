#pragma once
//Mirar si hay innecesarios
#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <cstring>

#include <GL/glut.h> 

#define PI 3.141592f
#define EPS 1e-5f
#define NUM_CIR_SEG 30
#define HEIGHT 800 //Height of the window
#define WIDTH 800 //Width of the window

using namespace std;

const float fps = 60.0f;
const float dt = 1.0f / fps;
const float density = 1.0f;

const float gravity = 10.0f * 75.0f;
const unsigned int iterations = 100;

extern float cos_table[NUM_CIR_SEG];
extern float sin_table[NUM_CIR_SEG];

void renderString(int x, int y, string s);
void coordTo(int x, int y, float &fx, float &fy);
float pixelsTo(float px);
void glVertexC(float x, float y);
float degToRad(float deg);
bool save_screenshot(string filename, int w, int h);