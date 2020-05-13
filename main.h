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
#define NUM_CIR_SEG 360
#define HEIGHT 800 //Height of the window
#define WIDTH 800 //Width of the window

using namespace std;

const float fps = 60.0f;
const float dt = 1000.0f / fps;

extern float cos_table[NUM_CIR_SEG];
extern float sin_table[NUM_CIR_SEG];

void renderString(int x, int y, string s);
void coordTo(int x, int y, float &fx, float &fy);
float pixelsTo(float px);
void glVertexC(float x, float y);
float degToRad(float deg);