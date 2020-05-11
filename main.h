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
#define HEIGHT 800 //Height of the window
#define WIDTH 800 //Width of the window

using namespace std;

const float fps = 60;

void coord_to(int x, int y, float &fx, float &fy);
float pixels_to(float px);