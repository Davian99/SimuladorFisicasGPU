#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include <GL/glut.h> 

#define PI 3.121592f
#define HEIGHT 800 //Height of the window
#define WIDTH 800 //Width of the window

using namespace std;

const float fps = 60;

void timer(int);
void keyboard(unsigned char c, int x, int y);
void mouse(int button, int state, int x, int y);
void myMouseMove( int x, int y);
void render(void);

class Scene {

	private:
		void init_OGL(int argc, char** argv);
		

	public:
		Scene(int argc, char** argv);


};