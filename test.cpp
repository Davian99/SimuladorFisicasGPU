#include <GL/glut.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <unistd.h>

#define PI 3.121592f
#define HEIGHT 800
#define WIDTH 800

using namespace std;

vector<pair<float, float>> circles;

void keyboard(unsigned char c, int x, int y) {
    if (c == 27) {
        exit(0);
    }
}

void to_coord(int x, int y, float &fx, float &fy){
	fx = ((x * 2.0f) / WIDTH) - 1;
	fy = -1.0f * (((y * 2.0f) / HEIGHT) - 1);
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
    	if (state == GLUT_DOWN){
    		float fx, fy;
    		to_coord(x, y, fx, fy);
    		circles.push_back({fx, fy});
    		printf("M = (%f, %f)\n", fx, fy);
    		glutPostRedisplay();
    	}
        //exit(0);
    }

    if ((button == 3) || (button == 4)) {
        if (state == GLUT_UP) 
       		return; // Disregard redundant GLUT_UP events
       	printf("Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);
   	}
}

float to_percent(float pixels){
	return pixels / WIDTH;
}

void DrawCircle(float cx, float cy, float r, int num_segments) {
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++)   {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 
        float x = r * cos(theta);//calculate the x component 
        float y = r * sin(theta);//calculate the y component 
        glVertex2f(x + cx, y + cy);//output vertex 
    }
    glEnd();
}
 
void render(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for(int i = 0; i < circles.size(); ++i)
    	DrawCircle(circles[i].first, circles[i].second, to_percent(10), 360);
	//cout << circles.size() << endl;

    DrawCircle(0, 0, 1, 300);

    glutSwapBuffers();
}

int main(int argc, char** argv) {

	circles.clear();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Simple GLUT Application");

    glutDisplayFunc(render);       
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);

    glutMainLoop();
}