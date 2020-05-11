#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include <GL/glut.h> //For OpenGL

#define PI 3.121592f
#define HEIGHT 800 //Height of the window
#define WIDTH 800 //Width of the window

using namespace std;

int grav = 100, change = 2; //Grav es el diametro!!
const float fps = 60;

vector<pair<int, int>> circles;
pair<int, int> mouse_location = {0, 0};

void timer(int) {
    glutPostRedisplay();
    glutTimerFunc(1000/fps, timer, 0);
}

void keyboard(unsigned char c, int x, int y) {
    switch(c){
    	case 27:
    		exit(0);
  		case 'c':
  			circles.clear();
  			break;
    }
}

void to_coord(int x, int y, float &fx, float &fy){
	fx = ((x * 2.0f) / WIDTH) - 1;
	fy = -1.0f * (((y * 2.0f) / HEIGHT) - 1);
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
    	if (state == GLUT_DOWN){
    		//float fx, fy;
    		//to_coord(x, y, fx, fy);
    		circles.push_back({x, y});
    		printf("M = (%d, %d)\n", x, y);
    		//glutPostRedisplay();
    	}
    }

    if ((button == 3) || (button == 4)) {
        if (state == GLUT_UP) 
       		return; // Disregard redundant GLUT_UP events
        if(button == 3){
        	grav += change;
        	printf("Gravedad aumentada a %d\n", grav);
        }
        else{
        	grav = max(0, grav-change);
        	printf("Gravedad disminuida a %d\n", grav);
        }
   	}
}

void myMouseMove( int x, int y) {
	//printf("Mouse at (%d, %d)\n", x, y);
	mouse_location = {x, y};
}

float to_percent(float pixels){
	return pixels / WIDTH;
}

bool collide(float cx, float cy){
	float mx = mouse_location.first;
	float my = mouse_location.second;
	float dist = sqrtf((cx - mx)*(cx - mx) + (cy - my)*(cy - my));
	bool col = dist <= (grav);
	//if(dist == 0.0f)
	//	col = false;

	return col;
}

void DrawCircle(int cxp, int cyp, float r, int num_segments) {
	float cx, cy;
	to_coord(cxp, cyp, cx, cy);

    glBegin(GL_LINE_LOOP);
    if(collide(cxp, cyp)){
    	glColor3ub(255, 0, 0);
    }
   	else
   		glColor3ub(255, 255, 255);

    for (int ii = 0; ii < num_segments; ii++)   {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 
        float x = r * cos(theta);//calculate the x component 
        float y = r * sin(theta);//calculate the y component 
        glVertex2f(x + cx, y + cy);//output vertex 
    }
    glEnd();
    glColor3ub(255, 255, 255);
}

void render(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int x = rand() % WIDTH;
    int y = rand() % HEIGHT;
    circles.push_back({x, y});

    for(int i = 0; i < circles.size(); ++i){
    	DrawCircle(circles[i].first, circles[i].second, to_percent(grav), 360);
    	circles[i].first += 0;
    	circles[i].second += 0;
    }
	//cout << circles.size() << endl;

	
	DrawCircle(mouse_location.first, mouse_location.second, to_percent(grav), 360);


    glutSwapBuffers();
}

int main(int argc, char** argv) {
	srand(time(NULL));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Test Program");

    glutDisplayFunc(render);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutPassiveMotionFunc( myMouseMove);

    glutTimerFunc(1000/60, timer, 0);

    glutMainLoop();
}