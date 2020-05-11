#include "main.h"
#include "scene.h"

void timer(int);
void keyboard(unsigned char c, int x, int y);
void mouse(int button, int state, int x, int y);
void myMouseMove( int x, int y);
void render(void);

Scene scene;

void timer(int) {
    glutPostRedisplay();
    glutTimerFunc(1000/fps, timer, 0);
}

void keyboard(unsigned char c, int x, int y) {
    return;
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
    	if (state == GLUT_DOWN){
    		;
    	}
    }

    if ((button == 3) || (button == 4)) {
        if (state == GLUT_UP) 
       		return; // Disregard redundant GLUT_UP events
        if(button == 3){
        	;//UP
        }
        else{
        	;//DOWN
        }
   	}
}

void myMouseMove(int x, int y) {
	scene.upddate_mouse_pos(x, y);
}

void render(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    scene.render();

    glutSwapBuffers();
}

void init_OCL(int argc, char** argv){
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

int main(int argc, char** argv){
	init_OCL(argc, argv);

    
}