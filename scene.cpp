#include "scene.h"

Scene::Scene(int argc, char** argv){
	init_OGL(argc, argv);



}

void Scene::init_OGL(int argc, char** argv){
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
	return;
}

void render(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glutSwapBuffers();
}