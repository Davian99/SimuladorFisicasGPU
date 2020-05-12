#include "main.h"
#include "scene.h"

#define ESC_KEY 27

void timer(int);
void keyboard(unsigned char c, int x, int y);
void mouse(int button, int state, int x, int y);
void mousePosition( int x, int y);
void render(void);

float cos_table[NUM_CIR_SEG];
float sin_table[NUM_CIR_SEG];

Scene scene;

void timer(int) {
    glutPostRedisplay();
    glutTimerFunc(1000/fps, timer, 0);
}

void keyboard(unsigned char c, int x, int y) {
    switch(c){
        case ESC_KEY:
            exit(0);
        default:
            break;
    }
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
    	if (state == GLUT_DOWN){
            cout << "ABAJO\n";
    		scene.addCircle(x, y, 100);
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

void mousePosition(int x, int y) {
	scene.upddate_mouse_pos(x, y);
}

void render(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    scene.render();
    renderString(10, 24, "HOLA");

    glutSwapBuffers();
}

void initOCL(int argc, char** argv){
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Test Program");

    glutDisplayFunc(render);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutPassiveMotionFunc(mousePosition);

    glutTimerFunc(1000/60, timer, 0);

    glutMainLoop();
}

void initCosSinTables(){
    for (int ii = 0; ii < 180; ii++)   {
        float theta = 2.0f * PI * float(ii) / float(180);//get the current angle 
        cos_table[ii] = cos(theta);
        sin_table[ii] = sin(theta); 
    }
}

int main(int argc, char** argv){

    initCosSinTables();

	initOCL(argc, argv);
    
}