#include "main.h"
#include "scene.h"

#include <chrono>

#define ESC_KEY 27
#define SPC_KEY 32

void keyboard(unsigned char c, int x, int y);
void mouse(int button, int state, int x, int y);
void mousePosition( int x, int y);
void render(void);

float cos_table[NUM_CIR_SEG];
float sin_table[NUM_CIR_SEG];

float gravity = 10.0f * 75.0f;
int s_mul = 3;

Scene scene;

int _time, timebase = 0;
int frame = 0;

void specialInput(int key, int x, int y){
    switch(key){
        case GLUT_KEY_UP:
            s_mul++;
            break;
        case GLUT_KEY_DOWN:
            s_mul--;
            s_mul = max(s_mul, 1);
            break;
        case GLUT_KEY_LEFT:
            //do something here
            break;
        case GLUT_KEY_RIGHT:
            //do something here
            break;
    }
}

void keyboard(unsigned char c, int x, int y) {
    int char_value = c - '0';
    if(char_value > 0 && char_value < 10){
        scene.addCircle(x, y, char_value*s_mul, scene.static_circles);
        return;
    }
    switch(c){
        case 'b':
            scene.benchmark();
            break;
        case 'n':
            scene.reset();
            scene.normalDistribution();
            break;
        case 'p':
            printf("Mouse at: (%d, %d)\n", x, y);
            break;
        case 'm':
            scene.static_circles = !scene.static_circles;
            break;
        case SPC_KEY:
            scene.stepByStep = !scene.stepByStep;
            break; 
        case 'g':
            //gravity = gravity ? 0.0f : 10.0f * 75.0f;
            break;
        case 'f':
            scene.activated_physics = !scene.activated_physics;
            break;
        case ESC_KEY:
            exit(0);
            break;
        case 'c':
            scene.addCircle(x, y, 25, scene.static_circles);
            break;
        case 'r':
            scene.reset();
            break;
        case 's':
            save_screenshot("/images/frame" + to_string(scene.frame_count), WIDTH, HEIGHT);
            break;
        case 'w':
            scene.addWalls();
            break;
        default:
            break;
    }
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN){
            scene.addCircle(x, y, 3 + rand() % 27, scene.static_circles);
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
    scene.upddateMousePos(x, y);
}

void render(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    scene.render();

    glutSwapBuffers();
}

void idleFunction(){
    if(scene.benchmarking && scene.frame_count == scene.bench_frames){
        scene.elapsedTime();
        scene.benchmarking = false;
        exit(0);
    }
    glutPostRedisplay();
    frame++;
    _time = glutGet(GLUT_ELAPSED_TIME);

    if (_time - timebase > 1000.0f) {
        scene.frames_per_second = (frame * 1000.0f) / (_time - timebase);
        timebase = _time;
        frame = 0;
    }
}

void initOCL(int argc, char** argv){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(600, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Only Circles");

    glutDisplayFunc(render);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutPassiveMotionFunc(mousePosition);
    glutSpecialFunc(specialInput);

    glutIdleFunc(idleFunction);

    glutMainLoop();
}

void initCosSinTables(){
    for (int ii = 0; ii < NUM_CIR_SEG; ii++)   {
        float theta = 2.0f * PI * float(ii) / float(NUM_CIR_SEG);//get the current angle 
        cos_table[ii] = cos(theta);
        sin_table[ii] = sin(theta); 
    }
}

int main(int argc, char** argv){
    
    srand(1); //Static seed

    initCosSinTables();
    //scene.benchmark();
    initOCL(argc, argv);
}
