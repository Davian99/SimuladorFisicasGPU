#include "main.h"
#include "scene.h"

#define TAB_KEY 9
#define ESC_KEY 27
#define SPC_KEY 32

void keyboard(unsigned char c, int x, int y);
void mouse(int button, int state, int x, int y);
void mousePosition( int x, int y);
void render(void);

float cos_table[NUM_CIR_SEG];
float sin_table[NUM_CIR_SEG];

float gravity = 10.0f * 75.0f;
bool use_gpu = true;
bool random_solve_cols = true;
int s_mul = 3;
int separation = 8;

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
            separation--;
            separation = max(separation, 1);
            if(scene.benchmarking)
                scene.benchmark(separation);
            break;
        case GLUT_KEY_RIGHT:
            separation++;
            if(scene.benchmarking)
                scene.benchmark(separation);
            break;
    }
}

void keyboard(unsigned char c, int x, int y) {
    //printf("%d\n", c);
    int char_value = c - '0';
    if(char_value > 0 && char_value < 10){
        int mod = glutGetModifiers();
        if(mod & GLUT_ACTIVE_ALT){
            //printf("SHIFT + %d\n", char_value);
            scene.addSpawner(x, y, char_value*s_mul);
        }
        else
            scene.addCircle(x, y, char_value*s_mul, scene.static_circles);
        return;
    }
    switch(c){
        case TAB_KEY:
            scene.render_extra = !scene.render_extra;
            break;
        case 'b':
            scene.benchmark(separation);
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
            scene.activated_physics = false;
            scene.stepByStep = !scene.stepByStep;
            break; 
        case 'g':
            gravity = gravity ? 0.0f : 10.0f * 75.0f;
            break;
        case 'f':
            scene.activated_physics = !scene.activated_physics;
            break;
        case ESC_KEY:
            exit(0);
            break;
        case 'r':
            scene.reset();
            break;
        case 's':
            save_screenshot("frame" + to_string(scene.frame_count) + ".tga", WIDTH, HEIGHT);
            break;
        case 'w':
            scene.addWalls();
            break;
        case 'z':
            use_gpu = !use_gpu;
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
        //scene.benchmarking = false;
        //exit(0);
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

    int opt, i = 1;
    while ((opt = getopt(argc, argv, "cgbhr")) != -1) {
        switch (opt) {
            case 'r':
                random_solve_cols = false;
                break;
            case 'c':
                use_gpu = false;
                break;
            case 'g':
                use_gpu = true;
                break;
            case 'b':{
                    if(argc > i+1)
                        separation = atoi(argv[i+1]);
                    //printf("sep=%d, argv[%d]=%s\n", sep, i+1, argv[i+1]);
                    scene.benchmark(separation);
                    scene.no_ogl();
                    break;
                }
           
            case 'h':
                printf("Physics Simulator launch guide:\n");
                printf(" -g for GPU MODE\n");
                printf(" -c for CPU MODE\n");
                printf(" -b for benchmark mode (only simulation)\n");
                printf("     Add separation Integer after -b for heavier benchmarking)\n");
                printf("\nPhysics Simulator use guide:\n");
                printf(" - '1' to '9' to create a circle\n");
                printf(" - 'b' initiate benchmark\n");
                printf(" - 'r' reset all\n");
                printf(" - 'n' for reset\n");
                printf(" - 'm' for create static circles (press again to revert)\n");
                printf(" - 'f' stop or continue simulation\n");
                printf(" - 'space' step by step simulation\n");
                printf(" - 'g' enable or disable gravity\n");
                printf(" - 's' save screenshot\n");
                printf(" - 'w' add walls to scene\n");
                printf(" - 'z' change between GPU and CPU mode\n");
                break;
            default:
                printf("%d\n", opt);
                break;  
        }
        i++;
    }
    //If no benchmark mode, initiate OpenGL
    initOCL(argc, argv);    
    return 0;
}
