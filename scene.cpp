#include "scene.h"

Scene::Scene(){
	this->n_collisions = 0;
	this->frame_count = 0;
	this->phy = Physics(this);
	this->activated_physics = true;
	this->bench_frames = 300;
	this->frames_per_second = 0.0f;
}

void Scene::render(){
	if(activated_physics){
		this->frame_count++;
		this->phy.step();
	}
	else if(stepByStep){
		this->frame_count++;
		this->phy.step();
		stepByStep = false;
	}
	this->lro.renderAll();
	this->renderDefaultText();
}

void Scene::upddateMousePos(int x, int y){
	this->mx = x;
	this->my = y;
}	

void Scene::addCircle(int x, int y, int r, bool _static){
	this->lro.addCircle(x, y, r, _static);
}

void Scene::renderDefaultText(){
	string frames = "Frame: " + to_string(this->frame_count);
	renderString(10, 24, frames);
	string object_count = "Objects: " + to_string(this->lro.size());
	renderString(10, 48, object_count);
	string number_collisions = "Collisions: " + to_string(this->n_collisions);
	renderString(10, 72, number_collisions);

	string frames_ps = "FPS: " + to_string(this->frames_per_second);
	renderString(10, 96, frames_ps);
}

void Scene::reset(){
	this->frame_count = 0;
	this->lro.clear();
}

void Scene::addWalls(){
	for(int posx = 0; posx <= WIDTH; posx += 10)
		this->lro.addCircle(posx, HEIGHT+99, 100, true);

	for(int posx = 0; posx <= WIDTH; posx += 20)
		this->lro.addCircle(0-99, posx, 100, true);

	for(int posx = 0; posx <= WIDTH; posx += 20)
		this->lro.addCircle(WIDTH+99, posx, 100, true);
}

void Scene::normalDistribution(){
	this->addWalls();
	bool shift = true;
	for(int posy = 215; posy <= HEIGHT - 300; posy += 60){
		shift = !shift;
		for(int posx = 100; posx <= WIDTH-100; posx += 60){
			this->lro.addCircle(posx + shift * 30, posy, 5*4, true);
		}
	}
	for(int posx = 50; posx < WIDTH; posx += 50)
		for(int posy = 0; posy < 250; posy += 6)
			this->lro.addCircle(posx, HEIGHT - posy, 3, true);


	for(int posx = 175, posy = 0; posx < 380; posx += 3, posy += 2){
		this->lro.addCircle(posx, posy, 2, true);
	}

	for(int posx = 625, posy = 0; posx > 420; posx -= 3, posy += 2){
		this->lro.addCircle(posx, posy, 2, true);
	}
}

void Scene::benchmark(){
	this->benchmarking = true;
	this->reset();
	this->addWalls();
	bool shift = true;
	for(int posy = 185; posy <= HEIGHT - 150; posy += 30){
		shift = !shift;
		for(int posx = 15; posx <= WIDTH-15; posx += 30){
			this->lro.addCircle(posx + shift * 15, posy, 5*2, true);
		}
	}

	for(int x = 12; x < WIDTH - 10; x += 8){
		for(int y = 10; y < 150; y += 8){
			this->lro.addCircle(x, y, 2, false);
		}
	}
	this->begin = steady_clock::now();
}

void Scene::elapsedTime(){
	steady_clock::time_point end = steady_clock::now();
	duration<double, std::micro> microseconds = end - this->begin;
	float seconds = microseconds.count() / 1000000.0f;
	printf("Benchmark with %d frames completed in %f seconds giving FPS = %f\n", 
		this->bench_frames, seconds, this->bench_frames / seconds);
}

float pixelsTo(float px){
	return px / ((WIDTH + HEIGHT) / 2);
}

void coordTo(int x, int y, float &fx, float &fy){
	fx = ((x * 2.0f) / WIDTH) - 1;
	fy = -1.0f * (((y * 2.0f) / HEIGHT) - 1);
}

void renderString(int x, int y, string s){
	glColor3f(0.8f, 0.8f, 0.8f);
	float fx, fy;
	coordTo(x, y, fx, fy);
	glRasterPos2f(fx, fy);
	unsigned int l = s.size();
	for(unsigned int i = 0; i < l; ++i)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18 , s[i]);
}

void glVertexC(float x, float y){
	float fx = ((x * 2.0f) / WIDTH) - 1;
	float fy = -1.0f * (((y * 2.0f) / HEIGHT) - 1);
	glVertex2f(fx, fy);
}

float degToRad(float deg){
	return deg * PI / 180.f;
}

bool save_screenshot(string filename, int w, int h) {	
  //This prevents the images getting padded 
 // when the width multiplied by 3 is not a multiple of 4
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  int nSize = w*h*3;
  // First let's create our buffer, 3 channels per Pixel
  char* dataBuffer = (char*)malloc(nSize*sizeof(char));

  if (!dataBuffer) return false;
   
   // Let's fetch them from the backbuffer	
   // We request the pixels in GL_BGR format, thanks to Berzeger for the tip
   glReadPixels((GLint)0, (GLint)0,
		(GLint)w, (GLint)h,
		 GL_BGR, GL_UNSIGNED_BYTE, dataBuffer);
	
   //Now the file creation
   //string path = "/home/davian/Uni20/GPU/images/" + filename;
   string path = filename;
   FILE *filePtr = fopen(path.c_str(), "wb");
   if (!filePtr) return false;
	
 
   unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
   unsigned char header[6] = { w%256,w/256,h%256,h/256,24,0};
   // We write the headers
   fwrite(TGAheader,	sizeof(unsigned char),	12,	filePtr);
   fwrite(header,	sizeof(unsigned char),	6,	filePtr);
   // And finally our image data
   fwrite(dataBuffer,	sizeof(GLubyte),	nSize,	filePtr);
   fclose(filePtr);

   free(dataBuffer);

  return true;
}
