#include "scene.h"

Scene::Scene(){
	this->frame_count = 0;
	this->phy = Physics(this);
	this->activated_physics = true;
}

void Scene::render(){
	if(activated_physics){
		this->frame_count++;
		this->phy.step();
	}
	this->lro.renderAll();
	this->renderDefaultText();
}

void Scene::upddateMousePos(int x, int y){
	this->mx = x;
	this->my = y;
}	

void Scene::addCircle(int x, int y, int r){
	this->lro.addCircle(x, y, r);
}

void Scene::renderDefaultText(){
	string frames = "Frame: " + to_string(this->frame_count);
	renderString(10, 24, frames);
	string object_count = "Objects: " + to_string(this->lro.size());
	renderString(10, 48, object_count);
}

void Scene::reset(){
	this->frame_count = 0;
	this->lro.clear();
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
   string path = "/home/davian/Uni20/GPU/images/" + filename;
   FILE *filePtr = fopen(path.c_str(), "wb");
   if (!filePtr) return false;
	
   
   unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
   unsigned char header[6] = { w%256,w/256,
			       h%256,h/256,
			       24,0};
   // We write the headers
   fwrite(TGAheader,	sizeof(unsigned char),	12,	filePtr);
   fwrite(header,	sizeof(unsigned char),	6,	filePtr);
   // And finally our image data
   fwrite(dataBuffer,	sizeof(GLubyte),	nSize,	filePtr);
   fclose(filePtr);

   free(dataBuffer);

  return true;
}