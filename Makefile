CXX = g++
CC = gcc
NVCC = nvcc

#OpenCV
CXX_OPENCV_FLAGS+=`pkg-config opencv --cflags`
LD_OPENCV_FLAGS+=`pkg-config opencv --libs`


CFLAGS=-O3 -I. 
CXXFLAGS=-O3 -I.

LIBS = -lpng -lm -lcudart -lGL -lGLU -lglut

SRC = scene.o main.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

main: $(SRC) 
	$(CXX) -o main  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o main
