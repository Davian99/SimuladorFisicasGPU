CXX = g++
NVCC = nvcc

CFLAGS=-O1 -I. 
CXXFLAGS=-O1 -I.

LIBS = -lpng -lm -lcudart -lGL -lGLU -lglut

SRC = main.o circle.o scene.o list_render_object.o

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

main: $(SRC) 
	$(CXX) -o main  $(SRC) $(CXXFLAGS) $(LIBS) 
	./main

clean:
	rm -f *.o main
