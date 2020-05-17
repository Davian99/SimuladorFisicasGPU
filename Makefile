CXX = g++
NVCC = nvcc

CFLAGS=-O3 -I.
CXXFLAGS=-O3 -I.
MORECXXFLAGS=-Wno-narrowing -Wunused
LIBS = -lpng -lm -lcudart -lGL -lGLU -lglut

SRC = main.o circle.o scene.o list_render_object.o physics.o gpu_routines.o

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(MORECXXFLAGS) -c -o $@ $<

main: $(SRC) 
	$(CXX) -o main  $(SRC) $(CXXFLAGS) $(LIBS) 
	./main

clean:
	rm -f *.o main
