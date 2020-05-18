CXX = g++
NVCC = nvcc

CFLAGS=-O3 -I.
CXXFLAGS=-O3 -I.
MORECXXFLAGS=-Wno-narrowing -Wunused
LIBS = -lcuda -lcudart -lGL -lGLU -lglut

SRC = circle.o gpu_routines.o list_circles.o main.o physics.o scene.o

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(MORECXXFLAGS) -c -o $@ $<

main: $(SRC) 
	$(CXX) -o main  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o main
