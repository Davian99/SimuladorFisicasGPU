################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../gpu_routines.cu 

CPP_SRCS += \
../circle.cpp \
../list_render_object.cpp \
../main.cpp \
../physics.cpp \
../scene.cpp 

O_SRCS += \
../circle.o \
../gpu_routines.o \
../list_render_object.o \
../main.o \
../physics.o \
../scene.o 

OBJS += \
./circle.o \
./gpu_routines.o \
./list_render_object.o \
./main.o \
./physics.o \
./scene.o 

CU_DEPS += \
./gpu_routines.d 

CPP_DEPS += \
./circle.d \
./list_render_object.d \
./main.d \
./physics.d \
./scene.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"-lGL -lGLU -lglut" -G -g -O1 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"-lGL -lGLU -lglut" -G -g -O1 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"-lGL -lGLU -lglut" -G -g -O1 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"-lGL -lGLU -lglut" -G -g -O1 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


