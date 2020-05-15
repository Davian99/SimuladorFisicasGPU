################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../circle.cpp \
../list_render_object.cpp \
../main.cpp \
../physics.cpp \
../scene.cpp 

O_SRCS += \
../circle.o \
../list_render_object.o \
../main.o \
../physics.o \
../scene.o 

OBJS += \
./circle.o \
./list_render_object.o \
./main.o \
./physics.o \
./scene.o 

CPP_DEPS += \
./circle.d \
./list_render_object.d \
./main.d \
./physics.d \
./scene.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -lGL -lGLU -lglut -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


