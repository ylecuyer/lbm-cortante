################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cortante.cpp \
../src/fluid.cpp \
../src/fronteras.cpp 

OBJS += \
./src/cortante.o \
./src/fluid.o \
./src/fronteras.o 

CPP_DEPS += \
./src/cortante.d \
./src/fluid.d \
./src/fronteras.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include" -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include/cuda" -G -g -O0 -m32 -gencode arch=compute_10,code=sm_10 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include" -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include/cuda" -G -g -O0 -m32 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


