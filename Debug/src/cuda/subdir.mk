################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cuda/collide.cu \
../src/cuda/memory.cu \
../src/cuda/stream.cu 

CU_DEPS += \
./src/cuda/collide.d \
./src/cuda/memory.d \
./src/cuda/stream.d 

OBJS += \
./src/cuda/collide.o \
./src/cuda/memory.o \
./src/cuda/stream.o 


# Each subdirectory must supply rules for building sources it contributes
src/cuda/%.o: ../src/cuda/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include" -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include/cuda" -G -g -O0 -m32 -gencode arch=compute_10,code=sm_10 -odir "src/cuda" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc --compile -G -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include" -I"/home/ylecuyer/cuda-workspace/lbm-cortante/include/cuda" -O0 -g -gencode arch=compute_10,code=compute_10 -gencode arch=compute_10,code=sm_10 -m32  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


