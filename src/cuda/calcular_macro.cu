#include "helper.h"

#define GRID_SIZE 6

__global__ void calcular_macro(int X, int Y, int Z, float *cells_d, int current, float *rho_d, float *vel_d, float *fuerza_d) {


	float e_x[19] = {1.0f, -1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f,  0.0f,  0.0f,  0.0f, 0.0f};
	float e_y[19] = {0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, -1.0f, -1.0f, 0.0f};
	float e_z[19] = {0.0f,  0.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f,  0.0f,  0.0f,  1.0f, -1.0f, 1.0f, -1.0f,  1.0f, -1.0f, 0.0f};


	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if (i < X && j < Y && k < Z) {

		float rhol=0.0;
		float u_x=0.0;
		float u_y=0.0;
		float u_z=0.0;

		for(int l = 0; l < 19; l++){
			const float fi = CELLS_D(current, i, j, k, l);
			rhol += fi;
			u_x += fi*e_x[l];
			u_y += fi*e_y[l];
			u_z += fi*e_z[l];
		}

		RHO_D(i, j, k) = rhol;

		VEL_D(i, j, k, 0) = (u_x + FUERZA_D(i, j, k, 0))/rhol;
		VEL_D(i, j, k, 1) = (u_y + FUERZA_D(i, j, k, 1))/rhol;
		VEL_D(i, j, k, 2) = (u_z + FUERZA_D(i, j, k, 2))/rhol;

		FUERZA_D(i, j, k, 0)=0.0;
		FUERZA_D(i, j, k, 1)=0.0;
		FUERZA_D(i, j, k, 2)=0.0;

	}
}


void calcular_macro_wrapper(int X, int Y, int Z, float *cells_d, int current, float *rho_d, float *vel_d, float *fuerza_d) {


	//X*Y*Z = 9261;
	//Maximum number of threads per block:           1024

	dim3 grid_size;
	grid_size.x = GRID_SIZE;
	grid_size.y = GRID_SIZE;
	grid_size.z = GRID_SIZE;

	dim3 block_size;
	// 1000 threads per blocks
	block_size.x = 10;
	block_size.y = 10;
	block_size.z = 10;

	//Launch kernel
	calcular_macro<<<grid_size, block_size>>>(X, Y, Z, cells_d, current, rho_d, vel_d, fuerza_d);


}
