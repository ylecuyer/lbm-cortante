#include "helper.h"

#define GRID_SIZE 6

__global__ void stream(int X, int Y, int Z, float *cells_d, int current, int other, float *flags_d) {

	float e_x[19] = {1.0f, -1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f,  0.0f,  0.0f,  0.0f, 0.0f};
	float e_y[19] = {0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, -1.0f, -1.0f, 0.0f};
	float e_z[19] = {0.0f,  0.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f,  0.0f,  0.0f,  1.0f, -1.0f, 1.0f, -1.0f,  1.0f, -1.0f, 0.0f};
	int dfInv[19] = {1,0,3,2,5,4,11,10,13,12,7,6,9,8,17,16,15,14,18};

	const int FLUIDO  = 0;


	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if (i < X && j < Y && k > 0 &&  k < Z-1) {

		for (int l = 0; l < 19; l++) {
			int inv = dfInv[l];
			int a = i + e_x[inv];
			int b = j + e_y[inv];
			int c = k + e_z[inv];

			// Periodico en x
			if (a < 0) {
				a = X-1;
			}

			if (a > X-1) {
				a = 0;
			}

			// Periodico en y
			if (b < 0) {
				b = Y-1;
			}
			if( b > Y-1) {
				b = 0;
			}

			if(FLAGS_D(a, b, c) != FLUIDO){
				// Bounce - back
				CELLS_D(current, i, j, k, l) = CELLS_D(other, i, j, k, inv);
			}
			else{

				// Streaming - normal
				CELLS_D(current, i, j, k, l) = CELLS_D(other, a, b, c, l);
			}

		} // l

	}
}


void stream_wrapper(int X, int Y, int Z, float *cells_d, int current, int other, float *flags_d) {

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
	stream<<<grid_size, block_size>>>(X, Y, Z, cells_d, current, other, flags_d);


}
