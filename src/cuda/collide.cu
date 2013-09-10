#include "helper.h"

#define GRID_SIZE 6

__global__ void collide(int X, int Y, int Z, float *cells_d, float *fuerza_d, int current) {


	float w[19] = {(2./36.),(2./36.),(2./36.),(2./36.),(2./36.),(2./36.),
			(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),
			(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),
			(12./36.)};

	float e_x[19] = {1.0f, -1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f,  0.0f,  0.0f,  0.0f, 0.0f};
	float e_y[19] = {0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, -1.0f, -1.0f, 0.0f};
	float e_z[19] = {0.0f,  0.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f,  0.0f,  0.0f,  1.0f, -1.0f, 1.0f, -1.0f,  1.0f, -1.0f, 0.0f};


	const float cs = 0.57735026919f; // 1/sqrt(3)
	const float omega = 1.0;

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if (i < X && j < Y && k < Z) {



		// collision step

		float rho = 0.0, u_x=0.0, u_y=0.0, u_z=0.0;
		for (int l = 0; l < 19; l++) {
			const float fi = CELLS_D(current, i, j, k, l);
			rho += fi;
			u_x += e_x[l]*fi;
			u_y += e_y[l]*fi;
			u_z += e_z[l]*fi;
		}

		u_x = (u_x + (FUERZA_D(i, j, k, 0))*(1./2.))/rho;
		u_y = (u_y + (FUERZA_D(i, j, k, 1))*(1./2.))/rho;
		u_z = (u_z + (FUERZA_D(i, j, k, 2))*(1./2.))/rho;

		for (int l = 0; l < 19; l++) {
			const float tmp = (e_x[l]*u_x + e_y[l]*u_y + e_z[l]*u_z);
			// Función de equilibrio
			float feq = w[l] * rho * ( 1.0 -
					((3.0/2.0) * (u_x*u_x + u_y*u_y + u_z*u_z)) +
					(3.0 *     tmp) +
					((9.0/2.0) * tmp*tmp ) );
			// Fuerza por cada dirección i
			float v1[3]={0.0,0.0,0.0};
			v1[0]=(e_x[l]-u_x)/(cs*cs);
			v1[1]=(e_y[l]-u_y)/(cs*cs);
			v1[2]=(e_z[l]-u_z)/(cs*cs);

			v1[0]=v1[0]+(tmp*e_x[l])/(cs*cs*cs*cs);
			v1[1]=v1[1]+(tmp*e_y[l])/(cs*cs*cs*cs);
			v1[2]=v1[2]+(tmp*e_z[l])/(cs*cs*cs*cs);

			float Fi=0.0, tf=0.0;
			tf = (v1[0]*FUERZA_D(i, j, k, 0) + v1[1]*FUERZA_D(i, j, k, 1) + v1[2]*FUERZA_D(i, j, k, 2));
			Fi = (1.0-(omega/(2.0)))*w[l]*tf;

			CELLS_D(current, i, j, k, l) = CELLS_D(current, i, j, k, l) - omega*(CELLS_D(current, i, j, k, l) - feq) + Fi;
		}



	}
}


void collide_wrapper(int X, int Y, int Z, float *cells_d, float *fuerza_d, int current) {

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
	collide<<<grid_size, block_size>>>(X, Y, Z, cells_d, fuerza_d, current);


}
