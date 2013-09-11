#include "helper.h"

#define GRID_SIZE 6

// Condiciones de frontera para veocidad
__global__ void velNodoSuperior(float *cells_d, int current, int other, int X, int Y, int Z, float U, float V, float W)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = 0;

	if (i < X && j < Y) {


		// Calculate new distributions functions
		for(int l = 0; l < 19; l++)
			CELLS_D(current, i, j, k, l) = CELLS_D(other, i, j, k, l);

		float A=0.0, B=0.0, rho=0.0, Nx=0.0, Ny=0.0;
		A = CELLS_D(other, i, k, k, 0)
	    												+ CELLS_D(other, i, k, k, 1)
	    												+ CELLS_D(other, i, k, k, 2)
	    												+ CELLS_D(other, i, k, k, 3)
	    												+ CELLS_D(other, i, k, k, 6)
	    												+ CELLS_D(other, i, k, k, 10)
	    												+ CELLS_D(other, i, k, k, 11)
	    												+ CELLS_D(other, i, k, k, 7)
	    												+ CELLS_D(other, i, k, k, 18);

		B = CELLS_D(other, i, k, k, 4)
	    												+ CELLS_D(other, i, k, k, 8)
	    												+ CELLS_D(other, i, k, k, 12)
	    												+ CELLS_D(other, i, k, k, 14)
	    												+ CELLS_D(other, i, k, k, 16);

		rho = (A + 2.*B)/(W + 1.);



		Nx = (1./2.)*(CELLS_D(other, i, k, k, 0)+CELLS_D(other, i, k, k, 6)+CELLS_D(other, i, k, k, 7)-(CELLS_D(other, i, k, k, 1)+CELLS_D(other, i, k, k, 10)+CELLS_D(other, i, k, k, 11)))-(1./3.)*rho*U;
		Ny = (1./2.)*(CELLS_D(other, i, k, k, 2)+CELLS_D(other, i, k, k, 6)+CELLS_D(other, i, k, k, 10)-(CELLS_D(other, i, k, k, 3)+CELLS_D(other, i, k, k, 7)+CELLS_D(other, i, k, k, 11)))-(1./3.)*rho*V;

		CELLS_D(current, i, j, k, 5) = CELLS_D(other, i, k, k, 4)-(1./3.)*rho*W;
		CELLS_D(current, i, j, k, 9) = CELLS_D(other, i, k, k, 12)+(rho/6.)*(-W+U)-Nx;
		CELLS_D(current, i, j, k, 13) = CELLS_D(other, i, k, k, 8)+(rho/6.)*(-W-U)+Nx;
		CELLS_D(current, i, j, k, 15) = CELLS_D(other, i, k, k, 16)+(rho/6.)*(-W+V)-Ny;
		CELLS_D(current, i, j, k, 17) = CELLS_D(other, i, k, k, 14)+(rho/6.)*(-W-V)+Ny;
	}
}

__global__ void velNodoInferior(float *cells_d, int current, int other, int X, int Y, int Z, float U, float V, float W)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = Z-1;

	if (i < X && j < Y) {

		// Calculate new distributions functions
		for (int l = 0; l < 19; l++)
			CELLS_D(other, i, j, k, l) = CELLS_D(current, i, j, k, l);


		float A=0.0, B=0.0, rho=0.0, Nx=0.0, Ny=0.0;
		A = CELLS_D(current, i, j, k, 0)
														+ CELLS_D(current, i, j, k, 1)
														+ CELLS_D(current, i, j, k, 2)
														+ CELLS_D(current, i, j, k, 3)
														+ CELLS_D(current, i, j, k, 6)
														+ CELLS_D(current, i, j, k, 7)
														+ CELLS_D(current, i, j, k, 10)
														+ CELLS_D(current, i, j, k, 11)
														+ CELLS_D(current, i, j, k, 18);

		B = CELLS_D(current, i, j, k, 5)
		    										+CELLS_D(current, i, j, k, 9)
		    										+CELLS_D(current, i, j, k, 13)
		    										+CELLS_D(current, i, j, k, 15)
		    										+CELLS_D(current, i, j, k, 17);
		rho = (A + 2.*B)/(1. - W);

		Nx=(1./2.)*(CELLS_D(current, i, j, k, 0)+CELLS_D(current, i, j, k, 6)+CELLS_D(current, i, j, k, 7)-(CELLS_D(current, i, j, k, 1)+CELLS_D(current, i, j, k, 10)+CELLS_D(current, i, j, k, 11)))-(1./3.)*rho*-U;
		Ny=(1./2.)*(CELLS_D(current, i, j, k, 2)+CELLS_D(current, i, j, k, 6)+CELLS_D(current, i, j, k, 10)-(CELLS_D(current, i, j, k, 3)+CELLS_D(current, i, j, k, 7)+CELLS_D(current, i, j, k, 11)))-(1./3.)*rho*V;

		CELLS_D(other, i, j, k, 4) =CELLS_D(current, i, j, k, 5)+(1./3.)*rho*W;
		CELLS_D(other, i, j, k, 8) =CELLS_D(current, i, j, k, 13)+(rho/6.)*(W-U)-Nx;
		CELLS_D(other, i, j, k, 12) =CELLS_D(current, i, j, k, 9)+(rho/6.)*(W+U)+Nx;
		CELLS_D(other, i, j, k, 14) =CELLS_D(current, i, j, k, 17)+(rho/6.)*(W+V)-Ny;
		CELLS_D(other, i, j, k, 16) =CELLS_D(current, i, j, k, 15)+(rho/6.)*(W-V)+Ny;
	}
}

void wrapper_velNodoSuperior(float *cells_d, int current, int other, int X, int Y, int Z, float U, float V, float W) {

	//X*Y*Z = 9261;
	//Maximum number of threads per block:           1024

	dim3 grid_size;
	grid_size.x = GRID_SIZE;
	grid_size.y = GRID_SIZE;

	dim3 block_size;
	// 1000 threads per blocks
	block_size.x = 10;
	block_size.y = 10;

	//Launch kernel
	velNodoSuperior<<<grid_size, block_size>>>(cells_d, current, other, X, Y, Z, U, V, W);

}

void wrapper_velNodoInferior(float *cells_d, int current, int other, int X, int Y, int Z, float U, float V, float W) {

	//X*Y*Z = 9261;
	//Maximum number of threads per block:           1024

	dim3 grid_size;
	grid_size.x = GRID_SIZE;
	grid_size.y = GRID_SIZE;

	dim3 block_size;
	// 1000 threads per blocks
	block_size.x = 10;
	block_size.y = 10;


	//Launch kernel
	velNodoInferior<<<grid_size, block_size>>>(cells_d, current, other, X, Y, Z, U, V, W);

}
