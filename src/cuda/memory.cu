#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void alloc_memory_GPU(int X, int Y, int Z, float **cells_d, float **flags_d, float **vel_d, float **rho_d, float **fuerza_d) {

	gpuErrchk( cudaMalloc(cells_d, 2*X*Y*Z*19*sizeof(float)) );
	gpuErrchk( cudaMalloc(flags_d, X*Y*Z*sizeof(float)) );
	gpuErrchk( cudaMalloc(vel_d, X*Y*Z*3*sizeof(float)) );
	gpuErrchk( cudaMalloc(rho_d, X*Y*Z*sizeof(float)) );
	gpuErrchk( cudaMalloc(fuerza_d, X*Y*Z*3*sizeof(float)) );

}

void free_memory_GPU(float *cells_d, float *flags_d, float *vel_d, float *rho_d, float *fuerza_d) {

	gpuErrchk( cudaFree(cells_d) );
	gpuErrchk( cudaFree(flags_d) );
	gpuErrchk( cudaFree(vel_d) );
	gpuErrchk( cudaFree(rho_d) );
	gpuErrchk( cudaFree(fuerza_d) );

}

void send_data_to_GPU(int X, int Y, int Z, float *cells, float *cells_d, float *flags, float *flags_d, float *vel, float *vel_d, float *rho, float *rho_d, float *fuerza, float *fuerza_d) {

	gpuErrchk( cudaMemcpy(cells_d, cells, 2*X*Y*Z*19*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(flags_d, flags, X*Y*Z*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(vel_d, vel, X*Y*Z*3*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(rho_d, rho, X*Y*Z*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(fuerza_d, fuerza, X*Y*Z*3*sizeof(float), cudaMemcpyHostToDevice) );

}

void retrieve_data_from_GPU(int X, int Y, int Z, float *cells, float *cells_d, float *flags, float *flags_d, float *vel, float *vel_d, float *rho, float *rho_d, float *fuerza, float *fuerza_d) {

	gpuErrchk( cudaMemcpy(cells, cells_d, 2*X*Y*Z*19*sizeof(float), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(flags, flags_d, X*Y*Z*sizeof(float), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(vel, vel_d, X*Y*Z*3*sizeof(float), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(rho, rho_d, X*Y*Z*sizeof(float), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(fuerza, fuerza_d, X*Y*Z*3*sizeof(float), cudaMemcpyDeviceToHost) );

}
