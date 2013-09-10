#ifndef _MEMORY_H_
#define _MEMORY_H_

void alloc_memory_GPU(int X, int Y, int Z, float **cells_d, float **flags_d, float **vel_d, float **rho_d, float **fuerza_d);
void free_memory_GPU(float *cells_d, float *flags_d, float *vel_d, float *rho_d, float *fuerza_d);

void send_data_to_GPU(int X, int Y, int Z, float *cells, float *cells_d, float *flags, float *flags_d, float *vel, float *vel_d, float *rho, float *rho_d, float *fuerza, float *fuerza_d);
void retrieve_data_from_GPU(int X, int Y, int Z, float *cells, float *cells_d, float *flags, float *flags_d, float *vel, float *vel_d, float *rho, float *rho_d, float *fuerza, float *fuerza_d);


#endif
