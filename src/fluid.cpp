#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fluid.h"
#include "fronteras.h"
#include "debug.h"
#include "helper.h"
#include "collide.h"
#include "stream.h"
#include "calcular_macro.h"
#include "vel_nodo.h"

float w[19] = {(2./36.),(2./36.),(2./36.),(2./36.),(2./36.),(2./36.),
		(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),
		(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),
		(12./36.)};

float e_x[19] = {1.0f, -1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f,  0.0f,  0.0f,  0.0f, 0.0f};
float e_y[19] = {0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, -1.0f, -1.0f, 0.0f};
float e_z[19] = {0.0f,  0.0f, 0.0f,  0.0f, 1.0f, -1.0f, 0.0f,  0.0f, 1.0f, -1.0f,  0.0f,  0.0f,  1.0f, -1.0f, 1.0f, -1.0f,  1.0f, -1.0f, 0.0f};
int dfInv[19] = {1,0,3,2,5,4,11,10,13,12,7,6,9,8,17,16,15,14,18};

const int FLUIDO  = 0, TOP = 1, BOTTOM  = 2, NOSLIP = 3;
const float V = 0.00, W = 0.00, cs=1.0/sqrt(3.0);
const float   omega = 1.0;
int current = 0, other = 1;

fluid::~fluid() {

	free(cells);
	free(flags);
	free(vel);
	free(rho);
	free(fuerza);

}


fluid::fluid(int x, int y, int z)
{
	X = x;
	Y = y;
	Z = z;

	flags = (float*)malloc(X*Y*Z*sizeof(float));

	if (flags == NULL) {

		_DEBUG("Error allocating flags");
		exit(-1);
	}

	memset(flags, 0, X*Y*Z*sizeof(float));

	vel = (float*)malloc(X*Y*Z*3*sizeof(float));

	if (vel == NULL) {

		_DEBUG("Error allocating vel");
		exit(-1);
	}

	memset(vel, 0, X*Y*Z*3*sizeof(float));

	rho = (float*)malloc(X*Y*Z*sizeof(float));

	if (rho == NULL) {

		_DEBUG("Error allocating rho");
		exit(-1);
	}

	memset(rho, 0, X*Y*Z*sizeof(float));

	fuerza = (float*)malloc(X*Y*Z*3*sizeof(float));

	if (fuerza == NULL) {

		_DEBUG("Error allocating fuerza");
		exit(-1);
	}

	memset(fuerza, 0, X*Y*Z*3*sizeof(float));

	cells = (float*)malloc(2*X*Y*Z*19*sizeof(float));

	if (cells == NULL) {

		_DEBUG("Error allocating cells");
		exit(-1);
	}

	for(int s = 0; s < 2; s++)
	{
		for(int i = 0; i < X; i++)
		{
			for(int j = 0; j < Y; j++)
			{
				for(int k = 0; k < Z; k++)
				{
					for(int l = 0; l < 19; l++)
					{
						CELLS(s, i, j, k, l) = w[l];
					}
				}
			}
		}
	}

	//Calcular macro moved to cortante
	//calcularMacro(X, Y, Z, cells_d, current, rho_d, vel_d, fuerza_d);

}


float* fluid::get_cells(void) {
	return cells;
}

float* fluid::get_flags(void) {
	return flags;
}

float* fluid::get_vel(void) {
	return vel;
}

float* fluid::get_rho(void) {
	return rho;
}

float* fluid::get_fuerza(void) {
	return fuerza;
}

void fluid::stream(float *cells_d, float *flags_d)
{

	stream_wrapper(X, Y, Z, cells_d, current, other, flags_d);

			wrapper_velNodoInferior(cells_d, current, other, X, Y, Z, U, V, W);
			wrapper_velNodoSuperior(cells_d, current, other, X, Y, Z, U, V, W);
}

void fluid::collide(float *cells_d, float*fuerza_d)
{
	collide_wrapper(X, Y, Z, cells_d, fuerza_d, current);

	// We're done for one time step, switch the grid...
	other = current;
	current = (current+1)%2;
}

void fluid::calcularMacro(float *cells_d, float *rho_d, float *vel_d, float *fuerza_d)
{

	calcular_macro_wrapper(X, Y, Z, cells_d, current, rho_d, vel_d, fuerza_d);

}

// Save fluid in structured grid format .vts
int fluid::guardar(int s) {

	FILE *archivo;/*El manejador de archivo*/
	char ruta[80];

	sprintf(ruta, "temp/fluido-%d.vtk", s);

	archivo=fopen(ruta, "w");
	if(archivo==NULL){/*Si no lo logramos abrir, salimos*/
		printf("No se puede guardar archivo\n");
		return 1;}
	else{
		// Escribir datos al archivo
		// 1. Escribir cabecera.
		fprintf(archivo, "# vtk DataFile Version 3.0\n");
		fprintf(archivo, "vtk output\n");
		fprintf(archivo, "ASCII\n");

		fprintf(archivo, "DATASET STRUCTURED_POINTS\n");
		fprintf(archivo, "DIMENSIONS %d %d %d\n", X, Y, Z);
		fprintf(archivo, "ORIGIN 0 0 0\n");
		fprintf(archivo, "SPACING 1 1 1\n");


		// 3. Escribir datos sobre puntos
		// Densidad
		fprintf(archivo, "POINT_DATA %d\n", X*Y*Z);
		fprintf(archivo, "SCALARS Densidad float\n");
		fprintf(archivo, "LOOKUP_TABLE default\n");
		for(int k = 0 ;k<Z;k++){
			for(int j = 0 ;j<Y;j++)
				for(int i = 0 ;i<X;i++)
					if(FLAGS(i, j, k) == FLUIDO)
					{
						fprintf(archivo, "%e \n", darDensidad(i,j,k));
					}
					else{
						fprintf(archivo, "%e \n", 0.0);
					}
		}


		// Velocidad
		fprintf(archivo, "\nVECTORS Velocidad float\n");
		for(int k = 0 ;k<Z;k++){
			for(int j = 0 ;j<Y;j++)
				for(int i = 0 ;i<X;i++)
					fprintf(archivo, "%e %e %e\n", VEL(i, j, k, 0), VEL(i, j, k, 1), VEL(i, j, k, 2));
		}

		// Escribir vectores fuerza en el algoritmo
		fprintf(archivo, "VECTORS fuerza float\n");
		for(int k = 0 ;k<Z;k++)
			for(int j = 0 ;j<Y;j++)
				for(int i = 0 ;i<X;i++)
				{
					fprintf(archivo, "%e %e %e\n", FUERZA(i, j, k, 0), FUERZA(i, j, k, 0), FUERZA(i, j, k, 0));
				}
		fclose(archivo);/*Cerramos el archivo*/
		return 0;
	}

}


float fluid::darVelocidad(int x, int y, int z, int f)
{
	return -1; //vel[x][y][z][f];
}


float fluid::darDensidad(int x, int y, int z)
{
	float rho = 0.0;
	for(int l = 0; l < 19 ; l++)
	{
		rho += CELLS(current, x, y, z, l);
	}
	return rho;
}

void fluid::setFuerza(int x, int y, int z, float f[3])
{
	FUERZA(x, y, z, 0) = f[0];
	FUERZA(x, y, z, 1) = f[1];
	FUERZA(x, y, z, 2) = f[2];
}

void fluid::addFuerza(int x, int y, int z, float f[3])
{
	FUERZA(x, y, z, 0) += f[0];
	FUERZA(x, y, z, 1) += f[1];
	FUERZA(x, y, z, 2) += f[2];
}


void fluid::setVelocidad(float u)
{
	U = u;
}
