#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fluid.h"
#include "fronteras.h"
#include "debug.h"
#include "helper.h"

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

void fluid::inicializar(int x, int y, int z)
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

	vel = new float***[x];
	fuerza = new float***[x];
	rho = new float**[x];
	for(int i=0;i<x;i++)
	{
		vel[i] = new float**[y];
		fuerza[i] = new float**[y];
		rho[i] = new float*[y];
		for(int j = 0; j<y;j++)
		{
			vel[i][j] = new float*[z];
			fuerza[i][j] = new float*[z];
			rho[i][j] = new float[z];
			for(int k=0; k<z ; k++)
			{
				vel[i][j][k] = new float[3];
				fuerza[i][j][k] = new float[3];
				rho[i][j][k] = 0;
			}
		}
	}


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

	calcularMacro();
}

void fluid::stream()
{

	for (int i=0;i<X;i++)
		for (int j=0;j<Y;j++)
			for (int k=1;k<Z-1;k++)
				for (int l=0;l<19;l++) {
					int inv = dfInv[l];
					int a = i + e_x[inv];
					int b = j + e_y[inv];
					int c = k + e_z[inv];

					// Periodico en x
					if(a<0){a=X-1;}
					if(a>(X-1)){a=0;}

					// Periodico en y
					if(b<0){b=Y-1;}
					if(b>(Y-1)){b=0;}

					if(FLAGS(a, b, c) != FLUIDO){
						// Bounce - back
						CELLS(current, i, j, k, l) = CELLS(other, i, j, k, inv);
					}
					else{

						// Streaming - normal
						CELLS(current, i, j, k, l) = CELLS(other, a, b, c, l);
					}

				}//Stream

	for (int i=0;i<X;i++)
		for (int j=0;j<Y;j++){
		velNodoInferior(cells, current, other, X, Y, Z, i, j, 0, U, V, W);
		velNodoSuperior(cells, current, other, X, Y, Z, i, j, Z-1, U, V, W);
		}
}

void fluid::collide()
{
	// collision step
			for (int i=0;i<X;i++)
				for (int j=0;j<Y;j++)
					for (int k=0;k<Z;k++) {

						float rho = 0.0, u_x=0.0, u_y=0.0, u_z=0.0;
						for (int l=0;l<19;l++) {
							const float fi = CELLS(current, i, j, k, l);
							rho += fi;
							u_x += e_x[l]*fi;
							u_y += e_y[l]*fi;
							u_z += e_z[l]*fi;
						}

						u_x = (u_x + (fuerza[i][j][k][0])*(1./2.))/rho;
						u_y = (u_y + (fuerza[i][j][k][1])*(1./2.))/rho;
						u_z = (u_z + (fuerza[i][j][k][2])*(1./2.))/rho;

						for (int l=0;l<19;l++) {
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
							tf = (v1[0]*fuerza[i][j][k][0] + v1[1]*fuerza[i][j][k][1] + v1[2]*fuerza[i][j][k][2]);
							Fi = (1.0-(omega/(2.0)))*w[l]*tf;

							CELLS(current, i, j, k, l) = CELLS(current, i, j, k, l) - omega*(CELLS(current, i, j, k, l) - feq) + Fi;
						}
					} // ijk
			// We're done for one time step, switch the grid...
			other = current;
			current = (current+1)%2;
}

void fluid::calcularMacro()
{
	for(int i = 0 ;i<X;i++)
			for(int j = 0 ;j<Y;j++)
				for(int k = 0 ;k<Z;k++)
				{
					float rhol=0.0;
					float u_x=0.0;
					float u_y=0.0;
					float u_z=0.0;

					for(int l = 0 ;l<19;l++){
						const float fi = CELLS(current, i, j, k, l);
						rhol+= fi;
						u_x+=fi*e_x[l];
						u_y+=fi*e_y[l];
						u_z+=fi*e_z[l];
					}

					rho[i][j][k] = rhol;
					vel[i][j][k][0] = (u_x+fuerza[i][j][k][0])/rhol;
					vel[i][j][k][1] = (u_y+fuerza[i][j][k][1])/rhol;
					vel[i][j][k][2] = (u_z+fuerza[i][j][k][2])/rhol;
					fuerza[i][j][k][0]=0.0;
					fuerza[i][j][k][1]=0.0;
					fuerza[i][j][k][2]=0.0;
				}
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
						fprintf(archivo, "%f \n", darDensidad(i,j,k));
					}
					else{
						fprintf(archivo, "%f \n", 0.0);
					}
				}


		// Velocidad
		fprintf(archivo, "\nVECTORS Velocidad float\n");
		for(int k = 0 ;k<Z;k++){
			for(int j = 0 ;j<Y;j++)
				for(int i = 0 ;i<X;i++)
					fprintf(archivo, "%f %f %f\n", vel[i][j][k][0], vel[i][j][k][1], vel[i][j][k][2]);
				}

		// Escribir vectores fuerza en el algoritmo
		fprintf(archivo, "VECTORS fuerza float\n");
		for(int k = 0 ;k<Z;k++)
			for(int j = 0 ;j<Y;j++)
				for(int i = 0 ;i<X;i++)
				{
					fprintf(archivo, "%f %f %f\n", fuerza[i][j][k][0], fuerza[i][j][k][0], fuerza[i][j][k][0]);
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
	fuerza[x][y][z][0] = f[0];
	fuerza[x][y][z][1] = f[1];
	fuerza[x][y][z][2] = f[2];
}

void fluid::addFuerza(int x, int y, int z, float f[3])
{
	fuerza[x][y][z][0] += f[0];
	fuerza[x][y][z][1] += f[1];
	fuerza[x][y][z][2] += f[2];
}


void fluid::setVelocidad(float u)
{
	U = u;
}
