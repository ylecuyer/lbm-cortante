/*
 * fronteras.cpp
 *
 *  Created on: Mar 23, 2011
 *      Author: oscar
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fronteras.h"
#include "helper.h"

// Condiciones de frontera para veocidad
void velNodoSuperior(float *cells, int current, int other, int X, int Y, int Z, int i, int j, int k, float U, float V, float W)
{
	// Calculate new distributions functions
	for(int l = 0; l < 19; l++)
		CELLS(current, i, j, k, l) = CELLS(other, i, j, k, l);

	float A=0.0, B=0.0, rho=0.0, Nx=0.0, Ny=0.0;
	A = CELLS(other, i, k, k, 0)
	    								+ CELLS(other, i, k, k, 1)
	    								+ CELLS(other, i, k, k, 2)
	    								+ CELLS(other, i, k, k, 3)
	    								+CELLS(other, i, k, k, 6)
	    								+ CELLS(other, i, k, k, 10)
	    								+ CELLS(other, i, k, k, 11)
	    								+ CELLS(other, i, k, k, 7)
	    								+ CELLS(other, i, k, k, 18);

	B = CELLS(other, i, k, k, 4)
	    								+ CELLS(other, i, k, k, 8)
	    								+ CELLS(other, i, k, k, 12)
	    								+ CELLS(other, i, k, k, 14)
	    								+ CELLS(other, i, k, k, 16);

	rho = (A + 2.*B)/(W + 1.);



	Nx = (1./2.)*(CELLS(other, i, k, k, 0)+CELLS(other, i, k, k, 6)+CELLS(other, i, k, k, 7)-(CELLS(other, i, k, k, 1)+CELLS(other, i, k, k, 10)+CELLS(other, i, k, k, 11)))-(1./3.)*rho*U;
	Ny = (1./2.)*(CELLS(other, i, k, k, 2)+CELLS(other, i, k, k, 6)+CELLS(other, i, k, k, 10)-(CELLS(other, i, k, k, 3)+CELLS(other, i, k, k, 7)+CELLS(other, i, k, k, 11)))-(1./3.)*rho*V;

	CELLS(current, i, j, k, 5) = CELLS(other, i, k, k, 4)-(1./3.)*rho*W;
	CELLS(current, i, j, k, 9) = CELLS(other, i, k, k, 12)+(rho/6.)*(-W+U)-Nx;
	CELLS(current, i, j, k, 13) = CELLS(other, i, k, k, 8)+(rho/6.)*(-W-U)+Nx;
	CELLS(current, i, j, k, 15) = CELLS(other, i, k, k, 16)+(rho/6.)*(-W+V)-Ny;
	CELLS(current, i, j, k, 17) = CELLS(other, i, k, k, 14)+(rho/6.)*(-W-V)+Ny;
}

void velNodoInferior(float *cells, int current, int other, int X, int Y, int Z, int i, int j, int k, float U, float V, float W)
{
	// Calculate new distributions functions
	for (int l = 0; l < 19; l++)
		CELLS(other, i, j, k, l) = CELLS(current, i, j, k, l);


	float A=0.0, B=0.0, rho=0.0, Nx=0.0, Ny=0.0;
	A = CELLS(current, i, j, k, 0)
										+ CELLS(current, i, j, k, 1)
										+ CELLS(current, i, j, k, 2)
										+ CELLS(current, i, j, k, 3)
										+ CELLS(current, i, j, k, 6)
										+ CELLS(current, i, j, k, 7)
										+ CELLS(current, i, j, k, 10)
										+ CELLS(current, i, j, k, 11)
										+ CELLS(current, i, j, k, 18);

	B = CELLS(current, i, j, k, 5)
		    						+CELLS(current, i, j, k, 9)
		    						+CELLS(current, i, j, k, 13)
		    						+CELLS(current, i, j, k, 15)
		    						+CELLS(current, i, j, k, 17);
	rho = (A + 2.*B)/(1. - W);

	Nx=(1./2.)*(CELLS(current, i, j, k, 0)+CELLS(current, i, j, k, 6)+CELLS(current, i, j, k, 7)-(CELLS(current, i, j, k, 1)+CELLS(current, i, j, k, 10)+CELLS(current, i, j, k, 11)))-(1./3.)*rho*-U;
	Ny=(1./2.)*(CELLS(current, i, j, k, 2)+CELLS(current, i, j, k, 6)+CELLS(current, i, j, k, 10)-(CELLS(current, i, j, k, 3)+CELLS(current, i, j, k, 7)+CELLS(current, i, j, k, 11)))-(1./3.)*rho*V;

	CELLS(other, i, j, k, 4) =CELLS(current, i, j, k, 5)+(1./3.)*rho*W;
	CELLS(other, i, j, k, 8) =CELLS(current, i, j, k, 13)+(rho/6.)*(W-U)-Nx;
	CELLS(other, i, j, k, 12) =CELLS(current, i, j, k, 9)+(rho/6.)*(W+U)+Nx;
	CELLS(other, i, j, k, 14) =CELLS(current, i, j, k, 17)+(rho/6.)*(W+V)-Ny;
	CELLS(other, i, j, k, 16) =CELLS(current, i, j, k, 15)+(rho/6.)*(W-V)+Ny;
}

void velNodoIzquierdo(float g[19], float f[19], float U, float V, float W)
{
	// Calculate new distributions functions
	g=f;
	float A=0.0, B=0.0, rho=0.0, xNy=0.0, xNz=0.0;
	A=f[2]+f[3]+f[4]+f[5]+f[14]+f[15]+f[16]+f[17]+f[18];
	B=f[1]+f[10]+f[11]+f[12]+f[13];
	rho = (A+2.*B)/(U+1.);

	xNy=(1./2.)*(f[2]+f[14]+f[15]-(f[3]+f[16]+f[17]))-(1./3.)*rho*V;
	xNz=(1./2.)*(f[4]+f[16]+f[14]-(f[5]+f[15]+f[18]))-(1./3.)*rho*W;

	g[0]=f[1]-(1./3.)*rho*U;
	g[7]=f[10]+(rho/6.)*(U-V)-xNy;
	g[6]=f[11]+(rho/6.)*(U+V)-xNy;
	g[8]=f[13]+(rho/6.)*(U+W)-xNz;
	g[9]=f[12]+(rho/6.)*(U-W)+xNz;
}

void velNodoDerecho(float g[19], float f[19], float U, float V, float W)
{
	// Calculate new distributions functions
	g=f;
	float A=0.0, B=0.0, rho=0.0, xNy=0.0, xNz=0.0;
	A=f[2]+f[3]+f[4]+f[5]+f[14]+f[15]+f[16]+f[17]+f[18];
	B=f[1]+f[10]+f[11]+f[12]+f[13];
	rho = (A+2*B)/(U+1);

	xNy=(1./2.)*(f[2]+f[14]+f[15]-(f[3]+f[16]+f[17]))-(1./3.)*rho*V;
	xNz=(1./2.)*(f[4]+f[16]+f[14]-(f[5]+f[15]+f[18]))-(1./3.)*rho*W;

	g[0]=f[1]-(1./3.)*rho*U;
	g[7]=f[10]+(rho/6.)*(U-V)-xNy;
	g[6]=f[11]+(rho/6.)*(U+V)-xNy;
	g[8]=f[13]+(rho/6.)*(U+W)-xNz;
	g[9]=f[12]+(rho/6.)*(U-W)+xNz;
}

