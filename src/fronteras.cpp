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

