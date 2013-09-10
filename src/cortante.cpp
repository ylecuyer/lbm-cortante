#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "ibm.h"
#include "fluid.h"
#include "mesh.h"

using namespace std;

int main(int argc, char *argv[])
{
	fluid fluido;
	float dt = 1.0;
	float dx = 1.0;
	float X = 51;
	float Y = 21;
	float Z = 21;
	int VTK = 50;

	// Parametros adimensionales
	float rho = 1.0;
	float nu = 1./6.;
	float Re = 0.5; 
	float G = 0.5;
	float R = Z/5;
	float gamma_dot = (Re*nu)/(rho*pow(R,2));
	float ks = (gamma_dot*nu*R)/(G);
	float kb = ks*1.0e-6;
	float kp = (gamma_dot)/(G);
	float STEPS = 12.0/kp;
	printf("A completar %f iteraciones\n", STEPS);

	// Fluido
	fluido.inicializar(X,Y,Z);
	fluido.setVelocidad(gamma_dot);

	for(int ts = 0 ; ts < STEPS ; ts++)
		{
		fluido.collide();
		fluido.stream();
		fluido.calcularMacro();

		if(ts%VTK==0)
		{
			fluido.guardar(ts);
			printf("%d\n",ts);
		}
	}//Ciclo principal

	return 0;
}
