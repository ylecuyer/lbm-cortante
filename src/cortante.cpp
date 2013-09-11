#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fluid.h"
#include "memory.h"

using namespace std;

int main(int argc, char *argv[])
{
	float X = 51;
	float Y = 21;
	float Z = 21;

	fluid fluido(X, Y, Z);

	float dt = 1.0;
	float dx = 1.0;

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

	float *cells_f = fluido.get_cells();
	float *flags_f = fluido.get_flags();
	float *rho_f = fluido.get_rho();
	float *fuerza_f = fluido.get_fuerza();
	float *vel_f = fluido.get_vel();

	float *cells_d = NULL;
	float *flags_d = NULL;
	float *rho_d = NULL;
	float *fuerza_d = NULL;
	float *vel_d = NULL;

	alloc_memory_GPU(X, Y, Z, &cells_d, &flags_d, &vel_d, &rho_d, &fuerza_d);

	send_data_to_GPU(X, Y, Z, cells_f, cells_d, flags_f, flags_d, vel_f, vel_d, rho_f, rho_d, fuerza_f, fuerza_d);


	// Fluido
	//From fluid constructor
	fluido.calcularMacro(cells_d, rho_d, vel_d, fuerza_d);
	fluido.setVelocidad(gamma_dot);


	for(int ts = 0 ; ts < STEPS ; ts++)
	{
		fluido.collide(cells_d, fuerza_d);
		fluido.stream(cells_d, flags_d);
		fluido.calcularMacro(cells_d, rho_d, vel_d, fuerza_d);

		if(ts%VTK==0)
		{
			printf("%d\n",ts);

			retrieve_data_from_GPU(X, Y, Z, cells_f, cells_d, flags_f, flags_d, vel_f, vel_d, rho_f, rho_d, fuerza_f, fuerza_d);

			fluido.guardar(ts);
		}
	}//Ciclo principal


	free_memory_GPU(cells_d, flags_d, vel_d, rho_d, fuerza_d);

	return 0;
}
