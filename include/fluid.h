#ifndef FLUID_H
#define FLUID_H
using namespace std;

class fluid{

private:

	int ts;
	int X, Y, Z;
	float *****cells;
	float ***flags;
	float ****vel;
	float ***rho;
	float ****fuerza;
	float U;

public:
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fluid.h"

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

	/*
	 * Función encargada de construir las estructuras de datos en memoria
	 * 1. Define los nodos que son frontera
	 * 2. Inicializa cada celda con el valor inicial de densidad
	 */
	void inicializar(int x, int y, int z);

	/*
	 * Implementa el paso de streaming referido en la Ecuación 3, del documento guía.
	 * 1. Se encarga de propagar las funciones de distribución actuales hacia la siguiente celda
	 */
	void stream();

	/*
	 * Implementa el paso de colisión utilizando la aproximación BGK Ec 2. y la función de equilibrio
	 * utilizando la Ec 5.
	 * 1. Calcular la función de equilibrio en cada celda utilizando Ec 5.
	 * 2. Calcular el valor del operador de colisión utilizando Ec 3.
	 */
	void collide();

	/*
	 * Esta función se encarga de guardar todas las variables macroscópicas del fluido en un archivo
	 * con formato *.vtu, el nombre de cada archivo es fluido-#.vts
	 * 1. Guarda las coordenadas de cada nodo en la malla del método Lattice-Boltzmann.
	 * 2. Guarda el valor de la densidad de cada nodo.
	 * 3. Guarda cada componente de la velocidad ux, uy, uz.
	 * 4. Guarda el valor de la presión en cada nodo.
	 * 6. Debe estar disponible el directorio temp en el cual se guardan los archivos.
	 */
	int guardar(int s);

	/*
	 * Entrega la velocidad del nodo ubicado en la posición x,y, y z. El parametro f indica que componente
	 * de velocidad se ha de retornar f=0 retorna ux, f=1 retorna uy, f=2 retorna uz.
	 * @param int x, Coordenada en x
	 * @param int y, Coordenada en y
	 * @param int z, Coordenada en z
	 * @param int f, indicador de componente de velocidad
	 * @return float, el valor de la componente de velocidad
	 */
	float darVelocidad(int x, int y, int z, int f);


	/*
	 * Retorna el valor de la densidad del nodo ubicado en la posición x y y z.
	 * @param int x, Coordenada en x
	 * @param int y, Coordenada en y
	 * @param int z, Coordenada en z
	 * @return float, el valor de la densidad en el nodo
	 */
	float darDensidad(int x, int y, int z);

	/*
	 * Establece un vector fuerza para el nodo ubicado en la posición xyz. Esta función se utiliza para agregar
	 * los terminos de fuerza egnerados por la membrana corresponde al termino Fi de la ecuación 2.
	 * @param int x, Coordenada en x
	 * @param int y, Coordenada en y
	 * @param int z, Coordenada en z
	 * @param float[3], el vector fuerza
	 */
	void setFuerza(int x, int y, int z, float f[3]);


	/*
	 * Adiciona un vector fuerza para el nodo ubicado en la posición xyz. Esta función se utiliza para agregar
	 * los terminos de fuerza egnerados por la membrana corresponde al termino Fi de la ecuación 2.
	 * @param int x, Coordenada en x
	 * @param int y, Coordenada en y
	 * @param int z, Coordenada en z
	 * @param float[3], el vector fuerza
	 */
	void addFuerza(int x, int y, int z, float f[3]);


	/*
	 * Calcula el valor de todas las variables macroscópicas en el fluido densidad, velocidad, presión
	 * el valor es almacenado en cada atributo de la clase fluid
	 */
	void calcularMacro();

	/*
	 * Esta función se utiliza únicamente para crear el flujo cortante, en donde u corresponde a
	 * la velocidad de las placas que se mueven en direcciones opuestas.
	 */
	void setVelocidad(float u);

	/*
	 * Constructor de la clase - no utilizado -
	 */
	fluid() {
	}

	~fluid() {
	}
};
#endif
