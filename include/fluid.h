#ifndef FLUID_H
#define FLUID_H
using namespace std;

class fluid{

private:

	int ts;
	int X, Y, Z;
	float *cells;
	float *flags;
	float *vel;
	float *rho;
	float *fuerza;
	float U;

public:

	float* get_cells(void);
	float* get_flags(void);
	float* get_vel(void);
	float* get_rho(void);
	float* get_fuerza(void);

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
	void stream(float *cells_d, float *flags_d);

	/*
	 * Implementa el paso de colisión utilizando la aproximación BGK Ec 2. y la función de equilibrio
	 * utilizando la Ec 5.
	 * 1. Calcular la función de equilibrio en cada celda utilizando Ec 5.
	 * 2. Calcular el valor del operador de colisión utilizando Ec 3.
	 */
	void collide(float *cells_d, float *fuerza_d);

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
	void calcularMacro(float *cells_d, float *rho_d, float *vel_d, float *fuerza_d);

	/*
	 * Esta función se utiliza únicamente para crear el flujo cortante, en donde u corresponde a
	 * la velocidad de las placas que se mueven en direcciones opuestas.
	 */
	void setVelocidad(float u);

	/*
	 * Constructor de la clase - no utilizado -
	 */
	fluid(int X, int Y, int Z);

	~fluid();
};
#endif
