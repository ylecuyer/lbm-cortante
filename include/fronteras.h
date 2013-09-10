#ifndef FRONTERAS_H
	#define FRONTERAS_H
	using namespace std;

	void velNodoSuperior(float *cells, int current, int other, int X, int Y, int Z, int i, int j, int k, float U, float V, float W);
	void velNodoInferior(float *cells, int current, int other, int X, int Y, int Z, int i, int j, int k, float U, float V, float W);
	void velNodoIzquierdo(float g[19], float f[19], float U, float V, float W);
	void velNodoDerecho(float g[19], float f[19], float U, float V, float W);

#endif
