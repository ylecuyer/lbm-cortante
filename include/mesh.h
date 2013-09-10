#ifndef MESH_H
	#define MESH_H
	using namespace std;

	class mesh{

		private:
		    int id;
		    int nNodos;
		    int nCeldas;
			float cX, cY, cZ;
			float **vertex;
			float **velocidad;
			float **velocidad2;
			float **fuerza;
			float *area;
			int **faces;
			float **normalesPorNodo;
			float **normalesPorCara;
			float **carasPorNodo;
			float **angulosPorNodo;
			float **vecinosPorNodo;
			float *laplaceKg;
			float *laplaceKh;
			int *nodosProblema;
			float areaS;
			float volumenE;

		public:
			void setID(int ID){id=ID;}
			int getID(){return id;}
			float** darNodos(){return vertex;}
			int** darCeldas(){return faces;}
			int darNumeroNodos(){return nNodos;}
			int darNumeroCeldas(){return nCeldas;}
			int posicionNodo(float x, float y, float z);
			int guardarVTU(int t);
			int agregarNodo(float x, float y, float z);
			int agregarCelda(int a, int b, int c);
			int existeNodo(float x, float y, float z);
			void mesh_refine_tri4();
			void proyectarEsfera(float r);
			void proyectarElipsoide(float a, float b, float c);
			void proyectarRBC(float r);
			void moverCentro(float x, float y, float z);
			void rotarEstructura(float alpha, float phi, float theta);
			float darKgPromedioPorNodo(int nodo);
			float darKhPromedioPorNodo(int nodo);
			float darLaplaceKgPromedioPorNodo(int nodo);
			float darLaplaceKhPromedioPorNodo(int nodo);
			float darKgPorNodo(int nodo);
			float darKhPorNodo(int nodo);
			float darK1PorNodo(int nodo);
			float darK2PorNodo(int nodo);
			void darMeanCurvatureOperator(int nodo, float dmco[3]);
			void iniciarGeometria();
			void actualizarGeometria();
			bool contieneNodo(int nodo, int cara);
			void darNormalPromedio(int nodo, float normal[3]);
			void calcularCurvaturaGaussiana();
			float calcularAngulosPorNodo(int nodo, float angulos[7]);
			void darCarasPorNodo(int nodo, int caras[7]);
			void darNodosPorElemento(int cara, int nodos[3]);
			float darAreaAlrededorPorNodo(int nodo);
			float darAreaVoronoiPorNodo(int nodo);
			float darAreaVoronoiParcial(int nodoA, int nodoB);
			float darAreaBaricentricaPorNodo(int nodo);
			void darNodosVecinos(int nodo, int vecinos[7]);
			void darCarasSegunDosNodos(int nodoA, int nodoB, int caras[2]);
			float darLaplaceKg(int nodo);
			float darLaplaceKh(int nodo);
			void darNormalCara(int i, float normal[3]);
			void darPosNodo(int n, float pos[3]);
			void darFuerzaNodo(int n, float f[3]);
			void setVelocidad(int n, float ux, float uy, float uz);
			void moverNodos(float dt, float dx);
			void calcularFuerzasFEM(mesh referencia, float ks);
			void calcularFuerzasHelfrich(float kb);
			void actualizarNodos(float **);
			void calcularCambioArea(mesh ref);
			float calcularAreaSuperficial();
			float darAreaElemento(int i);
			float darVolumenElemento(int i);
			float calcularVolumen();
			void calcularFuerzasVolumen(float v0, float ka);
			void calcularFuerzaNeta(float fNeta[3]);
			float calcularEnergia();
			void calcularMomentoNeto(float fMomento[3]);
			void encontrarNodosProblema();
			bool esNodoProblema(int nodo);
			mesh();
			~mesh();
};
#endif
