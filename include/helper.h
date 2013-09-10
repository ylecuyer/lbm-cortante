#ifndef _HELPER_H_
#define _HELPER_H_

#include "nd-array.h"

#define CELLS(s, x, y, z, a) ACCESS5(cells, 2, X, Y, Z, 19, s, x, y, z, a)
#define FLAGS(x, y, z) ACCESS3(flags, X, Y, Z, x, y, z)
#define VEL(x, y, z, k) ACCESS4(vel, X, Y, Z, 3, x, y, z, k)
#define RHO(x, y, z) ACCESS3(rho, X, Y, Z, x, y, z)
#define FUERZA(x, y, z, k) ACCESS4(fuerza, X, Y, Z, 3, x, y, z, k)

#define CELLS_D(s, x, y, z, a) ACCESS5(cells_d, 2, X, Y, Z, 19, s, x, y, z, a)
#define FLAGS_D(x, y, z) ACCESS3(flags_d, X, Y, Z, x, y, z)
#define VEL_D(x, y, z, k) ACCESS4(vel_d, X, Y, Z, 3, x, y, z, k)
#define RHO_D(x, y, z) ACCESS3(rho_d, X, Y, Z, x, y, z)
#define FUERZA_D(x, y, z, k) ACCESS4(fuerza_d, X, Y, Z, 3, x, y, z, k)

#endif
