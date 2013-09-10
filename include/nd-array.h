#ifndef _ND_ARRAY_H_
#define _ND_ARRAY_H_

#define ACCESS1(tab, XSIZE, x) (*(tab + x))
#define ACCESS2(tab, XSIZE, YSIZE, x, y) (*(tab + x + XSIZE*y))
#define ACCESS3(tab, XSIZE, YSIZE, ZSIZE, x, y, z) (*(tab + x + XSIZE*(y + YSIZE*z)))
#define ACCESS4(tab, XSIZE, YSIZE, ZSIZE, JSIZE, x, y, z, j) (*(tab + x + XSIZE*(y + YSIZE*(z + ZSIZE*j))))
#define ACCESS5(tab, XSIZE, YSIZE, ZSIZE, JSIZE, KSIZE, x, y, z, j, k) (*(tab + x + XSIZE*(y + YSIZE*(z + ZSIZE*(j + JSIZE*k)))))

#endif
