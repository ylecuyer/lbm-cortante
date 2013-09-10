#ifndef _DEBUG_H_
#define _DEBUG_H_

#ifdef DEBUG
    #define _DEBUG(fmt, args...) printf("%s:%s:%d: " fmt "\n", __FILE__, __func__, __LINE__, ##args)
#else
    #define _DEBUG(fmt, args...)
#endif

#endif
