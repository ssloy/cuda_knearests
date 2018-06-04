// conservative settings (for white noise)
#define PRESET 1 

#if 0==PRESET
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  32
#define _K_             190
#define _MAX_P_         64
#define _MAX_T_         96
#elif 1==PRESET
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  32
#define _K_             90
#define _MAX_P_         50
#define _MAX_T_         96
#elif 2==PRESET
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  64
#define _K_             35
#define _MAX_P_         32
#define _MAX_T_         96
#endif

#define IF_VERBOSE(x) x

