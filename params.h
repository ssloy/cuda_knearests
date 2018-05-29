// default settings
#define POINTS_PER_BLOCK 64

#define DEFAULT_NB_PLANES 35
#define MAX_CLIPS  41
#define MAX_T  64

#define IF_EXPORT_HISTO(x) 
#define IF_OUTPUT_TET(x) 
#define IF_OUTPUT_P2BARY(x) 
#define IF_EXPORT_DECOMPOSITION(x) 
#define IF_VERBOSE(x) 
#define IF_NORMALIZE_PTS(x) 



// choose settings: only one to be 1
#define STAT_MODE 1
#define LOG_GPU_SPEED 0
#define LOYD_MODE 0
#define SIMU_MODE 0




#if LOG_GPU_SPEED
#define IF_VERBOSE(x) x
#endif 

// predefined settings
#if STAT_MODE
#define DEFAULT_NB_PLANES 90
#define MAX_CLIPS  50
#define MAX_T  50
#define IF_EXPORT_HISTO(x) x 
#define IF_OUTPUT_TET(x) x
#define IF_OUTPUT_P2BARY(x) x
//#define IF_EXPORT_DECOMPOSITION(x) x
#define IF_VERBOSE(x) x
#define IF_NORMALIZE_PTS(x) x
#endif

    
#if SIMU_MODE
#define REPLACE_BARY_BY_PRESSURE 0

#define IF_OUTPUT_P2BARY(x) x
#define IF_EXPORT_DECOMPOSITION(x) 
#define IF_VERBOSE(x) 
// required to apply pressure on neigs
#define IF_OUTPUT_TET(x) x
#endif

#if LOYD_MODE

#define DEFAULT_NB_PLANES 90
#define MAX_CLIPS  50   
#define MAX_T  50

#define REPLACE_BARY_BY_PRESSURE 0
#define IF_OUTPUT_P2BARY(x) x
#define IF_EXPORT_DECOMPOSITION(x) 
#define IF_VERBOSE(x) 
#endif






#if REPLACE_BARY_BY_PRESSURE 
#define IF_OUTPUT_TET(x) x
#endif 

IF_OUTPUT_TET(#define OUTPUT_TET)
