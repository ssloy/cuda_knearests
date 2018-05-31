// Sylvain Lefebvre 2017-10-04
#pragma once

//struct kn_problem;
typedef struct {
    int K;
    int dimx, dimy, dimz;
    int num_cell_offsets;
    int allocated_points;
    int *d_cell_offsets;         // cell offsets (sorted by rings), Nmax*Nmax*Nmax*Nmax (Nmax = 8)
    float *d_cell_max;
    float *d_cell_offset_dists;
    unsigned int *d_permutation;
    int *d_counters;             // counters per cell,   dimx*dimy*dimz
    int *d_ptrs;                 // cell start pointers, dimx*dimy*dimz
    int *d_globcounter;          // global allocation counter, 1
    float3 *d_stored_points;      // input points sorted, numpoints + 1
    unsigned int *d_knearests;   // knn, allocated_points * KN
} kn_problem;

// ------------------------------------------------------------


kn_problem   *kn_prepare(float3 *points, int numpoints);
void          kn_solve(kn_problem *kn);
void          kn_free(kn_problem **kn);

float3        *kn_get_points(kn_problem *kn);
unsigned int *kn_get_knearests(kn_problem *kn);
unsigned int *kn_get_permutation(kn_problem *kn);

void kn_print_stats(kn_problem *kn);

