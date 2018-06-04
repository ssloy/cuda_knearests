#ifndef __VORONOI_H__
#define __VORONOI_H__

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#include "params.h"
#include "knearests.h"

#define cuda_check(x) if (x!=cudaSuccess) exit(1);

#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); ++I)

typedef unsigned char uchar;  // local indices with special values

__shared__ uchar3 tr_data[VORO_BLOCK_SIZE * _MAX_T_]; // memory pool for chained lists of triangles
__shared__ uchar boundary_next_data[VORO_BLOCK_SIZE * _MAX_P_];
__shared__ float4 clip_data[VORO_BLOCK_SIZE * _MAX_P_]; // clipping planes

inline  __device__ uchar3& tr(int t) { return  tr_data[threadIdx.x*_MAX_T_ + t]; }
inline  __device__ uchar& boundary_next(int v) { return  boundary_next_data[threadIdx.x*_MAX_P_ + v]; }
inline  __device__ float4& clip(int v) { return  clip_data[threadIdx.x*_MAX_P_ + v]; }

static const uchar END_OF_LIST = 255;
enum Status { triangle_overflow = 0, vertex_overflow = 1, inconsistent_boundary = 2, security_radius_not_reached = 3, success = 4 } ;

struct ConvexCell {
    __device__ ConvexCell(int p_seed, float* p_pts, Status* p_status);
    __device__ void clip_by_plane(int vid);
    __device__ float4 compute_triangle_point(uchar3 t, bool persp_divide=true) const;
    __device__ inline  uchar& ith_plane(uchar t, int i);
    __device__ int new_point(int vid);
    __device__ void new_triangle(uchar i, uchar j, uchar k);
    __device__ void compute_boundary();
    __device__ bool is_security_radius_reached(float4 last_neig);

    Status* status;
    uchar nb_t;
    uchar nb_r;
    float* pts;
    int voro_id;
    float4 voro_seed;
    uchar nb_v;
    uchar first_boundary_;     
};

void compute_voro_diagram_GPU(std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary,int nb_Lloyd_iter);

#endif // __VORONOI_H__

