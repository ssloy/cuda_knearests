// Sylvain Lefebvre 2017-10-04
#include <stdio.h>

#include <cuda_runtime.h>
#include <assert.h>

#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <random>
#include <float.h>

// ------------------------------------------------------------

#define KN_global        36
#define POINTS_PER_BLOCK 64

// ------------------------------------------------------------

// it is supposed that all points fit in range [0,1000]^3
__device__ int cellFromPoint(int xdim, int ydim, int zdim, float x, float y, float z) {
    int   i = (int)floor(x * (float)xdim / 1000.f);
    int   j = (int)floor(y * (float)ydim / 1000.f);
    int   k = (int)floor(z * (float)zdim / 1000.f);
    i = max(0, min(i, xdim - 1));
    j = max(0, min(j, ydim - 1));
    k = max(0, min(k, zdim - 1));
    return i + j*xdim + k*xdim*ydim;
}

__global__ void count(const float *points, int numPoints, int xdim, int ydim, int zdim, int *counters) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < numPoints) {
        float x = points[id*3 + 0];
        float y = points[id*3 + 1];
        float z = points[id*3 + 2];
        int cell = cellFromPoint(xdim, ydim, zdim, x, y, z);
        atomicAdd(counters + cell, 1);
    }
}

__global__ void reserve(int xdim, int ydim, int zdim, const int *counters, int *globalcounter, int *ptrs) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < xdim*ydim*zdim) {
        int cnt = counters[id];
        if (cnt > 0) {
            ptrs[id] = atomicAdd(globalcounter, cnt);
        }
    }
}

// it supposes that counters buffer is set to zero
__global__ void store(const float *points, int numPoints, int xdim, int ydim, int zdim, const int *ptrs, int *counters, int num_stored, float *stored_points, unsigned int *permutation) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < numPoints) {
        float x = points[id*3+0];
        float y = points[id*3+1];
        float z = points[id*3+2];
        int cell = cellFromPoint(xdim, ydim, zdim, x, y, z);
        int pos = ptrs[cell] + atomicAdd(counters + cell, 1);
        permutation[pos] = id;
        stored_points[pos*3+0] = x;
        stored_points[pos*3+1] = y;
        stored_points[pos*3+2] = z;
    }
}

template <typename T> __device__ void inline swap_test_device(T& a, T& b) {
    T c(a); a=b; b=c;
}

__global__ void knearest(int xdim, int ydim, int zdim, int num_stored, const int *ptrs, const int *counters, const float *stored_points, int num_cell_offsets, const int *cell_offsets, const float *cell_offset_distances, unsigned int *g_knearests) {
    // each thread updates its k-nearests
    __shared__ unsigned int knearests      [KN_global*POINTS_PER_BLOCK];
    __shared__ float        knearests_dists[KN_global*POINTS_PER_BLOCK];

    int point_in = threadIdx.x + blockIdx.x*POINTS_PER_BLOCK;
    if (point_in >= num_stored) return;

    // point considered by this thread
    float x = stored_points[point_in*3 + 0];
    float y = stored_points[point_in*3 + 1];
    float z = stored_points[point_in*3 + 2];

    int cell_in = cellFromPoint(xdim, ydim, zdim, x, y, z);
    int offs = threadIdx.x*KN_global;

    for (int i = 0; i < KN_global; i++) {
        knearests      [offs + i] = UINT_MAX;
        knearests_dists[offs + i] = FLT_MAX;
    }

    for (int o=0; o<num_cell_offsets; o++) {
        float min_dist = cell_offset_distances[o];
        if (knearests_dists[offs] < min_dist) break;

        int cell = cell_in + cell_offsets[o];
        if (cell>=0 && cell<xdim*ydim*zdim) {
            int cell_base = ptrs[cell];
            int num = counters[cell];
            for (int ptr=cell_base; ptr<cell_base+num; ptr++) {
                if (ptr==point_in) continue; // exclude the point itself from its neighbors
                float x_cmp = stored_points[ptr*3 + 0];
                float y_cmp = stored_points[ptr*3 + 1];
                float z_cmp = stored_points[ptr*3 + 2];

                float d = (x_cmp - x)*(x_cmp - x) + (y_cmp - y)*(y_cmp - y) + (z_cmp - z)*(z_cmp - z);

                if (d < knearests_dists[offs]) {
                    // replace current max
                    knearests[offs] = ptr;
                    knearests_dists[offs] = d;

                    int j = 0; // max-heapify
                    while (true) { 
                        int left  = 2*j+1;
                        int right = 2*j+2;
                        int largest = j;
                        if (left<KN_global && knearests_dists[offs+left]>knearests_dists[offs+largest]) {
                            largest = left;
                        }
                        if (right<KN_global && knearests_dists[offs+right]>knearests_dists[offs+largest]) {
                            largest = right;
                        }
                        if (largest==j) break;
                        swap_test_device(knearests_dists[offs+j], knearests_dists[offs+largest]);
                        swap_test_device(knearests      [offs+j], knearests      [offs+largest]);
                        j = largest;
                    }
                }
            } // pts inside the cell
        } // valid cell id
    } // cell offsets

    // store result
    for (int i = 0; i < KN_global; i++) {
        g_knearests[point_in + i*num_stored] = knearests[offs + i];
    }
}

// ------------------------------------------------------------

typedef struct {
    int K;
    int dimx, dimy, dimz;
    int num_cell_offsets;
    int allocated_points;
    int *d_cell_offsets;         // cell offsets (sorted by rings), Nmax*Nmax*Nmax*Nmax (Nmax = 8)
    float *d_cell_offset_dists;
    unsigned int *d_permutation;
    int *d_counters;             // counters per cell,   dimx*dimy*dimz
    int *d_ptrs;                 // cell start pointers, dimx*dimy*dimz
    int *d_globcounter;          // global allocation counter, 1
    float *d_stored_points;      // input points sorted, numpoints + 1
    unsigned int *d_knearests;   // knn, allocated_points * KN_global
} kn_problem;

// ------------------------------------------------------------

void kn_firstbuild(kn_problem *kn,float *d_points, int numpoints) {
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    { // count points per grid cell
        int threadsPerBlock = 256;
        int blocksPerGrid = (numpoints + threadsPerBlock - 1) / threadsPerBlock;
        count << <blocksPerGrid, threadsPerBlock >> >(d_points, numpoints, kn->dimx, kn->dimy, kn->dimz, kn->d_counters);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    { // reserve memory for stored points
        int threadsPerBlock = 256;
        int blocksPerGrid = (kn->dimx*kn->dimy*kn->dimz + threadsPerBlock - 1) / threadsPerBlock;
        reserve << <blocksPerGrid, threadsPerBlock >> >(kn->dimx, kn->dimy, kn->dimz, kn->d_counters, kn->d_globcounter, kn->d_ptrs);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    { // store
        // zero counters
        cudaMemset(kn->d_counters, 0x00, kn->dimx*kn->dimy*kn->dimz*sizeof(int));
        // call kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numpoints + threadsPerBlock - 1) / threadsPerBlock;
        store << <blocksPerGrid, threadsPerBlock >> >(d_points, numpoints, kn->dimx, kn->dimy, kn->dimz, kn->d_ptrs, kn->d_counters, kn->allocated_points, kn->d_stored_points, kn->d_permutation);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cerr << "kn_firstbuild: " << milliseconds << " msec" << std::endl; 
}

// ------------------------------------------------------------

void gpuMalloc(void **ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void gpuMallocNCopy(void **dst, const void *src, size_t size) {
    gpuMalloc(dst, size);
    cudaError_t err = cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy from host to device (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void gpuMallocNMemset(void **ptr, int value, size_t size) {
    gpuMalloc(ptr, size);
    cudaError_t err = cudaMemset(*ptr, value, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to write to device memory (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ------------------------------------------------------------

kn_problem *kn_prepare(float *points, int numpoints) {
    kn_problem *kn = (kn_problem*)malloc(sizeof(kn_problem));
    kn->K = KN_global;
    kn->allocated_points = numpoints;

    kn->d_permutation       = NULL;
    kn->d_cell_offsets      = NULL;
    kn->d_cell_offset_dists = NULL;
    kn->d_counters          = NULL;
    kn->d_ptrs              = NULL;
    kn->d_globcounter       = NULL;
    kn->d_stored_points     = NULL;
    kn->d_knearests         = NULL;

    int sz = max(1,(int)round(pow(numpoints / 11.3f, 1.0f / 3.0)));
    kn->dimx = sz;
    kn->dimy = sz;
    kn->dimz = sz;

    int Nmax = 8;
    if (sz < Nmax) {
        std::cerr << "Current implementation does not support low number of input points" << std::endl;
        exit(EXIT_FAILURE);
    }
    // create cell offsets, very naive approach, should be fine, pre-computed once
    int alloc = Nmax*Nmax*Nmax*Nmax;
    int   *cell_offsets      =   (int*)malloc(alloc*sizeof(int));
    float *cell_offset_dists = (float*)malloc(alloc*sizeof(float));
    cell_offsets[0] = 0;
    cell_offset_dists[0] = 0.0f;
    kn->num_cell_offsets = 1;
    for (int ring = 1; ring < Nmax; ring++) {
        for (int k = -Nmax / 2; k <= Nmax / 2; k++) {
            for (int j = -Nmax / 2; j <= Nmax / 2; j++) {
                for (int i = -Nmax / 2; i <= Nmax / 2; i++) {
                    if (max(abs(i), max(abs(j), abs(k))) != ring) continue;

                    int id_offset = i + j*kn->dimx + k*kn->dimx*kn->dimy;
                    if (id_offset == 0) { 
                        std::cerr << "Error generating offsets" << std::endl;
                        exit(EXIT_FAILURE); 
                    }
                    cell_offsets[kn->num_cell_offsets] = id_offset;
                    float d = (float)(ring - 1) / (float)max(kn->dimx, max(kn->dimy, kn->dimz));
                    cell_offset_dists[kn->num_cell_offsets] = d*d; // squared
                    kn->num_cell_offsets++;
                    if (kn->num_cell_offsets >= alloc) {
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }

    size_t memory_used = 0, bufsize = 0;
    
    bufsize = kn->num_cell_offsets*sizeof(int); // allocate cell offsets
    memory_used += bufsize;
    gpuMallocNCopy((void **)&kn->d_cell_offsets, cell_offsets, bufsize);
    free(cell_offsets);

    bufsize = kn->num_cell_offsets*sizeof(float); // allocate cell offsets distances
    memory_used += bufsize;
    gpuMallocNCopy((void **)&kn->d_cell_offset_dists, cell_offset_dists, bufsize);
    free(cell_offset_dists);

    float *d_points = NULL;
    bufsize = numpoints*3*sizeof(float); // allocate input points
    memory_used += bufsize;
    gpuMallocNCopy((void **)&d_points, points, bufsize); 

    bufsize = kn->dimx*kn->dimy*kn->dimz*sizeof(int);  // allocate cell counters
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_counters, 0x00, bufsize); 
    
    bufsize = kn->dimx*kn->dimy*kn->dimz*sizeof(int); // allocate cell start pointers
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_ptrs, 0x00, bufsize); 

    bufsize = sizeof(int);
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_globcounter, 0x00, bufsize); 

    bufsize = kn->allocated_points*sizeof(float)*3; // allocate stored points
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_stored_points, 0x00, bufsize); 

    bufsize += kn->allocated_points*KN_global*sizeof(int);
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_knearests, 0xFF, bufsize); 

    bufsize += kn->allocated_points*sizeof(int); // keep the track of reordering
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_permutation, 0xFF, bufsize); 

    // construct initial structure
    kn_firstbuild(kn,d_points,numpoints);

    // we no longer need the initial points
    cudaFree(d_points);
    std::cerr << "GPU memory used: " << memory_used/1048576 << " Mb" << std::endl;
    return kn;
}

// ------------------------------------------------------------

void kn_solve(kn_problem *kn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = POINTS_PER_BLOCK;
    int blocksPerGrid = (kn->allocated_points + threadsPerBlock - 1) / POINTS_PER_BLOCK;

    std::cerr << "threads per block: " << threadsPerBlock << ", blocks per grid: " << blocksPerGrid << std::endl;

    cudaEventRecord(start);

    knearest << <blocksPerGrid, threadsPerBlock >> >(
            kn->dimx, kn->dimy, kn->dimz, kn->allocated_points,
            kn->d_ptrs, kn->d_counters, kn->d_stored_points,
            kn->num_cell_offsets, kn->d_cell_offsets, kn->d_cell_offset_dists,
            kn->d_knearests);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cerr << "kn_solve: " << milliseconds << " msec" << std::endl;
}

// ------------------------------------------------------------

void kn_free(kn_problem **kn) {
    cudaFree((*kn)->d_cell_offsets);
    cudaFree((*kn)->d_cell_offset_dists);
    cudaFree((*kn)->d_counters);
    cudaFree((*kn)->d_ptrs);
    cudaFree((*kn)->d_globcounter);
    cudaFree((*kn)->d_stored_points);
    cudaFree((*kn)->d_knearests);
    free(*kn);
    *kn = NULL;
}

float *kn_get_points(kn_problem *kn) {
    float *stored_points = (float*)malloc(kn->allocated_points * sizeof(float) * 3);
    cudaError_t err = cudaMemcpy(stored_points, kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_print_stats] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return stored_points;
}

unsigned int *kn_get_permutation(kn_problem *kn) {
    unsigned int *permutation = (unsigned int*)malloc(kn->allocated_points*sizeof(int));
    cudaError_t err = cudaMemcpy(permutation, kn->d_permutation, kn->allocated_points * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_print_stats] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return permutation;
}

unsigned int *kn_get_knearests(kn_problem *kn) {
    unsigned int *knearests = (unsigned int*)malloc(kn->allocated_points * KN_global * sizeof(int));
    cudaError_t err = cudaMemcpy(knearests, kn->d_knearests, kn->allocated_points * KN_global * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_print_stats] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return knearests;
}

void kn_print_stats(kn_problem *kn) {
    cudaError_t err = cudaSuccess;

    int *counters = (int*)malloc(kn->dimx*kn->dimy*kn->dimz*sizeof(int));
    err = cudaMemcpy(counters, kn->d_counters, kn->dimx*kn->dimy*kn->dimz*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_print_stats] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // stats on counters
    int tot = 0;
    int cmin = INT_MAX, cmax = 0;
    std::map<int, int> histo;
    for (int c = 0; c < kn->dimx*kn->dimy*kn->dimz; c++) {
        histo[counters[c]]++;
        cmin = min(cmin, counters[c]);
        cmax = max(cmax, counters[c]);
        tot += counters[c];
    }
    std::cerr << "Grid:  points per cell: " << cmin << " (min), " << cmax << " (max), " << (kn->allocated_points)/(float)(kn->dimx*kn->dimy*kn->dimz) << " avg, total " << tot << std::endl;
    for (auto H : histo) {
        std::cerr << "[" << H.first << "] => " << H.second << std::endl;
    }
    free(counters);
}

