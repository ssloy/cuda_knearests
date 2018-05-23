#ifndef VBW_H
#define VBW_H

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include "knearests.h"
#include "kd_tree.h"

#define cuda_check(x) if (x!=cudaSuccess) exit(1);
#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); ++I)

typedef unsigned char uchar;  // local indices with special values

// voronoi vertices are stored as triangles
#ifdef __CUDA_ARCH__
__shared__ uchar3 tr_data[32 * MAX_T]; // memory pool for chained lists of triangles
__shared__ uchar boundary_next_data[32 * MAX_CLIPS];
__shared__ float4 clip_data[32 * MAX_CLIPS]; // clipping planes

inline  __device__ uchar3& tr(int t) { return  tr_data[threadIdx.x*MAX_T + t]; }
inline  __device__ uchar& boundary_next(int v) { return  boundary_next_data[threadIdx.x*MAX_CLIPS + v]; }
inline  __device__ float4& clip(int v) { return  clip_data[threadIdx.x*MAX_CLIPS + v]; }
#else
uchar3 tr_data[MAX_T]; // memory pool for chained lists of triangles
uchar boundary_next_data[MAX_CLIPS];
float4 clip_data[MAX_CLIPS]; // clipping planes

inline uchar3& tr(int t) { return  tr_data[t]; }
inline uchar& boundary_next(int v) { return  boundary_next_data[v]; }
inline float4& clip(int v) { return  clip_data[v]; }
#endif

namespace VBW {

    struct GlobalStats {
        GlobalStats() { reset(); }
        void reset() {
            nb_clips_before_radius.resize(1000, 0);
            nb_removed_voro_vertex_per_clip.resize(1000, 0);
            compute_boundary_iter.resize(1000, 0);
            nbv.resize(1000, 0);
            nbt.resize(1000, 0);
        }
        void start_cell() {
            cur_clip = 6;/* the bbox */
        }
        void add_clip(int nb_conflict_vertices) {
            cur_clip++;
            nb_removed_voro_vertex_per_clip[nb_conflict_vertices]++;
        }
        void add_compute_boundary_iter(int nb_iter) { compute_boundary_iter[nb_iter]++; }
        void end_cell() {
            nb_clips_before_radius[cur_clip]++;
        }
        int cur_clip;
        std::vector<int> nbv;
        std::vector<int> nbt;
        std::vector<int> compute_boundary_iter;
        std::vector<int> nb_clips_before_radius;
        std::vector<int> nb_removed_voro_vertex_per_clip;
        void show() {
            //show_prop(nb_clips_before_radius, false, false," #clips ");
            //show_prop(nb_removed_voro_vertex_per_clip, true, true, " #nb_removed_voro_vertex_per_clip ");
            //show_prop(compute_boundary_iter, true, true, " #compute_boundary_iter ");
            //show_prop(nbv, true, true, " #nbv ");
            //show_prop(nbv, true, true, " #nbt ");
        }
    } gs;


    static const uchar END_OF_LIST = 255;
    enum Status { triangle_overflow = 0, vertex_overflow = 1, weird_cavity = 2, security_ray_not_reached = 3, success = 4 } ;
    char StatusStr[5][128] = { "triangle_overflow","vertex_overflow ","weird_cavity "," security_ray_not_reached ","success" };

    class ConvexCell {
        public:
           __host__ __device__ ConvexCell(int p_seed, float* p_pts);
           __host__ __device__ void clip_by_plane(int vid);
           __host__ __device__ float4 compute_triangle_point(uchar3 t) const;
           __host__ __device__ inline  uchar& ith_plane(uchar t, int i) {
                //return reinterpret_cast<uchar *>(tr_data + threadIdx.x*MAX_T + t)[i];
                return reinterpret_cast<uchar *>(&(tr(t)))[i];
            }

            template <typename T>__host__ __device__ void inline swap_on_device(T& a, T& b) {
                T c(a); a=b; b=c;
            }

           __host__ __device__ int new_point(int vid);
           __host__ __device__ void new_triangle(uchar i, uchar j, uchar k);
           __host__ __device__ void compute_boundary();

            Status status;

            uchar nb_t;
            uchar nb_conflicts;
            float* pts;
            int voro_id;
            float3 voro_seed;
            uchar nb_v;
#if OUTPUT_TETS
            int vorother_id[MAX_CLIPS]; 
#endif        
            uchar first_boundary_;     
    };

   __host__ __device__ ConvexCell::ConvexCell(int p_seed, float* p_pts) {
        float eps  = .1;
        float xmin = -eps;
        float ymin = -eps;
        float zmin = -eps;
        float xmax = 1000 + eps;
        float ymax = 1000 + eps;
        float zmax = 1000 + eps;
        pts = p_pts;
        first_boundary_ = END_OF_LIST;
        FOR(i, MAX_CLIPS) boundary_next(i) = END_OF_LIST;
        voro_id = p_seed;
        voro_seed = make_float3(pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2]);
        status = success;

        clip(0) = make_float4( 1.0,  0.0,  0.0, -xmin);
        clip(1) = make_float4(-1.0,  0.0,  0.0,  xmax);
        clip(2) = make_float4( 0.0,  1.0,  0.0, -ymin);
        clip(3) = make_float4( 0.0, -1.0,  0.0,  ymax);
        clip(4) = make_float4( 0.0,  0.0,  1.0, -zmin);
        clip(5) = make_float4( 0.0,  0.0, -1.0,  zmax);
        nb_v = 6;

        tr(0) = make_uchar3(2, 5, 0);
        tr(1) = make_uchar3(5, 3, 0);
        tr(2) = make_uchar3(1, 5, 2);
        tr(3) = make_uchar3(5, 1, 3);
        tr(4) = make_uchar3(4, 2, 0);
        tr(5) = make_uchar3(4, 0, 3);
        tr(6) = make_uchar3(2, 4, 1);
        tr(7) = make_uchar3(4, 3, 1);
        nb_t = 8;
    }

   __host__ __device__ inline float det2x2(float a11, float a12, float a21, float a22) { 
        return a11*a22 - a12*a21; 
    }

   __host__ __device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
        return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
    }

   __host__ __device__ float4 ConvexCell::compute_triangle_point(uchar3 t) const {
        float4 pi1 = clip(t.x);
        float4 pi2 = clip(t.y);
        float4 pi3 = clip(t.z);
        float4 result;
        result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w =  det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        return result;
    }

   __host__ __device__ void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
        if (nb_t+1 >= MAX_T) { 
            status = triangle_overflow; 
            return; 
        }
        tr(nb_t) = make_uchar3(i, j, k);
        nb_t++;
    }

   __host__ __device__ int ConvexCell::new_point(int vid) {
        if (nb_v == MAX_CLIPS) { 
            status = vertex_overflow; 
            return -1; 
        }
#if OUTPUT_TETS
        vorother_id[nb_v] = vid;
#endif        
        float3 B = make_float3(pts[3 * vid], pts[3 * vid + 1], pts[3 * vid + 2]);
        float3 dir  = make_float3(voro_seed.x - B.x, voro_seed.y - B.y, voro_seed.z - B.z);
        float3 ave2 = make_float3(voro_seed.x + B.x, voro_seed.y + B.y, voro_seed.z + B.z);
        float dot = ave2.x*dir.x + ave2.y*dir.y + ave2.z*dir.z;
        clip(nb_v) = make_float4(dir.x, dir.y, dir.z, -dot / 2.);
        nb_v++;
        return nb_v - 1;
    }

   __host__ __device__ void ConvexCell::compute_boundary() {
        while (first_boundary_ != END_OF_LIST) { // clean circular list of the boundary
            uchar last = first_boundary_;
            first_boundary_ = boundary_next(first_boundary_);
            boundary_next(last) = END_OF_LIST;
        }

        int nb_iter = 0;
        uchar t = nb_t;
        while (nb_conflicts>0) {
            if (nb_iter++ >50) { 
                status = weird_cavity; 
                return; 
            }
            bool is_in_border[3];
            bool next_is_opp[3];
            FOR(e, 3)   is_in_border[e] = (boundary_next(ith_plane(t, e)) != END_OF_LIST);
            FOR(e, 3)   next_is_opp[e] = (boundary_next(ith_plane(t, (e + 1) % 3)) == ith_plane(t, e));

            bool can_do_it_now = true;
            // check for non manifoldness
            FOR(e, 3) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) can_do_it_now = false;

            // check for more than one boundary ... or first triangle
            if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
                if (first_boundary_ == END_OF_LIST) {
                    FOR(e, 3) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                    first_boundary_ = tr(t).x;
                }
                else can_do_it_now = false;
            }

            if (!can_do_it_now) {
                t++;
                if (t == nb_t + nb_conflicts) t = nb_t;
                continue;
            }

            // link next
            FOR(e, 3) if (!next_is_opp[e]) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);

            // destroy link from removed vertices
            FOR(e, 3)  if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
                if (first_boundary_ == ith_plane(t, (e + 1) % 3)) first_boundary_ = boundary_next(ith_plane(t, (e + 1) % 3));
                boundary_next(ith_plane(t, (e + 1) % 3)) = END_OF_LIST;
            }

            swap_on_device(tr(t), tr(nb_t+nb_conflicts-1));

            t = nb_t;
            nb_conflicts--;
        }
#ifndef __CUDA_ARCH__
        gs.add_compute_boundary_iter(nb_iter);
#endif
    }

   __host__ __device__ void  ConvexCell::clip_by_plane(int vid) {
        int cur_v= new_point(vid);
        if (status == vertex_overflow) return;
        float4 eqn = clip(cur_v);
        nb_conflicts = 0;


        // Step 1: find conflicts
        int i = 0;
        while (i < nb_t) {
            float4 pc = compute_triangle_point(tr(i));
            // check conflict
            if (eqn.x*pc.x + eqn.y*pc.y + eqn.z*pc.z + eqn.w*pc.w >0){// tr conflict
                swap_on_device(tr(i), tr(nb_t-1));
                nb_t--;
                nb_conflicts++;
            }
            else i++;
        }

#ifndef __CUDA_ARCH__
        gs.add_clip(nb_conflicts);
#endif

        if (nb_conflicts == 0) {
            nb_v--;
            return;
        }

        // Step 2: compute cavity boundary
        compute_boundary();
        if (status != success) return;
        if (first_boundary_ == END_OF_LIST) return;

        // Step 3: Triangulate cavity
        uchar cir = first_boundary_;
        do {
            new_triangle(cur_v, cir, boundary_next(cir));
            if (status != success) return;
            cir = boundary_next(cir);
        } while (cir != first_boundary_);
    }
}




//###################  KERNEL   ######################
__host__ __device__ void compute_voro_cell(float * pts, int nbpts, unsigned int* neigs, VBW::Status* gpu_stat, int *out_tets, int* nb_out_tet, float* out_pts,int seed) {
    FOR(d, 3) out_pts[3 * seed + d] = pts[3 * seed + d];

    //create BBox
    VBW::ConvexCell cc(seed, pts);

    // clip by halfspaces
    FOR(v, DEFAULT_NB_PLANES) {
        cc.clip_by_plane(neigs[DEFAULT_NB_PLANES * seed + v]);
        if (cc.status != VBW::success) {
            gpu_stat[seed] = cc.status;
            return;
        }
    }

    // check security ray
    float min_vertex_dist22seed = 1000000;
    FOR(i, cc.nb_t) {
        float4 pc = cc.compute_triangle_point(tr(i));
        float3 diff2seed = make_float3(pc.x / pc.w - cc.voro_seed.x, pc.y / pc.w - cc.voro_seed.y, pc.z / pc.w - cc.voro_seed.z);
        float d22seed = diff2seed.x*diff2seed.x + diff2seed.y*diff2seed.y + diff2seed.z*diff2seed.z;
        min_vertex_dist22seed = min(d22seed, min_vertex_dist22seed);
    }
    float max_neig_dist22seed = 0;
    FOR(v, DEFAULT_NB_PLANES) {
        unsigned int vid = neigs[DEFAULT_NB_PLANES * seed + v];
        float3 diff2seed = make_float3(pts[3 * vid] - cc.voro_seed.x, pts[3 * vid + 1] - cc.voro_seed.y, pts[3 * vid + 2] - cc.voro_seed.z);
        float d22seed = diff2seed.x*diff2seed.x + diff2seed.y*diff2seed.y + diff2seed.z*diff2seed.z;
        max_neig_dist22seed = max(d22seed, max_neig_dist22seed);
    }
    gpu_stat[seed] = cc.status;
    if (max_neig_dist22seed<4 * min_vertex_dist22seed)
        gpu_stat[seed] = VBW::security_ray_not_reached;

    //output tets
#if OUTPUT_TETS
    FOR(t, cc.nb_t) {
        if (tr(t).x > 5 && tr(t).y > 5 && tr(t).z > 5) {
            uint4 tet = make_uint4(cc.voro_id, 0, 0, 0);
            tet.y = cc.vorother_id[tr(t).x];
            tet.z = cc.vorother_id[tr(t).y];
            tet.w = cc.vorother_id[tr(t).z];
#ifdef __CUDA_ARCH__
            int top = atomicAdd(nb_out_tet, 1);
#else 
            (*nb_out_tet)++;
            int top = *nb_out_tet ;
#endif
            out_tets[top * 4] = cc.voro_id;
            FOR(f, 3) out_tets[top * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f)];
        }
    }
#endif    

}

__global__ void voro_cell_test_GPU_param(float * pts, int nbpts, unsigned int* neigs, VBW::Status* gpu_stat, int *out_tets, int* nb_out_tet, float* out_pts) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= nbpts) return;
    compute_voro_cell(pts, nbpts, neigs, gpu_stat, out_tets, nb_out_tet, out_pts, seed);
}

template <class T> struct GPUVar {
    GPUVar(T& val) {
        cuda_check(cudaMalloc((void**)& gpu_x, sizeof(T)));
        x = &val;
        cpu2gpu();
    }
    ~GPUVar() { cuda_check(cudaFree(gpu_x)); }
    void cpu2gpu() { cuda_check(cudaMemcpy(gpu_x, x, sizeof(T), cudaMemcpyHostToDevice)); }
    void gpu2cpu() { cuda_check(cudaMemcpy(x, gpu_x, sizeof(T), cudaMemcpyDeviceToHost)); }

    T* gpu_x;
    T* x;
};

template <class T> struct GPUBuffer {
    void init(T* data) {
        std::cerr << "GPU: " << size * sizeof(T)/1048576 << " Mb used" << std::endl;
        cpu_data = data;
        cuda_check(cudaMalloc((void**)& gpu_data, size * sizeof(T)));
        cpu2gpu();
    }
    GPUBuffer(std::vector<T>& v) {size = v.size() ;init(v.data());}
    ~GPUBuffer() { cuda_check(cudaFree(gpu_data)); }

    void cpu2gpu() { cuda_check(cudaMemcpy(gpu_data, cpu_data, size * sizeof(T), cudaMemcpyHostToDevice)); }
    void gpu2cpu() { cuda_check(cudaMemcpy(cpu_data, gpu_data, size * sizeof(T), cudaMemcpyDeviceToHost)); }

    T* cpu_data;
    T* gpu_data;
    int size;
};



//###################  CALL FUNCTION ######################
void compute_voro_diagram_GPU(std::vector<float>& pts, std::vector<int>& out_tets, int block_size , int &nb_tets, std::vector<VBW::Status> &stat, std::vector<float>& out_pts) {
    int nbpts = pts.size() / 3;
    nb_tets = 0;
    kn_problem *kn = NULL;
    {
        Stopwatch W("GPU KNN");
        kn = kn_prepare(pts.data(), nbpts);
        kn_solve(kn);
        kn_print_stats(kn);
    }


    GPUBuffer<float> out_pts_w(out_pts);
    GPUBuffer<int> tets_w(out_tets);
    GPUBuffer<VBW::Status> gpu_stat(stat);
    GPUVar<int> gpu_nb_out_tets(nb_tets);
    {
        Stopwatch W("GPU voro kernel only");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        voro_cell_test_GPU_param << < nbpts / block_size + 1, block_size >> > (kn->d_stored_points, nbpts, kn->d_knearests, gpu_stat.gpu_data, tets_w.gpu_data, gpu_nb_out_tets.gpu_x, out_pts_w.gpu_data);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cerr << "kn voro: " << milliseconds << " msec" << std::endl;
    }
    {
        Stopwatch W("copy data back to the cpu");
        cudaMemcpy(pts.data(), kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
#if OUTPUT_TETS
        tets_w.gpu2cpu();
#endif
        gpu_stat.gpu2cpu();
        gpu_nb_out_tets.gpu2cpu();
    }
    kn_free(&kn);
    std::cerr << " \n\n\n---------Summary of success/failure------------\n";
    std::vector<int> nb_statuss(5, 0);
    FOR(i, stat.size()) nb_statuss[stat[i]]++;
    FOR(r, 5) std::cerr << " " << VBW::StatusStr[r] << "   " << nb_statuss[r] << "\n";
}

void compute_voro_diagram_CPU(
    std::vector<float>& pts, std::vector<int>& out_tets, int &nb_tets, std::vector<VBW::Status> &stat, std::vector<float>& out_pts
) {
    nb_tets = 0;
    int nbpts = pts.size() / 3;
    kn_problem *kn = NULL;
    {
        Stopwatch W("GPU KNN");
        kn = kn_prepare(pts.data(), nbpts);
        kn_solve(kn);
        kn_print_stats(kn);
    }
    float* nvpts = kn_get_points(kn);
    unsigned int* knn = kn_get_knearests(kn);
    {
        Stopwatch W("CPU VORO KERNEL");
        FOR(seed, nbpts)
            compute_voro_cell(nvpts, nbpts, knn, stat.data(), out_tets.data(), &nb_tets, out_pts.data(), seed);
    }
    FOR(i, 3 * nbpts) pts[i] = nvpts[i];

    std::cerr << " \n\n\n---------Summary of success/failure------------\n";
    std::vector<int> nb_statuss(5, 0);
    FOR(i, stat.size()) nb_statuss[stat[i]]++;
    FOR(r, 5) std::cerr << " " << VBW::StatusStr[r] << "   " << nb_statuss[r] << "\n";
}



void update_knn(float * pts, int nbpts, int* neighbors) {
    Stopwatch W("Generate KNN");
    KdTree KD(3);
    KD.set_points(nbpts, pts);
#pragma omp parallel for
    FOR(v, nbpts) {
        int neigh[DEFAULT_NB_PLANES + 1];
        float sq_dist[DEFAULT_NB_PLANES + 1];
        KD.get_nearest_neighbors(DEFAULT_NB_PLANES + 1, v, neigh, sq_dist);
        FOR(j, DEFAULT_NB_PLANES) neighbors[v*DEFAULT_NB_PLANES + j] = neigh[j + 1];
    }
}

void drop_xyz_file(float * pts, int nbpts) {
    std::fstream file;
    static int fileid = 0;
    char filename[1024];
    sprintf(filename, "C:\\DATA\\drop_%d_.xyz", fileid);
    fileid++;
    file.open(filename, std::ios_base::out);
    file << nbpts<<std::endl;
    FOR(i, nbpts) file << pts[3 * i] << "  " << pts[3 * i+1] << "  " << pts[3 * i+2] << " \n";
    file.close();
}

void compute_voro_diagram(std::vector<float>& pts, std::vector<int>& out_tets, int &nb_tets) {
    int nbpts = pts.size() / 3;
    std::vector<VBW::Status> stat(nbpts);

    std::vector<float> out_pts(pts.size(),0);


    if (false){// CPU test /debug/stat
        Stopwatch W("CPU run");
        compute_voro_diagram_CPU(pts, out_tets, nb_tets, stat, out_pts);
        VBW::gs.show();
    }
    
    int iter = 5; {
        Stopwatch W("GPU run");
        int block_size = pow(2,iter);
        std::cerr << " block_size = "<< block_size << std::endl;
        compute_voro_diagram_GPU(pts, out_tets, block_size , nb_tets, stat, out_pts);
    }

}

#endif

