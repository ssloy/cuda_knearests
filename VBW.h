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
__shared__ uchar3 tr[32*MAX_T]; // memory pool for chained lists of triangles
__shared__ uchar boundary_next[32*MAX_CLIPS];
__shared__ float4 GPU_clip_eq[32*MAX_CLIPS]; // clipping planes

namespace VBW {

    struct GlobalStats {
        GlobalStats() { reset(); }
        void reset() {
            nb_clips_before_radius.resize(1000, 0);
            nb_non_zero_clips_before_radius.resize(1000, 0);
            nb_removed_voro_vertex_per_clip.resize(1000, 0);
            compute_boundary_iter.resize(1000, 0);
            nbv.resize(1000, 0);
            nbt.resize(1000, 0);
        }
        void start_cell() {
            cur_clip = 6;/* the bbox */
            cur_non_zero_clip = 6;/* the bbox */
        }
        void add_clip(int nb_conflict_vertices) {
            cur_clip++;
            if (nb_conflict_vertices > 0) cur_non_zero_clip++;
            nb_removed_voro_vertex_per_clip[nb_conflict_vertices]++;
        }
        void add_compute_boundary_iter(int nb_iter) { compute_boundary_iter[nb_iter]++; }
        void end_cell() {
            nb_clips_before_radius[cur_clip]++;
            nb_non_zero_clips_before_radius[cur_non_zero_clip]++;
        }
        int cur_clip;
        int cur_non_zero_clip;
        std::vector<int> nbv;
        std::vector<int> nbt;


        std::vector<int> compute_boundary_iter;
        std::vector<int> nb_clips_before_radius;
        std::vector<int> nb_non_zero_clips_before_radius;
        std::vector<int> nb_removed_voro_vertex_per_clip;
        void show() {
            //show_prop(nb_clips_before_radius, false, false," #clips ");
            //show_prop(nb_non_zero_clips_before_radius, true, true, " #NZclips ");
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
            __device__ ConvexCell(int p_seed, float* p_pts);
            __device__ void clip_by_plane(int vid);
            __device__ float4 compute_triangle_point(uchar3 t) const;
            __device__ inline  uchar& ith_plane(uchar t, int i) {
                //           if (i == 0) return tr[threadIdx.x*MAX_T+t].x; if (i == 1) return tr[threadIdx.x*MAX_T+t].y; return tr[threadIdx.x*MAX_T+t].z;
                return reinterpret_cast<uchar *>(tr + threadIdx.x*MAX_T + t)[i];
            }

            template <typename T> __device__ void inline swap_on_device(T& a, T& b) {
                T c(a); a=b; b=c;
            }

            __device__ int new_point(int vid);
            __device__ void new_triangle(uchar i, uchar j, uchar k);
            __device__ void compute_boundary();

            Status status;

            uchar nb_t;                     // API --- number of allocated triangles
            uchar nb_conflicts;             // API --- number of allocated triangles
            float* pts;                     // API --- input pointset
            int voro_id;                    // API --- id of the seed of the current voro cell
            float3 voro_seed;
            uchar nb_v;                     // API --- size of vertices
#if OUTPUT_TETS
            int vorother_id[MAX_CLIPS];     // API --- vertices ids (to compute clipping planes)
#endif        
            float3 B;               // last loaded voro seed position... dirty save of memory access for security ray
            uchar first_boundary_;  // boundary of the last set of triangles that hve been removed
    };

    __device__ ConvexCell::ConvexCell(int p_seed, float* p_pts) {
        float eps  = .1;
        float xmin = -eps;
        float ymin = -eps;
        float zmin = -eps;
        float xmax = 1000 + eps;
        float ymax = 1000 + eps;
        float zmax = 1000 + eps;
        pts = p_pts;
        first_boundary_ = END_OF_LIST;
        FOR(i, MAX_CLIPS) boundary_next[threadIdx.x*MAX_CLIPS+i] = END_OF_LIST;
        voro_id = p_seed;
        voro_seed = make_float3(pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2]);
        status = security_ray_not_reached;

        GPU_clip_eq[threadIdx.x*MAX_CLIPS+0] = make_float4( 1.0,  0.0,  0.0, -xmin);
        GPU_clip_eq[threadIdx.x*MAX_CLIPS+1] = make_float4(-1.0,  0.0,  0.0,  xmax);
        GPU_clip_eq[threadIdx.x*MAX_CLIPS+2] = make_float4( 0.0,  1.0,  0.0, -ymin);
        GPU_clip_eq[threadIdx.x*MAX_CLIPS+3] = make_float4( 0.0, -1.0,  0.0,  ymax);
        GPU_clip_eq[threadIdx.x*MAX_CLIPS+4] = make_float4( 0.0,  0.0,  1.0, -zmin);
        GPU_clip_eq[threadIdx.x*MAX_CLIPS+5] = make_float4( 0.0,  0.0, -1.0,  zmax);
        nb_v = 6;

        tr[threadIdx.x*MAX_T+0] = make_uchar3(2, 5, 0);
        tr[threadIdx.x*MAX_T+1] = make_uchar3(5, 3, 0);
        tr[threadIdx.x*MAX_T+2] = make_uchar3(1, 5, 2);
        tr[threadIdx.x*MAX_T+3] = make_uchar3(5, 1, 3);
        tr[threadIdx.x*MAX_T+4] = make_uchar3(4, 2, 0);
        tr[threadIdx.x*MAX_T+5] = make_uchar3(4, 0, 3);
        tr[threadIdx.x*MAX_T+6] = make_uchar3(2, 4, 1);
        tr[threadIdx.x*MAX_T+7] = make_uchar3(4, 3, 1);
        nb_t = 8;
    }

    __device__ inline float det2x2(float a11, float a12, float a21, float a22) { 
        return a11*a22 - a12*a21; 
    }

    __device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
        return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
    }

    __device__ float4 ConvexCell::compute_triangle_point(uchar3 t) const {
        float4 pi1 = GPU_clip_eq[threadIdx.x*MAX_CLIPS+t.x];
        float4 pi2 = GPU_clip_eq[threadIdx.x*MAX_CLIPS+t.y];
        float4 pi3 = GPU_clip_eq[threadIdx.x*MAX_CLIPS+t.z];
        float4 result;
        result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w =  det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        return result;
    }

    __device__ void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
        if (nb_t+1 >= MAX_T) { 
            status = triangle_overflow; 
            return; 
        }
        tr[threadIdx.x*MAX_T+nb_t] = make_uchar3(i, j, k);
        nb_t++;
    }

    __device__ int ConvexCell::new_point(int vid) {
        if (nb_v == MAX_CLIPS) { 
            status = vertex_overflow; 
            return -1; 
        }
#if OUTPUT_TETS
        vorother_id[nb_v] = vid;
#endif        
        B = make_float3(pts[3 * vid], pts[3 * vid + 1], pts[3 * vid + 2]);
        float3 dir  = make_float3(voro_seed.x - B.x, voro_seed.y - B.y, voro_seed.z - B.z);
        float3 ave2 = make_float3(voro_seed.x + B.x, voro_seed.y + B.y, voro_seed.z + B.z);
        float dot = ave2.x*dir.x + ave2.y*dir.y + ave2.z*dir.z;
        GPU_clip_eq[threadIdx.x*MAX_CLIPS+nb_v] = make_float4(dir.x, dir.y, dir.z, -dot / 2.);
        nb_v++;
        return nb_v - 1;
    }

    __device__ void ConvexCell::compute_boundary() {
        while (first_boundary_ != END_OF_LIST) { // clean circular list of the boundary
            uchar last = first_boundary_;
            first_boundary_ = boundary_next[threadIdx.x*MAX_CLIPS+first_boundary_];
            boundary_next[threadIdx.x*MAX_CLIPS+last] = END_OF_LIST;
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
            FOR(e, 3)   is_in_border[e] = (boundary_next[threadIdx.x*MAX_CLIPS+ith_plane(t, e)] != END_OF_LIST);
            FOR(e, 3)   next_is_opp[e] = (boundary_next[threadIdx.x*MAX_CLIPS+ith_plane(t, (e + 1) % 3)] == ith_plane(t, e));

            bool can_do_it_now = true;
            // check for non manifoldness
            FOR(e, 3) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) can_do_it_now = false;

            // check for more than one boundary ... or first triangle
            if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
                if (first_boundary_ == END_OF_LIST) {
                    FOR(e, 3) boundary_next[threadIdx.x*MAX_CLIPS+ith_plane(t, e)] = ith_plane(t, (e + 1) % 3);
                    first_boundary_ = tr[threadIdx.x*MAX_T+t].x;
                }
                else can_do_it_now = false;
            }

            if (!can_do_it_now) {
                t++;
                if (t == nb_t + nb_conflicts) t = nb_t;
                continue;
            }

            // link next
            FOR(e, 3) if (!next_is_opp[e]) boundary_next[threadIdx.x*MAX_CLIPS+ith_plane(t, e)] = ith_plane(t, (e + 1) % 3);

            // destroy link from removed vertices
            FOR(e, 3)  if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
                if (first_boundary_ == ith_plane(t, (e + 1) % 3)) first_boundary_ = boundary_next[threadIdx.x*MAX_CLIPS+ith_plane(t, (e + 1) % 3)];
                boundary_next[threadIdx.x*MAX_CLIPS+ith_plane(t, (e + 1) % 3)] = END_OF_LIST;
            }

            swap_on_device(tr[threadIdx.x*MAX_T + t], tr[threadIdx.x*MAX_T + nb_t+nb_conflicts-1]);

            t = nb_t;
            nb_conflicts--;
        }
#ifndef __CUDA_ARCH__
        gs.add_compute_boundary_iter(nb_iter);
#endif
    }

    __device__ void  ConvexCell::clip_by_plane(int vid) {
        int cur_v= new_point(vid);
        if (status == vertex_overflow) return;
        float4 eqn = GPU_clip_eq[threadIdx.x*MAX_CLIPS+cur_v];
        nb_conflicts = 0;

        float3 diff2seed = make_float3(B.x - voro_seed.x, B.y  - voro_seed.y, B.z - voro_seed.z);
        float d22seed = diff2seed.x*diff2seed.x + diff2seed.y*diff2seed.y + diff2seed.z*diff2seed.z;

        // Step 1: find conflicts
        int i = 0;
        float dmax2 = 0;
        while (i < nb_t) {
            float4 pc = compute_triangle_point(tr[threadIdx.x*MAX_T+i]);

            // update security radius
            float3 diff = make_float3( pc.x/pc.w - voro_seed.x, pc.y/pc.w - voro_seed.y, pc.z/pc.w - voro_seed.z );
            float d2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z ;
            if (d2 > dmax2) dmax2 = d2;

            // check conflict
            if (eqn.x*pc.x + eqn.y*pc.y + eqn.z*pc.z + eqn.w*pc.w >0){// tr conflict
                swap_on_device(tr[threadIdx.x*MAX_T + i], tr[threadIdx.x*MAX_T + nb_t-1]);
                nb_t--;
                nb_conflicts++;
            }
            else i++;
        }

#ifndef __CUDA_ARCH__
        gs.add_clip(nb_conflicts);
#endif

        if (d22seed > 4. * dmax2) { status = success; return; }
        if (nb_conflicts == 0) {
            nb_v--;
            return;
        }

        // Step 2: compute cavity boundary
        compute_boundary();
        if (status != security_ray_not_reached) return;
        if (first_boundary_ == END_OF_LIST) return;

        // Step 3: Triangulate cavity
        uchar cir = first_boundary_;
        do {
            new_triangle(cur_v, cir, boundary_next[threadIdx.x*MAX_CLIPS+cir]);
            if (status != security_ray_not_reached) return;
            cir = boundary_next[threadIdx.x*MAX_CLIPS+cir];
        } while (cir != first_boundary_);
    }
}


//###################  KERNEL   ######################
__global__ void voro_cell_test_GPU_param(float * pts, int nbpts, unsigned int* neigs, VBW::Status* gpu_stat, int *out_tets, int* nb_out_tet, float* out_pts){
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= nbpts) return;

    FOR(d, 3) out_pts[3 * seed + d] = pts[3 * seed + d];

    VBW::ConvexCell cc(seed, pts);

    FOR(v, DEFAULT_NB_PLANES) {
        cc.clip_by_plane(neigs[DEFAULT_NB_PLANES * seed + v]);
        if (cc.status == VBW::success) break;
        if (cc.status != VBW::security_ray_not_reached) {
            gpu_stat[seed] = cc.status;
            return;
        }
    }
    gpu_stat[seed] = cc.status;

#if OUTPUT_TETS
    FOR(t, nb_t) {
        if (tr[threadIdx.x*MAX_T+t].x > 5 && tr[threadIdx.x*MAX_T+t].y > 5 && tr[threadIdx.x*MAX_T+t].z > 5) {
            uint4 tet = make_uint4(cc.voro_id,0,0,0);
            tet.y = cc.vorother_id[tr[threadIdx.x*MAX_T+t].x];
            tet.z = cc.vorother_id[tr[threadIdx.x*MAX_T+t].y];
            tet.w = cc.vorother_id[tr[threadIdx.x*MAX_T+t].z];
            int top = atomicAdd(nb_out_tet, 1);
            out_tets[top* 4 ] = cc.voro_id;
            FOR(f, 3) out_tets[top * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f)];
        }
    }
#endif    
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
 /*   // CPU stats
    if (0) {
        Stopwatch W("CPU run");
        std::vector<int> neighbors(nbpts * DEFAULT_NB_PLANES, -1);
        update_knn(pts.data(), nbpts, neighbors.data());
        nb_tets = 0;
        FOR(seed, nbpts) voro_cell_test_CPU_param(pts, neighbors, stat.data(), out_tets, &nb_tets, out_pts, seed);
        VBW::gs.show();
        std::cerr << " \n\n\n---------Summary of success/failure------------\n";
        std::vector<int> nb_statuss(5, 0);
        FOR(i, stat.size()) nb_statuss[stat[i]]++;
        FOR(r, 5) std::cerr << " " << VBW::StatusStr[r] << "   " << nb_statuss[r] << "\n";
    }

*/
    //return;
    // test block size
    //FOR(iter, 10)
    int iter = 5; {
        VBW::gs.reset();

        Stopwatch W("GPU run");
        int block_size = pow(2,iter);
        std::cerr << " block_size = "<< block_size << std::endl;
        compute_voro_diagram_GPU(pts, out_tets, block_size , nb_tets, stat, out_pts);

        VBW::gs.show();
        std::cerr << " \n\n\n---------Summary of success/failure------------\n";
        std::vector<int> nb_statuss(5, 0);
        FOR(i, stat.size()) nb_statuss[stat[i]]++;
        FOR(r, 5) std::cerr << " " << VBW::StatusStr[r] << "   " << nb_statuss[r] << "\n";
    }
    //return;
    // test loyd iterations
    /*
    if (0) {
        Stopwatch W("CPU Loyd");
        FOR(iter, 20) {
            VBW::gs.reset();
            update_knn(pts.data(), nbpts, neighbors.data());
            nb_tets = 0;
            FOR(seed, nbpts) voro_cell_test_CPU_param(pts, neighbors, stat.data(), out_tets, &nb_tets, out_pts,seed);
            //VBW::gs.show();
            int block_size = 256;
            compute_voro_diagram_GPU(pts, neighbors,  out_tets, block_size, nb_tets,stat, out_pts);


            FOR(i, 3 * nbpts) pts[i] = out_pts[i];
            drop_xyz_file(pts.data(), nbpts);
            W.tick();
        }
    }
    */
}

#endif

