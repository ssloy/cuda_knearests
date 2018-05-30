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
void cuda_check_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}

#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); ++I)

typedef unsigned char uchar;  // local indices with special values


#ifdef __CUDA_ARCH__
#define IF_CPU(x) 
#else
#define IF_CPU(x) x 
#endif




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






__host__ __device__ float4 point_from_ptr3(float* f) {
    return make_float4(f[0], f[1], f[2], 1);
}
__host__ __device__ float4 minus4(float4 A, float4 B) {
    return make_float4(A.x - B.x, A.y - B.y, A.z - B.z, A.w - B.w);
}
__host__ __device__ float4 plus4(float4 A, float4 B) {
    return make_float4(A.x + B.x, A.y + B.y, A.z + B.z, A.w + B.w);
}
__host__ __device__ float dot4(float4 A, float4 B) {
    return A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w;
}
__host__ __device__ float dot3(float4 A, float4 B) {
    return A.x * B.x + A.y * B.y + A.z * B.z;
}
__host__ __device__ float4 normalize4(float4 A) {
    return make_float4(A.x / A.w, A.y / A.w, A.z / A.w, 1);
}
__host__ __device__ float norm23(float4 A) {
    return dot3(A, A);
}
__host__ __device__ float4 mul3(float s, float4 A) {
    return make_float4(s*A.x, s*A.y, s*A.z, 1.);
}
__host__ __device__ float4 cross3(float4 A, float4 B) {
    return make_float4(A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x, 0);
}
__host__ __device__ float4 plane_from_point_and_normal(float4 P, float4 n) {
    return  make_float4(n.x, n.y, n.z, -dot3(P, n));
}


__host__ __device__ inline float det2x2(float a11, float a12, float a21, float a22) {
    return a11*a22 - a12*a21;
}
__host__ __device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
    return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}

__host__ __device__ void get_tet_volume(float& volume, float4 A, float4 B, float4 C) {
    volume = -det3x3(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z)/6.;
}
__host__ __device__ void get_tet_volume_and_barycenter(float4& bary, float& volume, float4 A, float4 B, float4 C, float4 D) {
    get_tet_volume(volume, minus4(A, D), minus4(B, D), minus4(C, D));
    bary = make_float4(.25*(A.x + B.x + C.x + D.x), .25*(A.y + B.y + C.y + D.y), .25*(A.z + B.z + C.z + D.z),1);
}

__host__ __device__ float4 project_on_plane(float4 P, float4 plane) {
    float4 n = make_float4(plane.x, plane.y, plane.z,0);
    float lambda;
    lambda = (dot3(n, P) + plane.w) / norm23(n);
    return plus4(P, mul3(-lambda, n));
}




template <typename T>__host__ __device__ void inline swap_on_device(T& a, T& b) { T c(a); a = b; b = c; }

void drop_xyz_file(std::vector<float>& pts, const char* filename = NULL) {
    if (filename == NULL) {
        static int fileid = 0;
        char name[1024];
        sprintf(name, "C:\\DATA\\dr_%d_.xyz", fileid);
        fileid++;
        drop_xyz_file(pts, name);
        return;
    }
    std::fstream file;
    file.open(filename, std::ios_base::out);
    file << pts.size() / 3 << std::endl;
    FOR(i, pts.size() / 3) file << pts[3 * i] << "  " << pts[3 * i + 1] << "  " << pts[3 * i + 2] << " \n";
    file.close();
}

void export_histogram(std::vector<int> h, const std::string& file_name, const std::string& xlabel, const std::string& ylabel) {
    {
        float sum = 0;
        FOR(i, h.size()) sum += .01*float(h[i]);
        if (sum == 0) sum = 1;
        int last = 0;
        FOR(i, h.size() - 1) if (h[i] > 0) last = i;
#if defined(__linux__)
        std::ofstream out("tmp.py");
#else
        std::ofstream out("C:\\DATA\\tmp.py");
#endif        
        out << "import matplotlib.pyplot as plt\n";
        out << "y = [";
        FOR(i, last) out << float(h[i]) / sum << " , ";
        out << float(h[last]) / sum;
        out << "]\n";
        out << "x = [v-.5 for v in range(len(y))]\n";
        out << "plt.fill_between(x, [0 for i in y], y, step=\"post\", alpha=.8)\n";
        out << "#plt.step(x, y, where='post')\n";
        out << "plt.tick_params(axis='both', which='major', labelsize=12)\n";
        out << "plt.ylabel('" + ylabel + "', fontsize=14)\n";
        out << "plt.xlabel('" + xlabel + "', fontsize=14)\n";
#if defined(__linux__)
        out << "plt.show()\n";
    }
    system("python3 *.py");
#else
        out << "plt.savefig(\"C:/DATA/" + file_name + ".pdf\")\n";
    }
    system("python.exe C:\\DATA\\tmp.py");
#endif
}

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
            export_histogram(nb_clips_before_radius, "nb_clips_before_radius", "#required clip planes", "proportion %");
            export_histogram(nbv, "nbv", "#intersecting clip planes", "proportion %");
            export_histogram(nbt, "nbt", "#Voronoi vertices", "proportion %");
            export_histogram(nb_removed_voro_vertex_per_clip, "nb_removed_voro_vertex_per_clip", "#R", "proportion %");
            export_histogram(compute_boundary_iter, "compute_boundary_iter", "#iter compute void boundary", "proportion %");
        }
    } gs;


    static const uchar END_OF_LIST = 255;
    enum Status { triangle_overflow = 0, vertex_overflow = 1, weird_cavity = 2, security_ray_not_reached = 3, success = 4 } ;
    char StatusStr[5][128] = { "triangle_overflow","vertex_overflow ","weird_cavity "," security_ray_not_reached ","success" };

    class ConvexCell {
        public:
           __host__ __device__ ConvexCell(int p_seed, float* p_pts, Status* p_status);
           __host__ __device__ void clip_by_plane(int vid);
           __host__ __device__ float4 compute_triangle_point(uchar3 t) const;
           __host__ __device__ inline  uchar& ith_plane(uchar t, int i);
           __host__ __device__ int new_point(int vid);
           __host__ __device__ void new_triangle(uchar i, uchar j, uchar k);
           __host__ __device__ void compute_boundary();
           __host__ __device__  bool security_ray_is_reached(float4 last_neig);

            Status* status;
            uchar nb_t;
            uchar nb_conflicts;
            float* pts;
            int voro_id;
            float4 voro_seed;
            uchar nb_v;
            uchar first_boundary_;     
    };


    __host__ __device__  bool ConvexCell::security_ray_is_reached(float4 last_neig) {
        // finds furthest voro vertex distance2
        float v_dist = 0;
        FOR(i, nb_t) {
            float4 pc = compute_triangle_point(tr(i));
            pc = normalize4(pc);
            float4 diff = minus4(pc, voro_seed);
            float d2 = dot3(diff, diff);
            v_dist = max(d2, v_dist);
        }
        //compare to new neigborgs distance2
        float4 pc = last_neig;
        float4 diff = minus4(pc, voro_seed);
        float d2 = dot3(diff, diff);
        return (d2> 4 * v_dist);
    }


    __host__ __device__ inline  uchar& ConvexCell::ith_plane(uchar t, int i) {
        return reinterpret_cast<uchar *>(&(tr(t)))[i];
    }


   __host__ __device__ ConvexCell::ConvexCell(int p_seed, float* p_pts,Status *p_status) {
        float eps  = .1f;
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
        voro_seed = make_float4(pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2],1);
        status = p_status;
        *status = success;

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
            *status = triangle_overflow; 
            return; 
        }
        tr(nb_t) = make_uchar3(i, j, k);
        nb_t++;
    }

   __host__ __device__ int ConvexCell::new_point(int vid) {
        if (nb_v == MAX_CLIPS) { 
            *status = vertex_overflow; 
            return -1; 
        }


        float4 B = point_from_ptr3(pts + 3 * vid);
        float4 dir = minus4(voro_seed, B);
        float4 ave2 = plus4(voro_seed, B);
        float dot = dot3(ave2,dir);
        clip(nb_v) = make_float4(dir.x, dir.y, dir.z, -dot / 2.f);
        nb_v++;
        return nb_v - 1;
    }

   __host__ __device__ void ConvexCell::compute_boundary() {
       // clean circular list of the boundary
       FOR(i, MAX_CLIPS) boundary_next(i) = END_OF_LIST;
       first_boundary_ = END_OF_LIST;
       //while (first_boundary_ != END_OF_LIST) {
       //     uchar last = first_boundary_;
       //     first_boundary_ = boundary_next(first_boundary_);
       //     boundary_next(last) = END_OF_LIST;
       // } 
        

        int nb_iter = 0;
        uchar t = nb_t;
        while (nb_conflicts>0) {
            if (nb_iter++ >100) { 
               * status = weird_cavity; 
                return; 
            }
            bool is_in_border[3];
            bool next_is_opp[3];
            FOR(e, 3)   is_in_border[e] = (boundary_next(ith_plane(t, e)) != END_OF_LIST);
            FOR(e, 3)   next_is_opp[e] = (boundary_next(ith_plane(t, (e + 1) % 3)) == ith_plane(t, e));

            bool new_border_is_simple = true;
            // check for non manifoldness
            FOR(e, 3) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) new_border_is_simple = false;

            // check for more than one boundary ... or first triangle
            if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
                if (first_boundary_ == END_OF_LIST) {
                    FOR(e, 3) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                    first_boundary_ = tr(t).x;
                }
                else new_border_is_simple = false;
            }

            if (!new_border_is_simple) {
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

            //remove triangle from R, and restart iterating on R
            swap_on_device(tr(t), tr(nb_t+nb_conflicts-1));
            t = nb_t;
            nb_conflicts--;
        }

        IF_CPU(gs.add_compute_boundary_iter(nb_iter);)
    }

   __host__ __device__ void  ConvexCell::clip_by_plane(int vid) {
        int cur_v= new_point(vid);
        if (*status == vertex_overflow) return;
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

        IF_CPU(gs.add_clip(nb_conflicts);)

        // do not waste memory with clipping plane that did not intersect the cell
        if (nb_conflicts == 0) {
            nb_v--;
            return;
        }

        // Step 2: compute cavity boundary
        compute_boundary();
        if (*status != success) return;
        if (first_boundary_ == END_OF_LIST) return;

        // Step 3: Triangulate cavity
        uchar cir = first_boundary_;
        do {
            new_triangle(cur_v, cir, boundary_next(cir));
            if (*status != success) return;
            cir = boundary_next(cir);
        } while (cir != first_boundary_);
    }






   __host__ __device__ void get_tet_decomposition_of_vertex(ConvexCell& cc, int t, float4* P) {
       float4 C = cc.voro_seed;
       float4 A4 = cc.compute_triangle_point(tr(t));
       float4 A = normalize4(A4);
       FOR(i,3)        P[2*i] = project_on_plane(C, clip(cc.ith_plane(t,i)));
       FOR(i, 3)        P[2 * i+1] = project_on_plane(A, plane_from_point_and_normal(C, cross3(minus4(P[2*i], C), minus4(P[(2*(i+1))%6], C))));
   }

   std::vector<float4> decompose_tet;
   __host__ __device__ void export_bary_and_volume(ConvexCell& cc,float* out_pts,int seed) {
  
       float4 tet_bary; 
       float tet_vol;
       float4 bary_sum = make_float4(0, 0, 0,0); 
       float cell_vol = 0;
       float4 P[6];
       float4 C = cc.voro_seed;

       FOR(t, cc.nb_t) {
           float4 A4 = cc.compute_triangle_point(tr(t));
           float4 A = normalize4(A4);

           get_tet_decomposition_of_vertex(cc, t, P);
           FOR(i, 6) {
               get_tet_volume_and_barycenter(tet_bary, tet_vol, P[i], P[(i + 1) % 6], C, A);
               bary_sum = plus4(bary_sum, mul3(tet_vol, tet_bary));
               cell_vol += tet_vol;
           }
//#ifdef EXPORT_DECOMPOSITION
#ifndef __CUDA_ARCH__
           FOR(i, 6) {
               decompose_tet.push_back(P[i]);
               decompose_tet.push_back(P[(i + 1) % 6]);
               decompose_tet.push_back(cc.voro_seed);
               decompose_tet.push_back(A);
           }
#endif
//#endif
       }
       out_pts[3 * seed] = bary_sum.x / cell_vol;
       out_pts[3 * seed + 1] = bary_sum.y / cell_vol;
       out_pts[3 * seed + 2] = bary_sum.z / cell_vol;
       return;
   }



    __host__ void  export_decomposition() {
        std::ofstream out("C:\\DATA\\outdecomp.tet");
        out << decompose_tet.size() << " vertices" << std::endl;
        out << decompose_tet.size() / 4 << " tets" << std::endl;
        FOR(v, decompose_tet.size())   out << decompose_tet[v].x << " " << decompose_tet[v].y << " " << decompose_tet[v].z << std::endl;
        FOR(j, decompose_tet.size() / 4)  out << "4 " << 4 * j << " " << 4 * j + 1 << " " << 4 * j + 2 << " " << 4 * j + 3 << " \n";
        decompose_tet.clear();
    }






//###################  KERNEL   ######################
   __host__ __device__ void compute_voro_cell(float * pts, int nbpts, unsigned int* neigs, Status* gpu_stat, float* out_pts, int seed) {

       FOR(d, 3) out_pts[3 * seed + d] = pts[3 * seed + d];

       //create BBox
       ConvexCell cc(seed, pts, &(gpu_stat[seed]));

       // clip by halfspaces
       FOR(v, DEFAULT_NB_PLANES) {
           cc.clip_by_plane(neigs[DEFAULT_NB_PLANES * seed + v]);
#ifndef __CUDA_ARCH__
           if (cc.security_ray_is_reached(point_from_ptr3(pts + 3*neigs[DEFAULT_NB_PLANES * seed + v]))) {
               IF_CPU(gs.nb_clips_before_radius[v]++);
               break;
           }
#endif
           if (gpu_stat[seed] != success) return;
       }
       IF_CPU(gs.nbv[cc.nb_v]++;);
       IF_CPU(gs.nbt[cc.nb_t]++;);
       // check security ray
       if (!cc.security_ray_is_reached(point_from_ptr3(pts + 3 * neigs[DEFAULT_NB_PLANES * (seed+1) -1]))) {
           gpu_stat[seed] = security_ray_not_reached;
           return;
       }

       
       //if (gpu_stat[seed] != success ) return; // compute bary anyway
       
       export_bary_and_volume(cc, out_pts, seed);
   }



//#################################################################################"
//#############################      GPU ONLY        ####################################"
//#################################################################################"



//----------------------------------KERNEL
__global__ void voro_cell_test_GPU_param(float * pts, int nbpts, unsigned int* neigs, Status* gpu_stat, float* out_pts) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= nbpts) return;
    compute_voro_cell(pts, nbpts, neigs, gpu_stat, out_pts, seed);
}


//----------------------------------WRAPPER
template <class T> struct GPUBuffer {
    void init(T* data) {
        IF_VERBOSE(std::cerr << "GPU: " << size * sizeof(T)/1048576 << " Mb used" << std::endl);
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



//----------------------------------FUNCTION TO CALL
void compute_voro_diagram_GPU(std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary, int block_size,int nb_Lloyd_iter) {
    int nbpts = pts.size() / 3;
    kn_problem *kn = NULL;
    {
        IF_VERBOSE(Stopwatch W("GPU KNN"));
        kn = kn_prepare((float3*) pts.data(), nbpts);
        kn_solve(kn);
        kn_print_stats(kn);
    }

    GPUBuffer<float> out_pts_w(bary);
    GPUBuffer<Status> gpu_stat(stat);
    {
        IF_VERBOSE(Stopwatch W("GPU voro kernel only"));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        voro_cell_test_GPU_param << < nbpts / block_size + 1, block_size >> > ((float*)kn->d_stored_points, nbpts, kn->d_knearests, gpu_stat.gpu_data, out_pts_w.gpu_data);
        cuda_check_error();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        IF_VERBOSE(std::cerr << "kn voro: " << milliseconds << " msec" << std::endl);
    }

    // Lloyd
    FOR(lit,nb_Lloyd_iter){
        IF_VERBOSE(Stopwatch W("Loyd iterations"));
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        voro_cell_test_GPU_param << < nbpts / block_size + 1, block_size >> > (out_pts_w.gpu_data, nbpts, kn->d_knearests, gpu_stat.gpu_data, (float*)kn->d_stored_points);
        cuda_check_error();
        voro_cell_test_GPU_param << < nbpts / block_size + 1, block_size >> > ((float*)kn->d_stored_points, nbpts, kn->d_knearests, gpu_stat.gpu_data, out_pts_w.gpu_data);
        cuda_check_error();

        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        IF_VERBOSE(std::cerr << "kn voro: " << milliseconds << " msec" << std::endl);
    }


    {
        IF_VERBOSE(Stopwatch W("copy data back to the cpu"));
        cudaMemcpy(pts.data(), kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
        cuda_check_error();
        out_pts_w.gpu2cpu();
        gpu_stat.gpu2cpu();
    }



    kn_free(&kn);
    IF_VERBOSE(std::cerr << " \n\n\n---------Summary of success/failure------------\n");
    std::vector<int> nb_statuss(5, 0);
    FOR(i, stat.size()) nb_statuss[stat[i]]++;
    IF_VERBOSE(FOR(r, 5) std::cerr << " " << StatusStr[r] << "   " << nb_statuss[r] << "\n";)
        std::cerr << " " << StatusStr[4] << "   " << nb_statuss[4] << " /  " << nbpts << "\n";
}

//#################################################################################"
//#############################      CPU ONLY        ####################################"
//#################################################################################"

void compute_voro_diagram_CPU(
    std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary
) {
    int nbpts = pts.size() / 3;
    
    // compute knn
    kn_problem *kn = NULL;
    {
        Stopwatch W("GPU KNN");
        kn = kn_prepare((float3*) pts.data(), nbpts);
        kn_solve(kn);
        kn_print_stats(kn);
    }
    float* nvpts = (float*) kn_get_points(kn);
    unsigned int* knn = kn_get_knearests(kn);

    // run voro on the cpu
    {
        Stopwatch W("CPU VORO KERNEL");
            FOR(seed, nbpts)  compute_voro_cell(nvpts, nbpts, knn, stat.data(), bary.data(), seed);
#ifdef EXPORT_DECOMPOSITION
            export_decomposition();
#endif

            
    }
    drop_xyz_file(bary);
    drop_xyz_file(pts);
    
    // ouput stats
    IF_VERBOSE(std::cerr << " \n\n\n---------Summary of success/failure------------\n");
    
    FOR(r, 5) {
        std::vector<float> catpts;
        FOR(i, stat.size()) if (stat[i]==r) FOR(d,3) catpts.push_back(nvpts [3 * i + d]);
        drop_xyz_file(catpts);
    }

    std::vector<int> nb_statuss(5, 0);
    FOR(i, stat.size()) nb_statuss[stat[i]]++;
    IF_VERBOSE(FOR(r, 5) std::cerr << " " << StatusStr[r] << "   " << nb_statuss[r] << "\n";)
        std::cerr << " " << StatusStr[4] << "   " << nb_statuss[4] << " /  "<< nbpts <<"\n";
    IF_EXPORT_HISTO(gs.show();)
}

#endif

