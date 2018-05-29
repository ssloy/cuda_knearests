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


#ifdef __CUDA_ARCH__
#define IF_CPU(x) 
#define IF_GPU(x) x
#else
#define IF_CPU(x) x 
#define IF_GPU(x) 
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

__host__ __device__ float dist2plane(float4 P, float4 plane) {
    float4 proj = project_on_plane(P, plane);
    return sqrt(dot3(proj,P));
    //return dot4(P, plane);
}


template <typename T>__host__ __device__ void inline swap_on_device(T& a, T& b) {
    T c(a); a = b; b = c;
}


void export_histogram(std::vector<int> h, const std::string& file_name, const std::string& xlabel, const std::string& ylabel) {
    {
        float sum = 0;
        FOR(i, h.size()) sum += .01*float(h[i]);
        int last = 0;
        FOR(i, h.size() - 1) if (h[i] > 0) last = i;
        std::ofstream out("C:\\DATA\\tmp.py");  
        out << "import matplotlib.pyplot as plt\n";
        out << "plt.plot([";
        FOR(i, last) out << float(h[i]) / sum << " , ";
        out << float(h[last]) / sum;
        out << "], drawstyle = \"steps\")\n";
        out << "plt.ylabel('" + ylabel + "')\n";
        out << "plt.xlabel('" + xlabel + "')\n";
        out << "plt.savefig(\"C:/DATA/" + file_name + ".pdf\")\n";
        //out << "plt.show()\n";
    }
    system("python.exe C:\\DATA\\tmp.py");
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

            Status* status;
            uchar nb_t;
            uchar nb_conflicts;
            float* pts;
            int voro_id;
            float4 voro_seed;
            uchar nb_v;
            uchar first_boundary_;     
            IF_OUTPUT_TET(int vorother_id[MAX_CLIPS];)
    };

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

        IF_OUTPUT_TET(vorother_id[nb_v] = vid;)

        float4 B = point_from_ptr3(pts + 3 * vid);
        float4 dir = minus4(voro_seed, B);
        float4 ave2 = plus4(voro_seed, B);
        float dot = dot3(ave2,dir);
        clip(nb_v) = make_float4(dir.x, dir.y, dir.z, -dot / 2.f);
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
            if (nb_iter++ >100) { 
               * status = weird_cavity; 
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



   void export_tet_set(std::string filename, std::vector<float4>& decompose_tet) {
       std::ofstream out(filename);
       out << decompose_tet.size() << " vertices" << std::endl;
       out << decompose_tet.size() / 4 << " tets" << std::endl;
       FOR(v, decompose_tet.size())   out << decompose_tet[v].x << " " << decompose_tet[v].y << " " << decompose_tet[v].z << std::endl;
       FOR(j, decompose_tet.size() / 4)  out << "4 " << 4 * j << " " << 4 * j + 1 << " " << 4 * j + 2 << " " << 4 * j + 3 << " \n";
   }


   __host__ __device__ void get_tet_decomposition_of_vertex(ConvexCell& cc, int t, float4* P) {
       float4 C = cc.voro_seed;
       float4 A4 = cc.compute_triangle_point(tr(t));
       float4 A = normalize4(A4);
       FOR(i,3)        P[2*i] = project_on_plane(C, clip(cc.ith_plane(t,i)));
       FOR(i, 3)        P[2 * i+1] = project_on_plane(A, plane_from_point_and_normal(C, cross3(minus4(P[2*i], C), minus4(P[(2*(i+1))%6], C))));
   }


   __host__ __device__ void export_bary_and_volume(ConvexCell& cc,float* out_pts,int seed) {
  
       static float all_vol = 0;
       float4 bary; float vol;
       float4 bary_sum = make_float4(0, 0, 0,0); float vol_sum = 0;
       float4 P[6];
       float4 C = cc.voro_seed;
       FOR(t, cc.nb_t) {
           float4 A4 = cc.compute_triangle_point(tr(t));
           float4 A = normalize4(A4);
           get_tet_decomposition_of_vertex(cc, t, P);
           FOR(i, 6) {
               get_tet_volume_and_barycenter(bary, vol, P[i], P[(i + 1) % 6], C, A);
               bary_sum = plus4(bary_sum, mul3(vol, bary)); 
               vol_sum += vol;
           }
       }
       
      

#if REPLACE_BARY_BY_PRESSURE
       FOR(t, cc.nb_t) {
           float4 A4 = cc.compute_triangle_point(tr(t));
           float4 A = normalize4(A4);
           get_tet_decomposition_of_vertex(cc, t, P);
           FOR(i, 6) {
               get_tet_volume_and_barycenter(bary, vol, P[i], P[(i + 1) % 6], C, A);
               int clip_id = cc.ith_plane(t, ((i + 1) / 2) % 3);
               //float triangle_area = 6.*vol / dist2plane(C,clip(clip_id));
               float4 d0 = minus4(P[(i + 1) % 6], A);
               float4 d1 = minus4(P[i] , A);
               float4 cross = cross3(d0, d1);
               float triangle_area = .5*sqrt(dot3(cross,cross));
               if (vol < 1e-5) triangle_area *= -1.;

               float4 dir = clip(clip_id);
               dir.w = sqrt(dot3(dir, dir));
               dir = normalize4(dir);
               



               if (clip_id >5) {
                   float pressure = 1. / vol_sum;
                   float str = -triangle_area* pressure;

                   int ne = cc.vorother_id[clip_id];

                   out_pts[3 * ne + 0] += str * dir.x;
                   out_pts[3 * ne + 1] += str * dir.y;
                   out_pts[3 * ne + 2] += str * dir.z;
               }
               else {
                   float pressure = 3000./ 1e9;
                   float str = triangle_area* pressure;
                   out_pts[3 * seed + 0] += str *dir.x;
                   out_pts[3 * seed + 1] += str *dir.y;
                   out_pts[3 * seed + 2] += str * dir.z;

               }


               //point_from_ptr3(cc.pts + cc.vorother_id[clip_id]);
               //printf("%f \n",triangle_area);
           }
       }
#else
       //IF_CPU(all_vol += vol_sum; std::cerr << all_vol << "  \n";)
       out_pts[3 * seed] = bary_sum.x / vol_sum - C.x;
       out_pts[3 * seed + 1] = bary_sum.y / vol_sum - C.y;
       out_pts[3 * seed + 2] = bary_sum.z / vol_sum - C.z;
       return;
#endif
   }



   std::vector<float4> decompose_tet;
    __host__ void export_cell_decomposition(ConvexCell& cc) {
       // compute bary
       float4 P[6];
       FOR(t, cc.nb_t) {
           float4 A4 = cc.compute_triangle_point(tr(t));
           float4 A = normalize4(A4);
           get_tet_decomposition_of_vertex(cc, t, P);
           FOR(i, 6) {
               decompose_tet.push_back(P[i]);
               decompose_tet.push_back(P[(i + 1) % 6]);
               decompose_tet.push_back(cc.voro_seed);
               decompose_tet.push_back(A);
           }
       }

   }
    __host__ void  export_decomposition() {
        static int fileid = 0; 
        fileid++;
        char filename[1024];
        sprintf(filename, "C:\\DATA\\tet_%03d_decomp.tet", fileid);
        std::cerr << filename << std::endl;
        std::ofstream out(filename);
        //std::ofstream out("C:\\DATA\\outdecomp.tet");
        out << decompose_tet.size() << " vertices" << std::endl;
        out << decompose_tet.size() / 4 << " tets" << std::endl;
        FOR(v, decompose_tet.size())   out << decompose_tet[v].x << " " << decompose_tet[v].y << " " << decompose_tet[v].z << std::endl;
        FOR(j, decompose_tet.size() / 4)  out << "4 " << 4 * j << " " << 4 * j + 1 << " " << 4 * j + 2 << " " << 4 * j + 3 << " \n";
    }


   __host__ __device__ void output_tets(ConvexCell& cc, int *out_tets, int* nb_out_tet) {
       #ifdef OUTPUT_TET
       FOR(t, cc.nb_t) {
           if (tr(t).x > 5 && tr(t).y > 5 && tr(t).z > 5) {
               uint4 tet = make_uint4(cc.voro_id, 0, 0, 0);
               tet.y = cc.vorother_id[tr(t).x];
               tet.z = cc.vorother_id[tr(t).y];
               tet.w = cc.vorother_id[tr(t).z];
               int top;
               IF_GPU(top = atomicAdd(nb_out_tet, 1);)
               IF_CPU((*nb_out_tet)++; top = *nb_out_tet;)
               out_tets[top * 4] = cc.voro_id;
               FOR(f, 3) out_tets[top * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f)];
           }
       }
#endif
   }


//###################  KERNEL   ######################
   __host__ __device__ void compute_voro_cell(float * pts, int nbpts, unsigned int* neigs, Status* gpu_stat, int *out_tets, int* nb_out_tet, float* out_pts, int seed) {
       
       //create BBox
       ConvexCell cc(seed, pts, &(gpu_stat[seed]));

       // clip by halfspaces
       FOR(v, DEFAULT_NB_PLANES) {
           cc.clip_by_plane(neigs[DEFAULT_NB_PLANES * seed + v]);
           if (gpu_stat[seed] != success) return;
       }

       // check security ray
       {
           // finds furthest voro vertex distance
           float v_dist = 0;
           FOR(i, cc.nb_t) {
               float4 pc = cc.compute_triangle_point(tr(i));
               pc = normalize4(pc);
               float4 diff = minus4(pc, cc.voro_seed);
               float d2 = dot3(diff, diff);
               v_dist = max(d2, v_dist);
           }
           //find furthest neigborgs distance
           float neig_dist = 0;
           FOR(v, DEFAULT_NB_PLANES) {
               unsigned int vid = neigs[DEFAULT_NB_PLANES * seed + v];

               if (vid >= nbpts) continue;
                   //IF_CPU(if (vid >= nbpts) std::cerr << vid << " ";)
                float4 pc = point_from_ptr3(pts + 3 * vid);
               float4 diff = minus4(pc, cc.voro_seed);
               float d2 = dot3(diff, diff);
               neig_dist = max(d2, neig_dist);
           }
           // reject cell without enough neighborgs
           if (neig_dist < 4 * v_dist)
               gpu_stat[seed] = security_ray_not_reached;
       }

       
       if (gpu_stat[seed] != success && gpu_stat[seed] != security_ray_not_reached) return;
       if (gpu_stat[seed] != success ) return;

       IF_OUTPUT_P2BARY(export_bary_and_volume(cc, out_pts, seed));
       IF_OUTPUT_TET(output_tets(cc, out_tets, nb_out_tet));
       IF_EXPORT_DECOMPOSITION(export_cell_decomposition(cc));

   }



//#################################################################################"
//#############################      GPU ONLY        ####################################"
//#################################################################################"



//----------------------------------KERNEL
__global__ void voro_cell_test_GPU_param(float * pts, int nbpts, unsigned int* neigs, Status* gpu_stat, int *out_tets, int* nb_out_tet, float* out_pts) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= nbpts) return;
    compute_voro_cell(pts, nbpts, neigs, gpu_stat, out_tets, nb_out_tet, out_pts, seed);
}


//----------------------------------WRAPPER
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
void compute_voro_diagram_GPU(std::vector<float>& pts, std::vector<int>& out_tets , int &nb_tets, std::vector<Status> &stat, std::vector<float>& out_pts, unsigned int ** permutation, int block_size) {
    int nbpts = pts.size() / 3;
    nb_tets = 0;
    kn_problem *kn = NULL;
    {
        IF_VERBOSE(Stopwatch W("GPU KNN"));
        kn = kn_prepare(pts.data(), nbpts);
        kn_solve(kn);
        kn_print_stats(kn);
    }

    IF_OUTPUT_TET(FOR(i, pts.size()) out_pts[i] = 0;)
    GPUBuffer<float> out_pts_w(out_pts);
    GPUBuffer<int> tets_w(out_tets);
    GPUBuffer<Status> gpu_stat(stat);
    GPUVar<int> gpu_nb_out_tets(nb_tets);
    {
        IF_VERBOSE(Stopwatch W("GPU voro kernel only"));

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
        IF_VERBOSE(std::cerr << "kn voro: " << milliseconds << " msec" << std::endl);
    }
    {
        IF_VERBOSE(Stopwatch W("copy data back to the cpu"));
        cudaMemcpy(pts.data(), kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
        IF_OUTPUT_TET(tets_w.gpu2cpu());
        IF_OUTPUT_P2BARY(out_pts_w.gpu2cpu());
        gpu_stat.gpu2cpu();
        gpu_nb_out_tets.gpu2cpu();
    }

    if (permutation != NULL) *permutation = kn_get_permutation(kn);

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
    std::vector<float>& pts, std::vector<int>& out_tets, int &nb_tets, std::vector<Status> &stat, std::vector<float>& out_pts,unsigned int ** permutation
) {
    nb_tets = 0;
    int nbpts = pts.size() / 3;
    
    // compute knn
    kn_problem *kn = NULL;
    {
        Stopwatch W("GPU KNN");
        kn = kn_prepare(pts.data(), nbpts);
        kn_solve(kn);
        kn_print_stats(kn);
    }
    float* nvpts = kn_get_points(kn);
    unsigned int* knn = kn_get_knearests(kn);

    // run voro on the cpu
    {
        Stopwatch W("CPU VORO KERNEL");
        IF_OUTPUT_TET(FOR(i, pts.size()) out_pts[i] = 0;)
            FOR(seed, nbpts) {
            compute_voro_cell(nvpts, nbpts, knn, stat.data(), out_tets.data(), &nb_tets, out_pts.data(), seed);
            IF_EXPORT_DECOMPOSITION(export_decomposition();)
                decompose_tet.clear();
        }
    }
    FOR(i, 3 * nbpts) pts[i] = nvpts[i];
    if (permutation!=NULL) *permutation = kn_get_permutation(kn);

    
    // ouput stats
    IF_VERBOSE(std::cerr << " \n\n\n---------Summary of success/failure------------\n");

    std::vector<int> nb_statuss(5, 0);
    FOR(i, stat.size()) nb_statuss[stat[i]]++;
    IF_VERBOSE(FOR(r, 5) std::cerr << " " << StatusStr[r] << "   " << nb_statuss[r] << "\n";)
        std::cerr << " " << StatusStr[4] << "   " << nb_statuss[4] << " /  "<< nbpts <<"\n";
    IF_EXPORT_HISTO(gs.show();)
}

#endif

