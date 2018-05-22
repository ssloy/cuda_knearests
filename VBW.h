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

namespace VBW {
	void show_prop(std::vector<int> &h, bool show_cumul, bool show_numbers, const char * name) {
		bool cumulative = false;
		std::cerr << "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		std::cerr << "******************* HISTOGRAM OF " << name << "*****************\n";
		do {
			std::cerr << "\n\n#################    " << (cumulative ? "CUMULATIVE " : " NORMAL ") << "    #################\n";
			double sum = 0;
			FOR(i, h.size()) sum += h[i];
			int last = 0;
			FOR(i, h.size()) if (h[i] > 0) last = i;
			std::vector<int> display(last + 1);
			double acc = 0;
			float max_val = 0;
			FOR(i, last + 1) {
				double prop = double(h[i]) / sum;
				acc += prop;
				if (cumulative) display[i] = int(1000. - 1000.*acc); else display[i] = int(1000.*prop);
				if (display[i] > max_val)max_val = display[i];
			}
			if (cumulative)max_val = 1000.;
			float scale = max_val/10.;
			FOR(i, 100)std::cerr << "_";
			FOR(line, 10) {
				float l0 = 9 - line;
				float l1 = 10 - line;
				std::cerr << "\n" << int(l0*scale) << "\t";
				FOR(i, last + 1) {
					if (display[i] > l1*scale)  std::cerr << char(219);
					else if (display[i] > (.33*l0 + .64*l1)*scale) std::cerr << char(178);
					else if (display[i] > (64 * l0 + .33*l1)*scale) std::cerr << char(177);
					else if (display[i] > l0 *scale) std::cerr << char(176);
					else std::cerr << " ";
				}
			}
			std::cerr << "\n\t0";
			FOR(i, 10) {
				FOR(j, 8)	 std::cerr << char(249);
				std::cerr << 10 * (i + 1);
			}
			if (show_numbers) {
				int nb_val_per_line = 30;
				FOR(line, (last + 1) / nb_val_per_line + 1) {
					std::cerr << "\n\nid = ";
					for (int i = nb_val_per_line *line; i < nb_val_per_line * (line + 1) && i < last + 1; i++)
						std::cerr << std::setw(4) << i;
					std::cerr << "\nnb = ";
					for (int i = nb_val_per_line * line; i < nb_val_per_line * (line + 1) && i < last + 1; i++)
						std::cerr << std::setw(4) << display[i];
				}
			}
		cumulative = !cumulative;
	}while (cumulative && show_cumul);
	}



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
			show_prop(nb_clips_before_radius, false, false," #clips ");
			//show_prop(nb_non_zero_clips_before_radius, true, true, " #NZclips ");
			//show_prop(nb_removed_voro_vertex_per_clip, true, true, " #nb_removed_voro_vertex_per_clip ");
			//show_prop(compute_boundary_iter, true, true, " #compute_boundary_iter ");
			//show_prop(nbv, true, true, " #nbv ");
			//show_prop(nbv, true, true, " #nbt ");

		}
	} gs;


	typedef unsigned char uchar;		// local indices with special values
	static const  uchar  END_OF_LIST = 255;

	enum Statut { triangle_overflow = 0, vertex_overflow = 1, weird_cavity = 2, security_ray_not_reached = 3, success = 4 } ;
	char StatutStr[5][128]={ "triangle_overflow","vertex_overflow ","weird_cavity "," security_ray_not_reached ","success" };

	class ConvexCell {
	public:
		__host__ __device__ ConvexCell(int p_seed, float* p_pts);
		__host__ __device__ void clip_by_plane(int vid);
		__host__ __device__ float4 compute_triangle_point(uchar3 t) const;
		__host__ __device__ inline  uchar& ith_plane(uchar t, int i) { if (i == 0) return tr[t].x; if (i == 1) return tr[t].y; return tr[t].z; }
		__host__ __device__ void switch_triangles(uchar  t0, uchar t1);
		__host__ __device__ int new_point(int vid);
		__host__ __device__ void new_triangle(uchar i, uchar j, uchar k);
		__host__ __device__ void compute_boundary();

		Statut statut;

		/*voronoi vertices stored as triangles */
		uchar nb_t;					// API --- number of allocated triangles
		uchar nb_conflicts;				// API --- number of allocated triangles
		uchar3 tr[MAX_T];				// API --- memory pool for chained lists of triangles
		float4 X[MAX_T];				// position of voro vertices

		/*pointset*/
		float* pts;						// API --- input pointset
	
		/*clipping planes*/
		int voro_id;					// API --- id of the seed of the current voro cell 
		float3 voro_seed;
		uchar nb_v;					// API --- size of vertices
		int vorother_id[MAX_CLIPS];			// API --- vertices ids (to compute clipping planes)  
		float4 GPU_clip_eq[MAX_CLIPS];		// explicit plane equations from the bbox

		float3 B;						// last loaded voro seed position... dirty save of memory access for security ray




		/*boundary of the last set of triangles that hve been removed*/
		uchar first_boundary_;
		uchar boundary_next[MAX_CLIPS];



	};



	__host__ __device__ ConvexCell::ConvexCell(int p_seed, float* p_pts) {
		float eps = .1;
		float xmin = -eps; 
		float ymin = -eps;
		float zmin = -eps;
		float xmax = 1000+ eps;
		float ymax = 1000+ eps;
		float zmax = 1000+ eps;
		pts = p_pts;
		first_boundary_ = END_OF_LIST;
		FOR(i, MAX_CLIPS) boundary_next[i] = END_OF_LIST;
		voro_id = p_seed;
		voro_seed = make_float3(pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2]);
		statut = security_ray_not_reached;

		GPU_clip_eq[0] = make_float4(1.0, 0.0, 0.0, -xmin);
		GPU_clip_eq[1] = make_float4(-1.0, 0.0, 0.0, xmax);
		GPU_clip_eq[2] = make_float4(0.0, 1.0, 0.0, -ymin);
		GPU_clip_eq[3] = make_float4(0.0, -1.0, 0.0, ymax);
		GPU_clip_eq[4] = make_float4(0.0, 0.0, 1.0, -zmin);
		GPU_clip_eq[5] = make_float4(0.0, 0.0, -1.0, zmax);
		nb_v = 6;

		tr[0] = make_uchar3(2, 5, 0);		X[0] = make_float4(-xmin, -ymin, -zmax, -1);
		tr[1] = make_uchar3(5, 3, 0);		X[1] = make_float4(-xmin, -ymax, -zmax, -1);
		tr[2] = make_uchar3(1, 5, 2);		X[2] = make_float4(-xmax, -ymin, -zmax, -1);
		tr[3] = make_uchar3(5, 1, 3);		X[3] = make_float4(-xmax, -ymax, -zmax, -1);
		tr[4] = make_uchar3(4, 2, 0);		X[4] = make_float4(-xmin, -ymin, -zmin, -1);
		tr[5] = make_uchar3(4, 0, 3);		X[5] = make_float4(-xmin, -ymax, -zmin, -1);
		tr[6] = make_uchar3(2, 4, 1);		X[6] = make_float4(-xmax, -ymin, -zmin, -1);
		tr[7] = make_uchar3(4, 3, 1);		X[7] = make_float4(-xmax, -ymax, -zmin, -1);
		nb_t=8;
	}


	__host__ __device__ inline float det2x2(float a11, float a12, float a21, float a22) { return a11*a22 - a12*a21; }
	__host__ __device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
		return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
	}
	__host__ __device__ float4 ConvexCell::compute_triangle_point(uchar3 t) const {
		float4 pi1 = GPU_clip_eq[t.x];
		float4 pi2 = GPU_clip_eq[t.y];
		float4 pi3 = GPU_clip_eq[t.z];
		float4 result;
		result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
		result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
		result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
		result.w = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
		return result;
	}





	__host__ __device__ void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
		if (nb_t + 1 == MAX_T) { statut = triangle_overflow; return; }
		tr[nb_t] = make_uchar3(i, j, k);
		X[nb_t] = compute_triangle_point(make_uchar3(i, j, k));
		nb_t++;
	}

	__host__ __device__ int ConvexCell::new_point(int vid) {
		if (nb_v == MAX_CLIPS) { statut = vertex_overflow; return -1; }
		vorother_id[nb_v] = vid;
		B = make_float3(pts[3 * vid], pts[3 * vid + 1], pts[3 * vid + 2]);
		float3 dir = make_float3(voro_seed.x - B.x, voro_seed.y - B.y, voro_seed.z - B.z);
		float3 ave2 = make_float3(voro_seed.x + B.x, voro_seed.y + B.y, voro_seed.z + B.z);
		float dot = ave2.x*dir.x + ave2.y*dir.y + ave2.z*dir.z;
		GPU_clip_eq[nb_v] = make_float4(dir.x, dir.y, dir.z, -dot / 2.);
		nb_v++;
		return nb_v - 1;
	}


	__host__ __device__ void ConvexCell::switch_triangles(uchar t0, uchar t1) {
		uchar3	tmp_id = tr[t0];	tr[t0] = tr[t1];	tr[t1] = tmp_id;
		float4	tmp_X = X[t0];		X[t0] = X[t1];	X[t1] = tmp_X;
	}





	__host__ __device__ void ConvexCell::compute_boundary() {
		// clean circular list of the boundary
		while (first_boundary_ != END_OF_LIST) {
			uchar last = first_boundary_;
			first_boundary_ = boundary_next[first_boundary_];
			boundary_next[last] = END_OF_LIST;
		}

		int nb_iter = 0;
		uchar t = nb_t;
		while (nb_conflicts>0) {
			if (nb_iter++ >50) { statut = weird_cavity; return; }
			bool is_in_border[3];
			bool next_is_opp[3];
			FOR(e, 3)   is_in_border[e] = (boundary_next[ith_plane(t, e)] != END_OF_LIST);
			FOR(e, 3)   next_is_opp[e] = (boundary_next[ith_plane(t, (e + 1) % 3)] == ith_plane(t, e));

			bool can_do_it_now = true;
			// check for non manifoldness
			FOR(e, 3) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) can_do_it_now = false;

			// check for more than one boundary ... or first triangle
			if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
				if (first_boundary_ == END_OF_LIST) {
					FOR(e, 3) boundary_next[ith_plane(t, e)] = ith_plane(t, (e + 1) % 3);
					first_boundary_ = tr[t].x;
				}
				else can_do_it_now = false;
			}

			if (!can_do_it_now) {
				t++;
				if (t == nb_t + nb_conflicts) t = nb_t;
				continue;
			}

			// link next 
			FOR(e, 3)   if (!next_is_opp[e]) boundary_next[ith_plane(t, e)] = ith_plane(t, (e + 1) % 3);

			// destroy link from removed vertices
			FOR(e, 3)  if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
				if (first_boundary_ == ith_plane(t, (e + 1) % 3)) first_boundary_ = boundary_next[ith_plane(t, (e + 1) % 3)];
				boundary_next[ith_plane(t, (e + 1) % 3)] = END_OF_LIST;
			}
			switch_triangles(t,nb_t + nb_conflicts - 1);
			t = nb_t;
			nb_conflicts--; 
		}
#ifndef __CUDA_ARCH__
		gs.add_compute_boundary_iter(nb_iter);
#endif
	}



	__host__ __device__ void  ConvexCell::clip_by_plane(int vid) {

		int cur_v= new_point(vid);
		if (statut == vertex_overflow) return; 
		float4 eqn = GPU_clip_eq[cur_v];
		nb_conflicts = 0;

		float3 diff2seed = make_float3(B.x - voro_seed.x, B.y  - voro_seed.y, B.z - voro_seed.z);
		float d22seed = diff2seed.x*diff2seed.x + diff2seed.y*diff2seed.y + diff2seed.z*diff2seed.z;

		// Step 1: find conflicts 
		int i = 0;
		float dmax2 = 0;
		while (i < nb_t) {
			float4  pc = X[i];
			
			// update security radius
			float3 diff = make_float3( pc.x / pc.w - voro_seed.x,		pc.y / pc.w - voro_seed.y,		pc.z / pc.w - voro_seed.z);
			float d2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z ;
			if (d2 > dmax2) dmax2 = d2;

			// check conflict
			if (eqn.x*pc.x + eqn.y*pc.y + eqn.z*pc.z + eqn.w*pc.w >0){// tr conflict
				switch_triangles(i,nb_t - 1);
				nb_t--;
				nb_conflicts++;
			}
			else i++;
		}

#ifndef __CUDA_ARCH__
		gs.add_clip(nb_conflicts);
#endif


		if (d22seed > 4. * dmax2) { statut = success; return; }
		if (nb_conflicts == 0) {
			nb_v--;
			return;
		}

		// Step 2: compute cavity boundary
		compute_boundary();
		if (statut != security_ray_not_reached) return;
		if (first_boundary_ == END_OF_LIST) return;

		// Step 3: Triangulate cavity 
		uchar cir = first_boundary_;
		do {
			new_triangle(cur_v, cir, boundary_next[cir]);
			if (statut != security_ray_not_reached) return;
			cir = boundary_next[cir];
		} while (cir != first_boundary_);
	}

}


//###################  KERNEL   ######################
__global__ void voro_cell_test_GPU_param(float * pts, int nbpts, unsigned int* neigs, VBW::Statut* gpu_stat, int *out_tets, int* nb_out_tet, float* out_pts){
	int seed = blockIdx.x * blockDim.x + threadIdx.x;
	if (seed >= nbpts) return;

	FOR(d, 3)out_pts[3 * seed + d] = pts[3 * seed + d];
	
	VBW::ConvexCell cc(seed, pts);

	
	FOR(v, DEFAULT_NB_PLANES) {
		cc.clip_by_plane(neigs[DEFAULT_NB_PLANES * seed + v]);
		if (cc.statut == VBW::success) break;
		if (cc.statut != VBW::security_ray_not_reached) {
			gpu_stat[seed] = cc.statut;
			return;
		}
	}
	gpu_stat[seed] = cc.statut;

    /*
	FOR(t, cc.nb_t) {
		if (cc.tr[t].x > 5 && cc.tr[t].y > 5 && cc.tr[t].z > 5) {
			uint4 tet = make_uint4(cc.voro_id,0,0,0);
			tet.y = cc.vorother_id[cc.tr[t].x];
			tet.z = cc.vorother_id[cc.tr[t].y];
			tet.w = cc.vorother_id[cc.tr[t].z];
			int top = atomicAdd(nb_out_tet, 1);
			out_tets[top* 4 ] = cc.voro_id;
			FOR(f, 3) out_tets[top * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f)];
		}
	}
    */
}

float norm2(float3 f) { return f.x*f.x + f.y*f.y + f.z*f.z; }
float dot(float3 A, float3 B) { return A.x*B.x + A.y*B.y + A.z*B.z; }
float3 add(float3 A, float3 B) { return make_float3(A.x + B.x, A.y + B.y, A.z + B.z); }
float3 minus(float3 A, float3 B) { return make_float3(A.x - B.x, A.y - B.y, A.z - B.z); }
float3 mul(float s, float3 A) { return make_float3(s*A.x, s*A.y, s*A.z); }

float3 cross(float3 A, float3 B) { return make_float3(A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x); }
float4 plane_from_point_and_normal(float3 P, float3 n) {return  make_float4(n.x , n.y , n.z,-dot(P,n)); }
void show(float3 v) { std::cerr << v.x << " ,  " << v.y << "  , " << v.z << std::endl; }
float dist2plane(float3 P, float4 plane) { return P.x*plane.x + P.y*plane.y + P.z*plane.z + plane.w; }


void get_tet_volume(float& volume, float3 A, float3 B, float3 C) {
	volume = -VBW::det3x3(A.x, A.y, A.z,		B.x, B.y, B.z,		 C.x, C.y, C.z);
}
void get_tet_volume_and_barycenter(float3& bary, float& volume, float3 A, float3 B, float3 C, float3 D) {
		get_tet_volume(volume, minus(A, D), minus(B, D), minus(C, D));
		bary = make_float3(.25*(A.x + B.x + C.x + D.x), .25*(A.y + B.y + C.y + D.y), .25*(A.z + B.z + C.z + D.z));
}

float3 project_on_plane(float3 P, float4 plane) {
	float3 n = make_float3(plane.x, plane.y, plane.z);
	float lambda;
	lambda= (dot(n, P) + plane.w) / norm2(n);
	return add(P, mul(-lambda, n));
}


void voro_cell_test_CPU_param(std::vector<float>& pts, std::vector<int>& neigs, VBW::Statut* stat, std::vector<int>& out_tets, int* nb_out_tet, std::vector<float>& out_pts, int seed) {
	
	int nbpts = pts.size() / 3; 
	if (seed >= nbpts) return;
	VBW::ConvexCell cc(seed, pts.data());
	FOR(d,3)out_pts[3 * seed + d] = pts[3 * seed + d];


	VBW::gs.start_cell();
	FOR(v, DEFAULT_NB_PLANES) {
		cc.clip_by_plane(neigs[DEFAULT_NB_PLANES * seed + v]);
		if (cc.statut == VBW::success) break;
		if (cc.statut != VBW::security_ray_not_reached) {
			stat[seed] = cc.statut;
			VBW::gs.end_cell();
			VBW::gs.nbv[cc.nb_v]++; VBW::gs.nbt[cc.nb_t]++;
			return;
		}
	}
	VBW::gs.end_cell();
	VBW::gs.nbv[cc.nb_v]++; VBW::gs.nbt[cc.nb_t]++;

	stat[seed] = cc.statut;
	FOR(t, cc.nb_t) {
		if (cc.tr[t].x > 5 && cc.tr[t].y > 5 && cc.tr[t].z > 5) {
			uint4 tet = make_uint4(cc.voro_id, 0, 0, 0);
			tet.y = cc.vorother_id[cc.tr[t].x];
			tet.z = cc.vorother_id[cc.tr[t].y];
			tet.w = cc.vorother_id[cc.tr[t].z];
			
			int top = *nb_out_tet;
			(*nb_out_tet)++;
			out_tets[top * 4] = cc.voro_id;
			FOR(f, 3) out_tets[top * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f)];
		}
	}
	// compute bary
	float3 bary; float vol;
	float3 bary_sum = make_float3(0, 0, 0); float vol_sum = 0;
	FOR(t, cc.nb_t) {
		float3 C = cc.voro_seed;
		float4 A4 = cc.compute_triangle_point(cc.tr[t]);
		float3 A = make_float3(A4.x/A4.w, A4.y / A4.w, A4.z / A4.w);

		float3 Px = project_on_plane(C, cc.GPU_clip_eq[cc.tr[t].x]);
		float3 Py = project_on_plane(C, cc.GPU_clip_eq[cc.tr[t].y]);
		float3 Pz = project_on_plane(C, cc.GPU_clip_eq[cc.tr[t].z]);

		float3 Pxy = project_on_plane(A, plane_from_point_and_normal(C, cross(minus(Px, C), minus(Py, C))));
		float3 Pyz = project_on_plane(A, plane_from_point_and_normal(C, cross(minus(Py, C), minus(Pz, C))));
		float3 Pzx = project_on_plane(A, plane_from_point_and_normal(C, cross(minus(Pz, C), minus(Px, C))));


		get_tet_volume_and_barycenter(bary, vol, Px, Pxy, C, A); bary_sum = add(bary_sum, mul(vol,bary)); vol_sum += vol;
		get_tet_volume_and_barycenter(bary, vol, Pxy, Py, C, A); bary_sum = add(bary_sum, mul(vol, bary)); vol_sum += vol;
		get_tet_volume_and_barycenter(bary, vol, Py, Pyz, C, A); bary_sum = add(bary_sum, mul(vol, bary)); vol_sum += vol;
		get_tet_volume_and_barycenter(bary, vol, Pyz, Pz, C, A); bary_sum = add(bary_sum, mul(vol, bary)); vol_sum += vol;
		get_tet_volume_and_barycenter(bary, vol, Pz, Pzx, C, A); bary_sum = add(bary_sum, mul(vol, bary)); vol_sum += vol;
		get_tet_volume_and_barycenter(bary, vol, Pzx, Px, C, A); bary_sum = add(bary_sum, mul(vol, bary)); vol_sum += vol;

	}
	out_pts[3 * seed] = bary_sum.x / vol_sum;
	out_pts[3 * seed+1] = bary_sum.y / vol_sum;
	out_pts[3 * seed+2] = bary_sum.z / vol_sum;
	
}

template <class T>
struct GPUVar {
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

template <class T>
struct GPUBuffer {
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
void compute_voro_diagram_GPU(std::vector<float>& pts, std::vector<int>& out_tets, int block_size , int &nb_tets, std::vector<VBW::Statut> &stat, std::vector<float>& out_pts) {
    int nbpts = pts.size() / 3;
    kn_problem *kn = NULL;
    {
        Stopwatch W("GPU KNN");
        kn = kn_prepare(pts.data(), nbpts);
        kn_solve(kn);
        kn_print_stats(kn);
    }


    GPUBuffer<float> out_pts_w(out_pts);
//    GPUBuffer<int> tets_w(out_tets);
    GPUBuffer<VBW::Statut> gpu_stat(stat);
    GPUVar<int> gpu_nb_out_tets(nb_tets);
    {
        Stopwatch W("GPU voro kernel only");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        voro_cell_test_GPU_param << < nbpts / block_size + 1, block_size >> > (kn->d_stored_points, nbpts, kn->d_knearests, gpu_stat.gpu_data, NULL/*tets_w.gpu_data*/, gpu_nb_out_tets.gpu_x, out_pts_w.gpu_data);
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
//        tets_w.gpu2cpu();
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
    std::vector<VBW::Statut> stat(nbpts);

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
        std::vector<int> nb_statuts(5, 0);
        FOR(i, stat.size()) nb_statuts[stat[i]]++;
        FOR(r, 5) std::cerr << " " << VBW::StatutStr[r] << "   " << nb_statuts[r] << "\n";
    }

*/
    //return;
    // test block size
    //FOR(iter, 10) 
    int iter = 8; {
        VBW::gs.reset();

        Stopwatch W("GPU run");
        int block_size = pow(2,iter);
        std::cerr << " block_size = "<< block_size << std::endl;
        compute_voro_diagram_GPU(pts, out_tets, block_size , nb_tets, stat, out_pts);

        VBW::gs.show();
        std::cerr << " \n\n\n---------Summary of success/failure------------\n";
        std::vector<int> nb_statuts(5, 0);
        FOR(i, stat.size()) nb_statuts[stat[i]]++;
        FOR(r, 5) std::cerr << " " << VBW::StatutStr[r] << "   " << nb_statuts[r] << "\n";
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

