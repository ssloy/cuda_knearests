#ifndef VBW_H
#define VBW_H

#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#undef NDEBUG
//#include <windows.h>
//#include <assert.h>

#if defined(__linux__)
#   include <sys/times.h>
#elif defined(WIN32) || defined(_WIN64)
#   include <windows.h>
#endif

inline double now() {
#if defined(__linux__)
    tms now_tms;
    return double(times(&now_tms)) / 100.0;
#elif defined(WIN32) || defined(_WIN64)
    return double(GetTickCount()) / 1000.0;     
#else
    return 0.0;
#endif      
}




static const int MAX_V = 35;
static const int MAX_CLIP_EQ = 6;


static const int MAX_T = 64;

#define check(x)  
//#define check(x)  if (!(x)) __debugbreak();

#define plop(x)  
//#define plop(x) std::cerr << "    ->|plop|<-     "  << #x <<"     "<< x << std::endl

#if defined(__linux__)
#define cuda_check(x) ;
#else
#define cuda_check(x) if (x!=cudaSuccess) __debugbreak();
#endif

#define ALIGNED 
//#define ALIGNED __attribute__((__aligned__(8))) Does not seem to change much.




#define FOR(I,UPPERBND) for(int  I = 0; I<int(UPPERBND); ++I)

namespace VBW {

	struct vec4 { float x; float y; float z; float w; } ALIGNED;
	inline vec4 make_vec4(float x, float y, float z, float w) {
		vec4 result; result.x = x; result.y = y; result.z = z; result.w = w; return result;
	}
	__device__ inline float dot(vec4 v1, vec4 v2) { return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w); }
	__device__ inline float squared_length(vec4 v) { return (v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }
	__device__ inline float length(vec4 v) { return ::sqrt(squared_length(v)); }





	//typedef unsigned int uint;			// global indices.
	typedef unsigned char uchar;		// local indices with special values
	static const  uchar  END_OF_LIST = 255;




	/**
	 * \brief A triangle with the local indices of its three
	 *  vertices.
	 */
	struct Triangle { uchar i; uchar j; uchar k; uchar next; } ALIGNED;
	__device__ inline Triangle make_triangle(uchar i, uchar j, uchar k, uchar f) { Triangle result; result.i = i; result.j = j; result.k = k; result.next = f; return result; }



	/******************************************************************************/

	__device__ inline float det2x2(float a11, float a12, float a21, float a22) {
		return a11*a22 - a12*a21;
	}

	__device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
		return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
	}

	__device__ inline float det4x4(
		float a11, float a12, float a13, float a14,
		float a21, float a22, float a23, float a24,
		float a31, float a32, float a33, float a34,
		float a41, float a42, float a43, float a44
	) {
		float m12 = a21*a12 - a11*a22;
		float m13 = a31*a12 - a11*a32;
		float m14 = a41*a12 - a11*a42;
		float m23 = a31*a22 - a21*a32;
		float m24 = a41*a22 - a21*a42;
		float m34 = a41*a32 - a31*a42;

		float m123 = m23*a13 - m13*a23 + m12*a33;
		float m124 = m24*a13 - m14*a23 + m12*a43;
		float m134 = m34*a13 - m14*a33 + m13*a43;
		float m234 = m34*a23 - m24*a33 + m23*a43;

		return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
	}

	/******************************************************************************/


	    /**
	     * \brief Computes the intersection between a set of halfplanes using
	     *  Bowyer-Watson algorithm.
	     * \details Implementation does not use exact predicates, and does not
	     *  climb from a random vertex. Do not use with a large number of planes.
	     */
	class ConvexCell {
	public:
		__device__ ConvexCell(int p_seed, float* p_pts, int p_nbpts) {
			nbpts = p_nbpts; voro_id = p_seed; pts = p_pts; statut= success;
		}
		void init_with_box(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax);
		void clip_by_plane(int vid);
		vec4 compute_triangle_point(int t) const;
		__device__ inline  uchar& ith_plane(uchar t, int i) { check(t < MAX_T); check(i >= 0); check(i <3); if (i == 0) return t_[t].i; if (i == 1) return t_[t].j; return t_[t].k; }


		enum {
			triangle_overflow=0,
			vertex_overflow=1,
			weird_cavity=2,
			success=3
		} statut;

		/*voronoi vertices stored as triangles */
		uchar nb_t_;					// API --- number of allocated triangles
		Triangle t_[MAX_T];			// API --- memory pool for chained lists of triangles
		uchar first_valid_;				// API --- seed of the actual triangles
		uchar first_free_;				// seed of the free memory  list
		uchar first_conflict_;			// ref to one element of the circular list of triangles that are in conflict

		// --------HOW TO ITERATE OVER TRIANGLES----------
		//int t = first_valid_; while (t != END_OF_LIST) {/*do stuff*/ t = int(t_[t].next);}

		/*clipping planes and pointset*/
		int nbpts;
		float* pts;						// API --- input pointset
		int voro_id;					// API --- id of the seed of the current voro cell 
		uchar nb_v_;					// API --- size of vertices
		int vorother_id[MAX_V];		// API --- vertices ids (to compute clipping planes)  
									//!!!WARNING : offset of 6 due to bbox planes !!!
		vec4 bbox_plane_eqn_[MAX_CLIP_EQ];		// explicit plane equations from the bbox
							
		/*boundary of the last set of triangles that hve been removed*/
		uchar first_boundary_;
		uchar boundary_next[MAX_V+ MAX_CLIP_EQ];

	private:
		bool triangle_is_in_conflict(Triangle T, const vec4& eqn) const;
		void new_triangle(uchar i, uchar j, uchar k);
		void compute_boundary();
		vec4 vertex_plane(int v) const;
	};


	__device__ void ConvexCell::init_with_box(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax) {
		check(xmax - xmin>1e-2);
		check(ymax - ymin>1e-2);
		check(zmax - zmin>1e-2);
		nb_t_ = 0;
		nb_v_ = 0;
		first_free_ = END_OF_LIST;
		first_valid_ = END_OF_LIST;
		first_boundary_ = END_OF_LIST;
		FOR(i, MAX_V + MAX_CLIP_EQ) boundary_next[i] = END_OF_LIST;

		float vec_data[6][4] = {
			{ 1.0, 0.0, 0.0, -xmin },{ -1.0, 0.0, 0.0, xmax },
			{ 0.0, 1.0, 0.0, -ymin },{ 0.0, -1.0, 0.0, ymax },
			{ 0.0, 0.0, 1.0, -zmin },{ 0.0, 0.0, -1.0, zmax }
		};
		FOR(i, 6) bbox_plane_eqn_[i] = make_vec4(vec_data[i][0], vec_data[i][1], vec_data[i][2], vec_data[i][3]);
		uchar tr_data[8][3] = { { 2, 5, 0 },{ 5, 3, 0 },{ 1, 5, 2 },{ 5, 1, 3 },{ 4, 2, 0 },{ 4, 0, 3 },{ 2, 4, 1 },{ 4, 3, 1 } };
		FOR(i, 8) new_triangle(tr_data[i][0], tr_data[i][1], tr_data[i][2]);
		nb_v_ = MAX_CLIP_EQ;
	}


	__device__ bool ConvexCell::triangle_is_in_conflict(Triangle T, const vec4& eqn) const {
		vec4 p1 = vertex_plane(T.i);
		vec4 p2 = vertex_plane(T.j);
		vec4 p3 = vertex_plane(T.k);
		return det4x4(
			p1.x, p2.x, p3.x, eqn.x,
			p1.y, p2.y, p3.y, eqn.y,
			p1.z, p2.z, p3.z, eqn.z,
			p1.w, p2.w, p3.w, eqn.w
		)>0;
	}

	__device__ void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
		check(i<MAX_V + MAX_CLIP_EQ);
		check(j<MAX_V + MAX_CLIP_EQ);
		check(k<MAX_V + MAX_CLIP_EQ);
		uchar result = first_free_;
		if (result == END_OF_LIST) {
			result = nb_t_;
			++nb_t_;
			if (result == MAX_T) { statut = triangle_overflow; return; }
		}
		else  first_free_ = t_[first_free_].next;
		t_[result] = make_triangle(i, j, k, first_valid_);
		first_valid_ = result;
	}

	__device__ vec4 ConvexCell::vertex_plane(int v) const {
		if (v < MAX_CLIP_EQ) return bbox_plane_eqn_[v];
		check(v < MAX_V + MAX_CLIP_EQ);
		float dir[3];
		float ave2[3];
		float dot = 0;
		FOR(d, 3) {
			dir[d] = pts[3 * voro_id + d] - pts[3 * vorother_id[v - MAX_CLIP_EQ] + d];
			ave2[d] = pts[3 * voro_id + d] + pts[3 * vorother_id[v - MAX_CLIP_EQ] + d];
			dot += ave2[d] * dir[d];
		}
		return VBW::make_vec4(dir[0], dir[1], dir[2], -dot / 2.);
	}


	__device__ vec4 ConvexCell::compute_triangle_point(int t) const {
		// Get the plane equations associated with each vertex of t
		vec4 pi1 = vertex_plane(t_[t].i);
		vec4 pi2 = vertex_plane(t_[t].j);
		vec4 pi3 = vertex_plane(t_[t].k);

		vec4 result;
		result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
		result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
		result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
		result.w = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
		return result;
	}




	__device__ void ConvexCell::compute_boundary() {
		// clean circular list of the boundary
		while (first_boundary_ != END_OF_LIST) {
			uchar last = first_boundary_;
			first_boundary_ = boundary_next[first_boundary_];
			boundary_next[last] = END_OF_LIST;
		}

		first_boundary_ = END_OF_LIST;
		int max_iter = 500;
		while (true) {
			if (max_iter-- < 0) { statut = weird_cavity; return; }
			//std::cerr << " \nc=> " << int(first_conflict_);
			uchar t = t_[first_conflict_].next;
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
					first_boundary_ = t_[t].i;
				}
				else can_do_it_now = false;
			}
			if (!can_do_it_now) {
				//std::cerr << " \nconflict=> " << int(first_conflict_);
				//FOR(e, 3)   plop(is_in_border[e]);
				//FOR(e, 3)   plop(next_is_opp[e]);
				first_conflict_ = t_[first_conflict_].next;
				continue;
			}

			// link next 
			FOR(e, 3)   if (!next_is_opp[e]) boundary_next[ith_plane(t, e)] = ith_plane(t, (e + 1) % 3);

			// destroy link from removed vertices
			FOR(e, 3)  if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
				if (first_boundary_ == ith_plane(t, (e + 1) % 3)) first_boundary_ = boundary_next[ith_plane(t, (e + 1) % 3)];
				boundary_next[ith_plane(t, (e + 1) % 3)] = END_OF_LIST;
			}

			// move the current triangle from the conflict list ot the free list
			bool remove_last = (t_[t].next == t);
			if (!remove_last) t_[first_conflict_].next = t_[t].next;
			t_[t].next = first_free_;
			first_free_ = t;
			if (remove_last) return;
		}
	}

	__device__ void  ConvexCell::clip_by_plane(int vid) {
		check(statut == success);
		check(vid<nbpts);
		vorother_id[nb_v_ - MAX_CLIP_EQ] = vid;
		vec4 eqn = vertex_plane(nb_v_);

		// Step 1: Move voronoi vertices from valid list to conflict (circular) list
		first_conflict_ = END_OF_LIST;
		uchar t = first_valid_;
		uchar* prev = &first_valid_;
		while (t != END_OF_LIST) {
			if (triangle_is_in_conflict(t_[t], eqn)) {
				// remove t from valid list
				*prev = t_[t].next;
				// add t to conflict list
				if (first_conflict_ == END_OF_LIST)
					t_[t].next = t;
				else {
					t_[t].next = t_[first_conflict_].next;
					t_[first_conflict_].next = t;
				}
				first_conflict_ = t;
				// goto next
				t = *prev;
			}
			else { // update prev and goto next
				prev = &(t_[t].next);
				t = t_[t].next;
			}
		}

		if (first_conflict_ == END_OF_LIST) return;


		// Step 2: Add new vertex and Triangulate cavity
		nb_v_++;
		if (nb_v_ > MAX_V + MAX_CLIP_EQ) { statut = vertex_overflow; /*std::cerr << "  ----> " << nb_v_;*/ return; }
		compute_boundary();
		if (statut != success) return;
		if (first_boundary_ == END_OF_LIST) return;
		uchar cir = first_boundary_;
		do {
			new_triangle(nb_v_ - 1, cir, boundary_next[cir]);
			if (statut != success) return;
			cir = boundary_next[cir];
		} while (cir != first_boundary_);
	}

}


__global__ void voro_cell_test_GPU_param(float * pts, int nbpts, int* neigs, float bmin0, float bmin1, float bmin2, float bmax0, float bmax1, float bmax2, int *out_tets) {
	int seed = blockIdx.x * blockDim.x + threadIdx.x;
	if (seed >= nbpts) return;
	//out_tets[seed * 4 * MAX_T] = seed;
	//out_tets[seed * 4 * MAX_T+1] = 1;
	//out_tets[seed * 4 * MAX_T+2] = 2;
	//out_tets[seed * 4 * MAX_T+3] = 3;
	VBW::ConvexCell cc(seed, pts, nbpts);
	cc.init_with_box(bmin0, bmin1, bmin2, bmax0, bmax1, bmax2);

	FOR(v, MAX_V) {
		cc.clip_by_plane(neigs[MAX_V * seed + v]);
		if (cc.statut != cc.success) { /*plop(cc.statut)*/; return; }
	}
	int t = cc.first_valid_;
	int inc = 0;
	while (t != VBW::END_OF_LIST) {
		if (cc.t_[t].i > 5 && cc.t_[t].j > 5 && cc.t_[t].k > 5) {
			out_tets[seed * 4 * MAX_T + inc * 4] = cc.voro_id;
			FOR(f, 3) out_tets[seed * 4 * MAX_T + inc * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f) - 6];
			inc++;
		}
		t = int(cc.t_[t].next);
	}
}


//void voro_cell(int seed, float * pts, int nbpts, int* neigs, float bmin0, float bmin1, float bmin2, float bmax0, float bmax1, float bmax2, int *out_tets) {
//	VBW::ConvexCell cc(seed, pts, nbpts);
//	cc.init_with_box(bmin0, bmin1, bmin2, bmax0, bmax1, bmax2);
//
//	FOR(v, MAX_V) {
//		cc.clip_by_plane(neigs[MAX_V * seed + v]);
//		if (cc.statut != cc.success) { plop(cc.statut); return; }
//	}
//	int t = cc.first_valid_;
//	int inc = 0;
//	while (t != VBW::END_OF_LIST) {
//		if (cc.t_[t].i > 5 && cc.t_[t].j > 5 && cc.t_[t].k > 5) {
//			out_tets[seed * 4 * MAX_T + inc * 4] = cc.voro_id;
//			FOR(f, 3) out_tets[seed * 4 * MAX_T + inc * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f) - 6];
//			inc++;
//		}
//		t = int(cc.t_[t].next);
//	}
//}
	void compute_voro_diagram(float * pts, int nbpts, int* neigs, float bmin0, float bmin1, float bmin2, float bmax0, float bmax1, float bmax2, int *out_tets, bool verbose = false) {
		std::cerr << "\n*******copy data to GPU******\n";
		float *gpu_pts;
		int *gpu_knn;
		int *gpu_tets;
		cuda_check(cudaMalloc((void**)& gpu_pts, nbpts * 3 * sizeof(float)));
		cuda_check(cudaMalloc((void**)& gpu_knn, nbpts *MAX_V* sizeof(int)));
		cuda_check(cudaMalloc((void**)& gpu_tets, nbpts * 4 *MAX_T * sizeof(int)));

		cuda_check(cudaMemcpy(gpu_pts, pts, nbpts * 3 * sizeof(float), cudaMemcpyHostToDevice));
		cuda_check(cudaMemcpy(gpu_knn, neigs, nbpts *MAX_V * sizeof(int), cudaMemcpyHostToDevice));
		cuda_check(cudaMemcpy(gpu_tets, out_tets, nbpts *4*MAX_T * sizeof(int), cudaMemcpyHostToDevice));

		double prevt = now();
		voro_cell_test_GPU_param <<< nbpts / 256, 256 >> > (gpu_pts, nbpts, gpu_knn, bmin0, bmin1, bmin2, bmax0, bmax1, bmax2, gpu_tets);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		//FOR(seed, nbpts) {
		//	voro_cell(seed, pts, nbpts, neigs, bmin0, bmin1, bmin2, bmax0, bmax1, bmax2, out_tets);
		//}
		cuda_check(cudaMemcpy(out_tets, gpu_tets, nbpts * 4 *MAX_T * sizeof(int), cudaMemcpyDeviceToHost));
		std::cerr << "\n****************kernel executed in Voro computed in " << now() - prevt << " seconds\n";

		cuda_check(cudaFree(gpu_pts));
		cuda_check(cudaFree(gpu_knn));
		cuda_check(cudaFree(gpu_tets));

	}

#endif
