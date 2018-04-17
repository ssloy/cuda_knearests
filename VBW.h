#ifndef VBW_H
#define VBW_H

#include <string>
#include <vector>
#include <iostream>
#include <cmath>


static const int MAX_V = 35;

static const int MAX_T = 45;



#define plop(x) std::cerr << "    ->|plop|<-     "  << #x <<"     "<< x << std::endl

#define ALIGNED 
//#define ALIGNED __attribute__((__aligned__(8))) Does not seem to change much.




#define FOR(I,UPPERBND) for(int  I = 0; I<int(UPPERBND); ++I)

namespace VBW {

	struct vec4 { float x; float y; float z; float w; } ALIGNED;
	inline vec4 make_vec4(float x, float y, float z, float w) {
		vec4 result; result.x = x; result.y = y; result.z = z; result.w = w; return result;
	}
	inline float dot(vec4 v1, vec4 v2) { return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w); }
	inline float squared_length(vec4 v) { return (v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }
	inline float length(vec4 v) { return ::sqrt(squared_length(v)); }





	typedef unsigned int uint;			// global indices.
	typedef unsigned char uchar;		// local indices with special values
	static const  uchar  END_OF_LIST = 255;




	/**
	 * \brief A triangle with the local indices of its three
	 *  vertices.
	 */
	struct Triangle { uchar i; uchar j; uchar k; uchar next; } ALIGNED;
	inline Triangle make_triangle(uchar i, uchar j, uchar k, uchar f) { Triangle result; result.i = i; result.j = j; result.k = k; result.next = f; return result; }



	/******************************************************************************/

	inline float det2x2(float a11, float a12, float a21, float a22) {
		return a11*a22 - a12*a21;
	}

	inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
		return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
	}

	inline float det4x4(
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
		ConvexCell(uint p_seed, float* p_pts) { voro_id = p_seed; pts = p_pts;}
		void init_with_box(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax);
		void clip_by_plane(uint vid);
		vec4 compute_triangle_point(int t) const;
		inline  uchar& ith_plane(uchar t, int i) { if (i == 0) return t_[t].i; if (i == 1) return t_[t].j; return t_[t].k; }



		/*voronoi vertices stored as triangles */
		uchar nb_t_;					// API --- number of allocated triangles
		Triangle t_[MAX_T];			// API --- memory pool for chained lists of triangles
		uchar first_valid_;				// API --- seed of the actual triangles
		uchar first_free_;				// seed of the free memory  list
		uchar first_conflict_;			// ref to one element of the circular list of triangles that are in conflict

		// --------HOW TO ITERATE OVER TRIANGLES----------
		//int t = first_valid_; while (t != END_OF_LIST) {/*do stuff*/ t = int(t_[t].next);}

		/*clipping planes and pointset*/
		float* pts;					// API --- input pointset
		uint voro_id;					// API --- id of the seed of the current voro cell 
		uchar nb_v_;					// API --- size of vertices
		uint vorother_id[MAX_V];		// API --- vertices ids (to compute clipping planes)  
									//!!!WARNING : offset of 6 due to bbox planes !!!
		vec4 bbox_plane_eqn_[6];		// explicit plane equations from the bbox
							
		/*boundary of the last set of triangles that hve been removed*/
		uchar first_boundary_;
		uchar boundary_next[MAX_V];

	private:
		bool triangle_is_in_conflict(Triangle T, const vec4& eqn) const;
		void new_triangle(uchar i, uchar j, uchar k);
		void compute_boundary();
		vec4 vertex_plane(int v) const;
	};



}

#endif
