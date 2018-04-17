#include "VBW.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace VBW {

	void ConvexCell::init_with_box(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax) {
		nb_t_ = 0;
		nb_v_ = 0;
		first_free_ = END_OF_LIST;
		first_valid_ = END_OF_LIST;
		first_boundary_ = END_OF_LIST;
		FOR(i, MAX_V) boundary_next[i] = END_OF_LIST;
		float vec_data[6][4] = {
			{1.0, 0.0, 0.0, -xmin },{-1.0, 0.0, 0.0, xmax},
			{0.0, 1.0, 0.0, -ymin },{0.0, -1.0, 0.0, ymax },
			{0.0, 0.0, 1.0, -zmin },{0.0, 0.0, -1.0, zmax }
		};
		FOR(i, 6) bbox_plane_eqn_[i] = make_vec4(vec_data[i][0], vec_data[i][1], vec_data[i][2], vec_data[i][3]);
		uchar tr_data[8][3] = { {2, 5, 0},{5, 3, 0 },{1, 5, 2 },{5, 1, 3 },{4, 2, 0 },{4, 0, 3 },{2, 4, 1 },{4, 3, 1 } };
		FOR(i, 8) new_triangle(tr_data[i][0], tr_data[i][1], tr_data[i][2]);
		nb_v_ = 6;
	}

	void  ConvexCell::clip_by_plane(uint vid) {
		vorother_id[nb_v_ - 6] = vid;
		vec4 eqn = vertex_plane(nb_v_);


		// Step 1: Find conflict zone and link conflicted triangles
		// (to recycle them in free list).

		first_conflict_ = END_OF_LIST;
		// Classify triangles, compute conflict list and valid list.
		uchar t = first_valid_;
		uchar* prev = &first_valid_;
		while (t != END_OF_LIST) {
			if (triangle_is_in_conflict(t_[t], eqn)) {
				// remove t from valid list
				*prev = t_[t].next;
				// add t to conflict list
				if (first_conflict_ == END_OF_LIST)
					t_[t].next = t;
				else{
					t_[t].next = t_[first_conflict_].next;
					t_[first_conflict_].next = t;
				}
				first_conflict_ = t;
				// goto next
				t = *prev;
			} else {
				// update prev and goto next
				prev = &(t_[t].next);
				t = t_[t].next;
			}
		}

		// Special case: no triangle in conflict.
		if (first_conflict_ == END_OF_LIST) return ;
		nb_v_++;


		// Step 2: Triangulate cavity
		compute_boundary();
		uchar cir = first_boundary_;
		do {
			new_triangle(nb_v_ - 1, cir, boundary_next[cir]);
			cir = boundary_next[cir];
		} while (cir != first_boundary_);
	}

	/***********************************************************************/

	bool ConvexCell::triangle_is_in_conflict(
		Triangle T, const vec4& eqn
	) const {
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

	void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
		uchar result = first_free_;
		if (result == END_OF_LIST) {
			result = nb_t_;
			++nb_t_;
		}
		else  first_free_ = t_[first_free_].next;
		t_[result] = make_triangle(i, j, k, first_valid_);
		first_valid_ = result;
	}

	vec4 ConvexCell::vertex_plane(int v) const {
		if (v < 6) return bbox_plane_eqn_[v];
		float dir[3];
		float ave2[3];
		float dot = 0;
		FOR(d, 3) {
			dir[d] = pts[3 * voro_id + d] - pts[3 * vorother_id[v - 6] + d];
			ave2[d] = pts[3 * voro_id + d] + pts[3 * vorother_id[v - 6] + d];
			dot += ave2[d] * dir[d];
		}
		return VBW::make_vec4(dir[0], dir[1], dir[2], -dot / 2.);
	}


	vec4 ConvexCell::compute_triangle_point(int t) const {
		Triangle T = t_[t];
		// Get the plane equations associated with each vertex of t
		vec4 pi1 = vertex_plane(T.i);
		vec4 pi2 = vertex_plane(T.j);
		vec4 pi3 = vertex_plane(T.k);

		vec4 result;
		result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
		result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
		result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
		result.w = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
		return result;
	}



	void ConvexCell::compute_boundary() {
		// clean circular list of the boundary
		while (first_boundary_  != END_OF_LIST) {
			uchar last = first_boundary_;
			first_boundary_ = boundary_next[first_boundary_];
			boundary_next[last] = END_OF_LIST;
		}

		first_boundary_ = END_OF_LIST;
		while (true) {
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
				} else can_do_it_now = false;
			}
			if (!can_do_it_now) {
				first_conflict_ = t_[first_conflict_].next ;
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


}
