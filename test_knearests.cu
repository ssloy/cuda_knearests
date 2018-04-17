#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include "VBW.h"
#include "knearests.h"
#   include <windows.h>
const int DEFAULT_NB_PLANES = 35; // touche pas à ça

inline double now() {return double(GetTickCount()) / 1000.0;}


bool load_file(const char* filename, std::vector<float>& xyz) {
    std::ifstream in;
    in.open (filename, std::ifstream::in);
    if (in.fail()) return false;
    std::string line;
    int npts = 0;
    bool firstline = true;
    float x,y,z;
    while (!in.eof()) {
        std::getline(in, line);
        if (!line.length()) continue;
        std::istringstream iss(line.c_str());
        if (firstline) {
            iss >> npts;
            firstline = false;
        } else {
            iss >> x >> y >> z;
            xyz.push_back(x);
            xyz.push_back(y);
            xyz.push_back(z);
        }
    }
    assert(xyz.size() == npts*3);
    in.close();
    return true;
}

void get_bbox(const std::vector<float>& xyz, float& xmin, float& ymin, float& zmin, float& xmax, float& ymax, float& zmax) {
    int nb_v = xyz.size()/3;
    xmin = xmax = xyz[0];
    ymin = ymax = xyz[1];
    zmin = zmax = xyz[2];
    for(int i=1; i<nb_v; ++i) {
        xmin = std::min(xmin, xyz[3*i]);
        ymin = std::min(ymin, xyz[3*i+1]);
        zmin = std::min(zmin, xyz[3*i+2]);
        xmax = std::max(xmax, xyz[3*i]);
        ymax = std::max(ymax, xyz[3*i+1]);
        zmax = std::max(zmax, xyz[3*i+2]);	    
    }
    float d = xmax-xmin;
    d = std::max(d, ymax-ymin);
    d = std::max(d, zmax-zmin);
    d = 0.001f*d;
    xmin -= d;
    ymin -= d;
    zmin -= d;
    xmax += d;
    ymax += d;
    zmax += d;
}


void voro_cell(int seed, float * pts, int* neigs, float bmin0, float bmin1, float bmin2, float bmax0, float bmax1, float bmax2, int *out_tets) {
	//plop(seed); 
	VBW::ConvexCell cc(seed, pts);
	cc.init_with_box(bmin0, bmin1, bmin2, bmax0, bmax1, bmax2);

	for (int v = 0; v < DEFAULT_NB_PLANES; v++) cc.clip_by_plane(neigs[DEFAULT_NB_PLANES * seed+v]);

	int t = cc.first_valid_;
	int inc = 0;
	while (t != VBW::END_OF_LIST) {
		if (cc.t_[t].i > 5 && cc.t_[t].j > 5 && cc.t_[t].k > 5) {
			out_tets[seed*4 * MAX_T+ inc * 4] = cc.voro_id;
			FOR(f, 3) out_tets[seed * 4 * MAX_T + inc * 4 + f + 1] = cc.vorother_id[cc.ith_plane(t, f) - 6];
			inc++;
		}
		t = int(cc.t_[t].next);
	}

}

int main(int argc, char** argv) {
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " points.xyz" << std::endl;
        return 1;
    }
    
    std::vector<float> points;
    
    std::vector<int> neighbors;
    std::vector<double> watch(1, now());
    { 
        if (!load_file(argv[1], points)) {
            std::cerr << argv[1] << ": could not load file" << std::endl;
            return 1;
        }
	int NBPTS = points.size()/3;
	points.resize(3 * NBPTS);
	FOR(i, 3*NBPTS) points[i] = double(rand()) / RAND_MAX;
    }

    { // normalize point cloud between [0,1000]^3
        float xmin,ymin,zmin,xmax,ymax,zmax;
        get_bbox(points, xmin, ymin, zmin, xmax, ymax, zmax);

        float maxside = std::max(std::max(xmax-xmin, ymax-ymin), zmax-zmin);
        for (int i=0; i<points.size()/3; i++) {
            points[i*3+0] = (points[i*3+0]-xmin)/maxside;
            points[i*3+1] = (points[i*3+1]-ymin)/maxside;
            points[i*3+2] = (points[i*3+2]-zmin)/maxside;
        }
        for (int i=0; i<points.size(); i++) {
            points[i] *= 1000.;
        }
        get_bbox(points, xmin, ymin, zmin, xmax, ymax, zmax);
        std::cerr << "[" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << ", [" << zmin << ":" << zmax << "]" << std::endl;
    }

    std::cerr << "\n----------------------------------------pointset loaded in " << now() - watch.back() << " seconds\n"; watch.push_back(now());

    { // solve kn problem
        neighbors.resize(points.size()/3*DEFAULT_NB_PLANES);
	std::cerr << "\n----------------------------------------memory for neig reserved in " << now() - watch.back() << " seconds\n"; watch.push_back(now());
	kn_problem *kn = kn_prepare(points.data(), points.size()/3);
	std::cerr << "\n----------------------------------------KNN struct prepared  in " << now() - watch.back() << " seconds\n"; watch.push_back(now());
	kn_solve(kn);
	std::cerr << "\n----------------------------------------KNN precomputed in " << now() - watch.back() << " seconds\n"; watch.push_back(now());

        kn_iterator *it = kn_begin_enum(kn); // retrieve neighbors, skip the point itself
        for (int v=0; v<points.size()/3; v++) {
            unsigned int knpt = kn_first_nearest_id(it,v);
            int j = 0;
            while (knpt!=UINT_MAX) {
                if (v!=knpt) {
                    neighbors[v*DEFAULT_NB_PLANES + j] = knpt;
		    j++;
		}
                knpt = kn_next_nearest_id(it);
            }
            assert(j==DEFAULT_NB_PLANES);
        }

        // the data was re-ordered, so retreive it from the GPU
        float *fp = kn_point(it, 0); 
        for (int v=0; v<points.size(); v++) {
            points[v] = fp[v];
        }
        kn_free(&kn);
    }




    int nb_voro_cells =points.size() / 3;
    
    std::vector<int> tets(4*MAX_T* points.size() / 3,-1);
    watch.push_back(now());
    FOR(i, nb_voro_cells) {
	    //if (i%100==0) 
		    voro_cell(i, points.data(), neighbors.data(), 0, 0, 0, 1000, 1000, 1000, tets.data());
    }
    std::cerr << "\n----------------------------------------Voro computed in " << now() - watch.back() << " seconds\n"; watch.push_back(now());

    int nb_real_tets = 0;
    FOR(i, nb_voro_cells) FOR(j, MAX_T) if (tets[i * 4 * MAX_T + 4 * j] != -1
	    && tets[i * 4 * MAX_T + 4 * j] > tets[i * 4 * MAX_T + 4 * j+1] 
	    && tets[i * 4 * MAX_T + 4 * j] > tets[i * 4 * MAX_T + 4 * j + 2]
	    && tets[i * 4 * MAX_T + 4 * j] > tets[i * 4 * MAX_T + 4 * j + 3]
	    )nb_real_tets++;
    


    std::ofstream out("out.tet");
    out << points.size() / 3 << " vertices" << std::endl;
    out << nb_real_tets << " tets" << std::endl;
    FOR(v,points.size()/3)   out << points[3 * v] << " "<< points[3 * v + 1] << " "<< points[3 * v + 2] << std::endl;
    

    FOR(i, nb_voro_cells) {
	    int offset = i * 4 * MAX_T;
	    FOR(j, MAX_T) if (tets[offset + 4 * j] != -1
		    && tets[i * 4 * MAX_T + 4 * j] > tets[i * 4 * MAX_T + 4 * j + 1]
		    && tets[i * 4 * MAX_T + 4 * j] > tets[i * 4 * MAX_T + 4 * j + 2]
		    && tets[i * 4 * MAX_T + 4 * j] > tets[i * 4 * MAX_T + 4 * j + 3]
		    ) {
		    out << "4 "<<tets[offset + 4 * j] << " " << tets[offset + 4 * j + 1] << " " << tets[offset + 4 * j + 2] << " " << tets[offset + 4 * j + 3] << " \n";
	    }
    }

    return 0;
}

