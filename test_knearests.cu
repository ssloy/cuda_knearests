#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <set>

#include "knearests.h"

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

int main(int argc, char** argv) {
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " points.xyz" << std::endl;
        return 1;
    }
    
    std::vector<float> points;
    const int DEFAULT_NB_PLANES = 35; // touche pas à ça
    std::vector<int> neighbors;

    { // load point cloud file
        if (!load_file(argv[1], points)) {
            std::cerr << argv[1] << ": could not load file" << std::endl;
            return 1;
        }
        for (int i=0; i<points.size(); i++) {
            points[i] = rand()/(float)RAND_MAX;
        }
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
        std::cerr << "bbox [" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << ", [" << zmin << ":" << zmax << "]" << std::endl;
    }

    { // solve kn problem
        neighbors = std::vector<int>(points.size()/3*DEFAULT_NB_PLANES, -1);
        kn_problem *kn = kn_prepare(points.data(), points.size()/3);
        kn_solve(kn);

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
        kn_print_stats(kn);
        kn_check_for_dupes(kn);

//        kn_sanity_check(kn); // very slow sanity checks

        kn_free(&kn);
    }

    { // re-check for dupes
        for (int v=0; v<points.size()/3; v++) {
            std::set<int> kns;
            for (int i=0; i<DEFAULT_NB_PLANES; i++) {
                int kni = neighbors[v*DEFAULT_NB_PLANES+i];
                if (kni < UINT_MAX) {
                    if (kns.find(kni) != kns.end()) {
                        std::cerr << "ERROR duplicated entry for point " << v << std::endl;
                        return 1;
                    }
                    kns.insert(kni);
                }
            }
        }
    }
    return 0;
}

