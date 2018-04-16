#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

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
    if (!load_file(argv[1], points)) {
        std::cerr << argv[1] << ": could not load file" << std::endl;
        return 1;
    }
    int nb_points = points.size() / 3;
    assert(nb_points*3 == points.size());

    float xmin,ymin,zmin,xmax,ymax,zmax;
    get_bbox(points, xmin, ymin, zmin, xmax, ymax, zmax);

    float maxside = std::max(std::max(xmax-xmin, ymax-ymin), zmax-zmin);
    for (int i=0; i<nb_points; i++) {
        points[i*3+0] = (points[i*3+0]-xmin)/maxside;
        points[i*3+1] = (points[i*3+1]-ymin)/maxside;
        points[i*3+2] = (points[i*3+2]-zmin)/maxside;
    }
    for (int i=0; i<points.size(); i++) {
        points[i] *= 1000.;
    }
    get_bbox(points, xmin, ymin, zmin, xmax, ymax, zmax);
    std::cerr << "[" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << ", [" << zmin << ":" << zmax << "]" << std::endl;



    kn_problem *kn = kn_prepare(points.data(), nb_points);
    kn_solve(kn);

    // iterator, just show the few first points
    kn_iterator *it = kn_begin_enum(kn);
    for (int p = 0; p < min(3,kn_num_points(kn)); p++) {
        float *pt = kn_point(it, p);
        fprintf(stderr, "point %d (%f,%f,%f)\n",p,pt[0],pt[1],pt[2]);
        float *knpt = kn_first_nearest(it,p);
        int k = 0;
        while (knpt) {
            fprintf(stderr, "   knearest [%d] (%f,%f,%f)\n", k, knpt[0], knpt[1], knpt[2]);
            k++;
            knpt = kn_next_nearest(it);
        }
    }

    kn_sanity_check(kn); // very slow sanity checks

    kn_free(&kn);
    return 0;
}

