#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <set>

#include "knearests.h"
#include "kd_tree.h"

#if defined(__linux__)
#   include <sys/times.h>
#endif

class Stopwatch {
    public:
        Stopwatch(const char* taskname) :
            taskname_(taskname), start_(now()) {
                std::cout << taskname_ << "..." << std::endl;
            }
        ~Stopwatch() {
            double elapsed = now() - start_;
            std::cout << taskname_ << ": "
                << elapsed << "s" << std::endl;
        }
        static double now() {
#if defined(__linux__)
            tms now_tms;
            return double(times(&now_tms)) / 100.0;
#elif defined(WIN32) || defined(_WIN64)
            return double(GetTickCount()) / 1000.0;	    
#else
            return 0.0;
#endif	    
        }
    private:
        const char* taskname_;
        double start_;
};

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

void printDevProp() {
    int devCount; // Number of CUDA devices
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i=0; i<devCount; ++i) {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Major revision number:         %d\n",  devProp.major);
        printf("Minor revision number:         %d\n",  devProp.minor);
        printf("Name:                          %s\n",  devProp.name);
        printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
        printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
        printf("Warp size:                     %d\n",  devProp.warpSize);
        printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
        printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
        printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    }
}

int main(int argc, char** argv) {
    printDevProp();
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " points.xyz" << std::endl;
        return 1;
    }
    
    std::vector<float> points;
    const int DEFAULT_NB_PLANES = 36; // touche pas à ça
    std::vector<int> neighbors;

    { // load point cloud file
        if (!load_file(argv[1], points)) {
            std::cerr << argv[1] << ": could not load file" << std::endl;
            return 1;
        }
    }


    int nb_pts = 1000000;
    points.resize(nb_pts*3);
    for (int i=0; i<points.size(); i++) {
        points[i] = rand()/(float)RAND_MAX;
    }

    { // normalize point cloud between [0,1000]^3
        float xmin,ymin,zmin,xmax,ymax,zmax;
        get_bbox(points, xmin, ymin, zmin, xmax, ymax, zmax);

        float maxside = std::max(std::max(xmax-xmin, ymax-ymin), zmax-zmin);
#pragma omp parallel for
        for (int i=0; i<points.size()/3; i++) {
            points[i*3+0] = 1000.*(points[i*3+0]-xmin)/maxside;
            points[i*3+1] = 1000.*(points[i*3+1]-ymin)/maxside;
            points[i*3+2] = 1000.*(points[i*3+2]-zmin)/maxside;
        }
        get_bbox(points, xmin, ymin, zmin, xmax, ymax, zmax);
        std::cerr << "bbox [" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << ", [" << zmin << ":" << zmax << "]" << std::endl;
    }

    { // solve kn problem
        int nb_points = points.size()/3;
        Stopwatch W("knn gpu");

        kn_problem *kn = kn_prepare(points.data(), nb_points);
        kn_solve(kn);
        kn_print_stats(kn);

        unsigned int *knearests = kn_get_knearests(kn);

        neighbors = std::vector<int>(nb_points*DEFAULT_NB_PLANES);
        unsigned int *permutation = kn_get_permutation(kn);

#pragma omp parallel for
        for (int i=0; i<nb_points; i++) {
            for (int j=0; j<DEFAULT_NB_PLANES; j++) {
                neighbors[permutation[i]*DEFAULT_NB_PLANES+j] = permutation[knearests[i + j*nb_points]];
            }
        }

        if (1) { // sanity check for the permutation array
            std::sort(permutation, permutation+nb_points);
            assert(permutation[0]==0);
            for (int i=0; i<nb_points-1; i++) {
                assert(permutation[i]+1 == permutation[i+1]);
            }
        }
        free(permutation);
        free(knearests);
        kn_free(&kn);
    }

    { // re-check for dupes
        std::cerr << "checking for dupes...";
#pragma omp parallel for
        for (int v=0; v<points.size()/3; v++) {
            std::set<int> kns;
            for (int i=0; i<DEFAULT_NB_PLANES; i++) {
                int kni = neighbors[v*DEFAULT_NB_PLANES+i];
                if (kni < UINT_MAX) {
                    if (kns.find(kni) != kns.end()) {
                        std::cerr << "ERROR: duplicated entry for point " << v << std::endl;
                        break;
                    }
                    kns.insert(kni);
                }
            }
        }
        std::cerr << "ok" << std::endl;
    }
//    return 0;

    std::cerr << "Building KD-tree...";
    int nb_points = points.size()/3;
    std::vector<int> cpu_neighbors(nb_points*DEFAULT_NB_PLANES);
    KdTree KD(3);
    {
        Stopwatch W("knn cpu");
        KD.set_points(nb_points, points.data());
        std::cerr << "ok" << std::endl << "Querying the KD-tree...";

#pragma omp parallel for
        for (int v=0; v<nb_points; ++v) {
            int neigh[DEFAULT_NB_PLANES+1];
            float sq_dist[DEFAULT_NB_PLANES+1];	
            KD.get_nearest_neighbors(DEFAULT_NB_PLANES+1,v,neigh,sq_dist);

            for(int j=0; j<DEFAULT_NB_PLANES; ++j) {
                cpu_neighbors[v*DEFAULT_NB_PLANES+j] = neigh[j+1];
            }
        }
        std::cerr << "ok" << std::endl;
    }
    std::cerr << "Comparing CPU and GPU versions...";
#pragma omp parallel for
    for (int i=0; i<nb_points; i++) {
        std::sort(    neighbors.begin()+i*DEFAULT_NB_PLANES,     neighbors.begin()+(i+1)*DEFAULT_NB_PLANES);
        std::sort(cpu_neighbors.begin()+i*DEFAULT_NB_PLANES, cpu_neighbors.begin()+(i+1)*DEFAULT_NB_PLANES);
    }
#pragma omp parallel for
    for (int i=0; i<nb_points; i++) {
        for (int j=0; j<DEFAULT_NB_PLANES; j++) {
            if (cpu_neighbors[i*DEFAULT_NB_PLANES+j]==neighbors[i*DEFAULT_NB_PLANES+j]) continue;
            std::cerr << "Error in point " << i << " neighbor " << j << std::endl;
            for (int k=0; k<DEFAULT_NB_PLANES; k++) {
                std::cerr << cpu_neighbors[i*DEFAULT_NB_PLANES+k] << "-" << neighbors[i*DEFAULT_NB_PLANES+k] << std::endl;
            }
            assert(false);
        }
    }
    std::cerr << "ok" << std::endl;

    return 0;
}

