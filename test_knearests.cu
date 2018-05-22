#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

#define DEFAULT_NB_PLANES 35
#define MAX_CLIPS  41
#define MAX_T  64

#include "VBW.h"
#include "stopwatch.h"



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

bool load_file(const char* filename, std::vector<float>& xyz, bool normalize=true) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
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

    if (normalize) { // normalize point cloud between [0,1000]^3
        float xmin,ymin,zmin,xmax,ymax,zmax;
        get_bbox(xyz, xmin, ymin, zmin, xmax, ymax, zmax);

        float maxside = std::max(std::max(xmax-xmin, ymax-ymin), zmax-zmin);
#pragma omp parallel for
        for (int i=0; i<xyz.size()/3; i++) {
            xyz[i*3+0] = 1000.*(xyz[i*3+0]-xmin)/maxside;
            xyz[i*3+1] = 1000.*(xyz[i*3+1]-ymin)/maxside;
            xyz[i*3+2] = 1000.*(xyz[i*3+2]-zmin)/maxside;
        }
        get_bbox(xyz, xmin, ymin, zmin, xmax, ymax, zmax);
        std::cerr << "bbox [" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << ", [" << zmin << ":" << zmax << "]" << std::endl;
    }
    return true;
}

void export_tet_mesh(float* pts, int nb_pts, int* tets, int nb_tets){
    Stopwatch W("output out.tet");
    std::ofstream out("C:\\DATA\\out.tet");
    out << nb_pts << " vertices" << std::endl;
    out << nb_tets << " tets" << std::endl;
    FOR(v, nb_pts)   out << pts[3 * v] << " " << pts[3 * v + 1] << " " << pts[3 * v + 2] << std::endl;
    FOR(j, nb_tets)  out << "4 " << tets[4 * j] << " " << tets[4 * j + 1] << " " << tets[4 * j + 2] << " " << tets[4 * j + 3] << " \n";
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
    
    std::vector<float> pts;

    if (!load_file(argv[1], pts)) {
        std::cerr << argv[1] << ": could not load file" << std::endl;
        return 1;
    }

/*
    int n=216;
    pts.resize(n*n*n*3);
    for (int x=0; x<n; x++) {
        for (int y=0; y<n; y++) {
            for (int z=0; z<n; z++) {
                float noise[3] = {0.};
                for (int i=0; i<3; i++) {
                    noise[i] = 3.*static_cast<float>(rand())/static_cast<float>(RAND_MAX);
                }
                pts[(x+y*n+z*n*n)*3+0] = x/static_cast<float>(n)*1000. + noise[0];
                pts[(x+y*n+z*n*n)*3+1] = y/static_cast<float>(n)*1000. + noise[1];
                pts[(x+y*n+z*n*n)*3+2] = z/static_cast<float>(n)*1000. + noise[2];
            }
        }
    }
*/

    int nb_pts = pts.size()/3;
    std::vector<int> tets(0);//nb_pts * 4 * 50);
    int nb_tets = 0;

    {
        Stopwatch W("Compute Voro... may test different things cpu/gpu/etc.");
        compute_voro_diagram(pts,  tets, nb_tets);
    }
/*    std::cerr << "nb tets: " << nb_tets << std::endl;

    std::cerr << "EXPORT" << std::endl;
    export_tet_mesh(pts.data(),nb_pts, tets.data(), nb_tets);
*/
    return 0;
}

