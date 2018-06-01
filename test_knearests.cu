#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>



#include "params.h"
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
            xyz[i*3+0] = 1000.f*(xyz[i*3+0]-xmin)/maxside;
            xyz[i*3+1] = 1000.f*(xyz[i*3+1]-ymin)/maxside;
            xyz[i*3+2] = 1000.f*(xyz[i*3+2]-zmin)/maxside;
        }
        get_bbox(xyz, xmin, ymin, zmin, xmax, ymax, zmax);
        std::cerr << "bbox [" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << "], [" << zmin << ":" << zmax << "]" << std::endl;
    }
    return true;
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



void compute_voro_diagram(std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary, int nb_Lloyd_iter,bool GPU=true) {
    if (GPU) compute_voro_diagram_GPU(pts, stat, bary, nb_Lloyd_iter);
    else compute_voro_diagram_CPU(pts, stat, bary, nb_Lloyd_iter);
}


int main(int argc, char** argv) {
    printDevProp();
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " points.xyz" << std::endl;
        return 1;
    }
    int *initptr = NULL;
    cudaError_t err = cudaMalloc(&initptr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        return 1;
    }

    std::vector<float> pts;

    if (!load_file(argv[1], pts, false)) {
        std::cerr << argv[1] << ": could not load file" << std::endl;
        return 1;
    }

    
    //pts.resize(9000);
    //FOR(i, pts.size()) pts[i] = 1.+998.*double(rand()) / double(RAND_MAX);
    
    //FOR(i, pts.size() ) {
    //    if (i % 3 != 0) continue;
    //    float x = pts[ i] / 1000.;
    //    pts[i] = 1000.*x*x; continue;
    //    x = -1 + 2.*x;
    //    x = 5.*pow(x, 3.) - 3.*pow(x, 5.) +2*x;
    //    //pts[3 * i] = 1000.*pow(pts[3 * i] / 1000., 2);
    //    pts[i] = 1000.*(x -4.)/8.;
    //}


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
    std::vector<float> bary(pts.size(), 0);
    std::vector<Status> stat(nb_pts);
   


    bool run_on_GPU = true;
    {
        FOR(i, 3) { // to recompute the knn
        	Stopwatch W(" Lloyd");
//            std::cerr << "Lloyd #" << i << std::endl;
            compute_voro_diagram(pts, stat, bary, 1, run_on_GPU);
        }
//        if (run_on_GPU) drop_xyz_file(pts);
    }

    cudaFree(initptr);
    return 0;
}

 
