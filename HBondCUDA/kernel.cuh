#ifndef __KERNEL_H
#define __KERNEL_H

#include "device_launch_parameters.h"
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include "cuda_runtime.h"

#include "GPUTypes.h"


using namespace std;

cudaError_t waterFilterCuda(char *out, const GPUAtom *inWaters, const float centx, const float centy, const float centz, const float maxdist, const size_t nWaters, cudaDeviceProp &deviceProp);
cudaError_t bondDistCuda(char *out, const GPUAtom *inProteins, const GPUAtom *inWaters, const size_t nProteins, const size_t nWaters, cudaDeviceProp &deviceProp);
cudaError_t waterToAcceptorCuda(char *out, const GPUAtom *inAcceptor, const GPUAtom *inWater, const size_t nAcceptors, const size_t nWaters, cudaDeviceProp &deviceProp);
cudaError_t donorToWaterCuda(char *out, const GPUAtom *inDonor, const GPUAtom *inLinker, const GPUAtom *inWater, const size_t nDonors, const size_t nLinkers, const size_t nWaters, cudaDeviceProp &deviceProp);
cudaError_t timelineMapCuda2D(char * outMap, const int * timeline, const int * tllookup, const int * boundAAs, const int * boundwaters, const int window, const int threshold,
    const int ntimeline, const int nframes, const int nAAs, const int nwaters, cudaDeviceProp &deviceProp);
cudaError_t timelineMapCuda1D(char * outMap, const int * timeline, const int * tllookup, const int * boundAAs, const int * boundwaters, const int window, const int threshold,
    const int ntimeline, const int nframes, const int nAAs, const int nwaters, cudaDeviceProp &deviceProp);
cudaError_t visitAndBridgerAnalysisCuda(char * outbridger, char * outvisitlist, int * outframesbound, const char * timelinemap, const int nframes, const int nAAs, cudaDeviceProp &deviceProp);

#endif