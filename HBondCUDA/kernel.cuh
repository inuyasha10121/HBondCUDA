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

cudaError_t timelineWindowCUDA(char * out, int * inFlatTimeline, int * inTLLookup, const int window, const int threshold, const int currWater, const int numAAs,
    const int numFrames, const int numTimeline, const int numTLLookup, cudaDeviceProp &deviceProp);
cudaError_t visitListCUDA(char * outVisitList, char * inTimeline, const int numAAs, const int numFrames, cudaDeviceProp &deviceProp);
cudaError_t eventListCUDA(int * outTempEventList, char * inTimeline, const int numAAs, const int numFrames, cudaDeviceProp &deviceProp);

cudaError_t timelineWindowCUDA1D(char * outGlobalMem, int * inFlatTimeline, int * inTLLookup, const int window, const int threshold, const int currWater, const int numAAs,
    const int numFrames, const int numTimeline, const int numTLLookup, const int calcsToProcess, const int calcOffset, cudaDeviceProp &deviceProp);
cudaError_t eventListCUDA1D(int * outGlobalMem, char * inTimeline, const int numAAs, const int calcsToProcess, const int calcOffset, cudaDeviceProp &deviceProp);

cudaError_t loadTimelineCUDA(char * outGlobalTimeline, int * inTimeline, int * inLookUp, const int currWater, const int numTimeline, const int numLookUp,
    const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess, const int blockSize, const int gridSize, cudaDeviceProp &deviceProp);
void occupancyLoadTimeline(int & minGridSize, int & blockSize, const int calculationsRequested);

cudaError_t windowTimelineCUDA(char * ioGlobalTimeline, const int window, const int threshold, const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess,
    const int blockSize, const int gridSize, cudaDeviceProp &deviceProp);
void occupancyWindowTimeline(int & minGridSize, int & blockSize, const int calculationsRequested);

#endif