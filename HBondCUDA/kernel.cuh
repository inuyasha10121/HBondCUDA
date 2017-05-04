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

cudaError_t loadTimelineCUDA(char * outGlobalTimeline, int * inTimeline, int * inLookUp, const int currWater, const int numTimeline, const int numLookUp,
    const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess, const int blockSize, const int gridSize, cudaDeviceProp &deviceProp);
void occupancyLoadTimeline(int & minGridSize, int & blockSize, const int calculationsRequested);

cudaError_t windowTimelineCUDA(char * ioGlobalTimeline, const int window, const int threshold, const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess,
    const int blockSize, const int gridSize, cudaDeviceProp &deviceProp);
void occupancyWindowTimeline(int & minGridSize, int & blockSize, const int calculationsRequested);

cudaError_t timelineEventAnalysisCUDA(int * outGlobalEventList, char * inGlobalTimeline, const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess,
    const int blockSize, const int gridSize, cudaDeviceProp &deviceProp);
void occupancyTimelineEventAnalysis(int & minGridSize, int & blockSize, const int calculationsRequested);

cudaError_t timelineVisitAnalysisCUDA(char * outGlobalVisitList, char * inGlobalTimeline, const int numFrames, const int numAAs, const int AAOffset, const int AAsToProcess,
    const int blockSize, const int gridSize, cudaDeviceProp &deviceProp);
void occupancyTimelineVisitAnalysis(int & minGridSize, int & blockSize, const int calculationsRequested);

cudaError_t neighborAnalysisCUDA(int * outNearID, float * outNearDist, GPUAtom * inWater, GPUAtom * inProtein, const int numProtein, const int waterToProcess,
	const int waterOffset, const int blockSize, const int gridSize, cudaDeviceProp &deviceProp);
void occupancyNeighborAnalysis(int & minGridSize, int & blockSize, const int calculationsRequested);

#endif