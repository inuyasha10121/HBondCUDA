#ifndef CALCULAIONMETHODS_H
#define CALCULAIONMETHODS_H

#include <vector>
#include "kernel.cuh"

cudaDeviceProp setupCUDA(int id);
int getHBondsGPU(vector<GPUAtom> proteindonor, vector<GPUAtom> proteinacceptor, vector<GPUAtom> proteinlinker, vector<GPUAtom> water,
    vector<GPUAtom> & closewaters, vector<vector<int>> & donortowater, vector<vector<int>> & acceptortowater, cudaDeviceProp deviceProp);
int performFlatTimelineAnalysis(char * outGlobalMem, vector<int> & inFlatTimeline, vector<int> & inTLLookup, const int window, const int threshold, const int currWater, const int numAAs,
    const int numFrames, const float cudaMemPercentage, cudaDeviceProp &deviceProp);
int performTimelineEventAnalysis(int & outTotalEvents, int & outFramesBound, char * inTimeline, const int numAAs, const int numFrames, const float cudaMemPercentage, cudaDeviceProp &deviceProp);

int loadTimelineLauncher(char * outGlobalTimeline, int * inTimelineVector, int * inLookupVector, const int currWater, const int numTimeline, const int numLookUp, const int numFrames,
    const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp);

#endif