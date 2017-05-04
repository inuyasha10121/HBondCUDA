#ifndef CALCULATIONMETHODS_H
#define CALCULATIONMETHODS_H

#include <vector>
#include "kernel.cuh"

cudaDeviceProp setupCUDA(int id);
int getHBondsGPU(vector<GPUAtom> proteindonor, vector<GPUAtom> proteinacceptor, vector<GPUAtom> proteinlinker, vector<GPUAtom> water,
    vector<GPUAtom> & closewaters, vector<vector<int>> & donortowater, vector<vector<int>> & acceptortowater, cudaDeviceProp deviceProp);

int loadTimelineLauncher(char * outGlobalTimeline, int * inTimelineVector, int * inLookupVector, const int currWater, const int numTimeline, const int numLookUp, const int numFrames,
    const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp);
int windowTimelineLauncher(char * ioGlobalTimeline, const int window, const int threshold, const int numFrames, const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp);
int timelineEventAnalysisLauncher(int * outGlobalEventList, char * inGlobalTimeline, const int numFrames, const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp);
int timelineVisitAnalysisLauncher(char * outGlobalVisitList, char * inGlobalTimeline, const int numFrames, const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp);
void pingPongChecker(int & outNumStates, int & outNumStateChanges, int & outNumPingPongs, char * inTimeline, char * inVisitList, const int numFrames, const int numAAs);

int neighborAnalysisLauncher(int * outNearID, float * outNearDist, GPUAtom * inWater, GPUAtom * inProtein, const int numProtein, const int numWater, const float cudaMemPercentage, cudaDeviceProp &deviceProp);
int velBasedAnalysis(char * pdbFile, char * trjFile, char * vellog, char * neighborlog, float veldistcutoff, int dt, float cudaMemPercentage, cudaDeviceProp deviceProp);
#endif