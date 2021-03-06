#ifndef CALCULAIONMETHODS_H
#define CALCULAIONMETHODS_H

#include <vector>
#include "kernel.cuh"

cudaDeviceProp setupCUDA();
int getHBondsGPU(vector<GPUAtom> proteindonor, vector<GPUAtom> proteinacceptor, vector<GPUAtom> proteinlinker, vector<GPUAtom> water,
    vector<GPUAtom> & closewaters, vector<vector<int>> & donortowater, vector<vector<int>> & acceptortowater, cudaDeviceProp deviceProp);

#endif