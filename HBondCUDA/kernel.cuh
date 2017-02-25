#ifndef __KERNEL_H
#define __KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUTypes.h"

using namespace std;

cudaError_t waterFilterCuda(char *out, const GPUAtom *inWaters, const float centx, const float centy, const float centz, const float maxdist, const size_t nWaters, cudaDeviceProp &deviceProp);
cudaError_t bondDistCuda(char *out, const GPUAtom *inProteins, const GPUAtom *inWaters, const size_t nProteins, const size_t nWaters, cudaDeviceProp &deviceProp);
cudaError_t waterToAcceptorCuda(char *out, const GPUAtom *inAcceptor, const GPUAtom *inWater, const size_t nAcceptors, const size_t nWaters, cudaDeviceProp &deviceProp);
cudaError_t donorToWaterCuda(char *out, const GPUAtom *inDonor, const GPUAtom *inLinker, const GPUAtom *inWater, const size_t nDonors, const size_t nLinkers, const size_t nWaters, cudaDeviceProp &deviceProp);

#endif