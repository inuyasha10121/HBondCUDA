#define _USE_MATH_DEFINES
//#define BENCHMARK_TIMING

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <float.h>
#include <algorithm>

#include "kernel.cuh"

using namespace std;

__global__ void waterFilterKernel(char *out, const GPUAtom *inWaters, const float centx, const float centy, const float centz, const float maxdist, const size_t nWaters)
{
    //Find where we are in the GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //Make sure we aren't trying to access outside our pre-definied dimensions
    if (i < nWaters)
    {
        //Get the distance between the water and the center of geometry
        float distx = inWaters[i].x - centx;
        float disty = inWaters[i].y - centy;
        float distz = inWaters[i].z - centz;
        float dist = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
        //Flag if it is within the proper distance or not
        out[i] = (dist < maxdist);
    }
}

__global__ void bondDistKernel(char *out, const GPUAtom *inProtein, const GPUAtom *inWaters, const size_t nProteins, const size_t nWaters)
{
    //Find where we are in the GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //Make sure we aren't trying to access outside our pre-definied dimensions
    if (i < nWaters && j < nProteins)
    {
        //out[(j * nWaters) + i] = 'n';  //Set default to "No bond"
        if (i % 3 == 0) //Only look for oxygen atoms, which should be every third atom starting at atom index 0
        {
            //Get the distance between the heavy atoms
            float distx = inWaters[i].x - inProtein[j].x;
            float disty = inWaters[i].y - inProtein[j].y;
            float distz = inWaters[i].z - inProtein[j].z;
            float dist = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            out[(j * nWaters) + i] = (dist < 3.5f);
        }
    }
}

__global__ void waterToAcceptorKernel(char *out, const GPUAtom *inAcceptor, const GPUAtom *inWater, const size_t nAcceptors, const size_t nWaters)
{
    //Find where we are in the GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //Make sure we aren't trying to access outside our pre-definied dimensions
    if (i < nWaters && j < nAcceptors)
    {
        if (out[(j * nWaters) + i] == true) //Make sure we are in bonding distance from before
        {
            //Find which hydrogen is between the acceptor and the oxygen
            float distx = inWater[i + 1].x - inAcceptor[j].x;
            float disty = inWater[i + 1].y - inAcceptor[j].y;
            float distz = inWater[i + 1].z - inAcceptor[j].z;
            float dist1 = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            distx = inWater[i + 2].x - inAcceptor[j].x;
            disty = inWater[i + 2].y - inAcceptor[j].y;
            distz = inWater[i + 2].z - inAcceptor[j].z;
            float dist2 = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            int closestindex = -1;
            if (dist1 < dist2)
            {
                closestindex = i + 1;
            }
            else
            {
                closestindex = i + 2;
            }
            //Calculate the angle parameters
            distx = inWater[i].x - inAcceptor[j].x;
            disty = inWater[i].y - inAcceptor[j].y;
            distz = inWater[i].z - inAcceptor[j].z;
            float a = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            distx = inWater[i].x - inWater[closestindex].x;
            disty = inWater[i].y - inWater[closestindex].y;
            distz = inWater[i].z - inWater[closestindex].z;
            float b = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            distx = inWater[closestindex].x - inAcceptor[j].x;
            disty = inWater[closestindex].y - inAcceptor[j].y;
            distz = inWater[closestindex].z - inAcceptor[j].z;
            float c = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            float theta = (acosf(((a * a) + (b*b) - (c*c)) / (2 * a * b))) * (180.0f / M_PI);
            if (theta > 30.0f) //If the angle is too large, change the bond to not a bond
            {
                out[(j * nWaters) + i] = false;
            }
        }
    }
}


__global__ void donorToWaterKernel(char *out, const GPUAtom *inDonor, const GPUAtom *inLinker, const GPUAtom *inWater, const size_t nDonors, const size_t nLinkers, const size_t nWaters)
{
    //Find where we are in the GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //Make sure we aren't trying to access outside our pre-definied dimensions
    if (i < nWaters && j < nDonors)
    {
        if (out[(j * nWaters) + i] == true) //Make sure we are in bonding distance from before
        {
            int closestindex = -1;
            float mindist = FLT_MAX;  //Equivalent to "c"
            //Find the bridging linker hydrogen in the residue
            for (int k = 0; k < nLinkers; k++)
            {
                if (inLinker[k].resid == inDonor[j].resid) //Hydrogen belongs to same residue
                {
                    float distx = inLinker[k].x - inWater[i].x;
                    float disty = inLinker[k].y - inWater[i].y;
                    float distz = inLinker[k].z - inWater[i].z;
                    float dist = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
                    if (dist < mindist)
                    {
                        mindist = dist;
                        closestindex = k;
                    }
                    else if (inDonor[j].resid < inLinker[k].resid)
                    {
                        break;
                    }
                }
            }
            //Calculate the angle parameter
            float distx = inLinker[closestindex].x - inDonor[j].x;
            float disty = inLinker[closestindex].y - inDonor[j].y;
            float distz = inLinker[closestindex].z - inDonor[j].z;
            float a = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            distx = inWater[i].x - inDonor[j].x;
            disty = inWater[i].y - inDonor[j].y;
            distz = inWater[i].z - inDonor[j].z;
            float b = sqrtf((distx * distx) + (disty * disty) + (distz * distz));
            float theta = (acosf(((a * a) + (b*b) - (mindist*mindist)) / (2 * a * b))) * (180.0f / M_PI);
            if (theta > 30.0f) //If the angle is too large, change the bond to not a bond
            {
                out[(j * nWaters) + i] = false;
            }
        }
    }
}

cudaError_t waterFilterCuda(char *out, const GPUAtom *inWater, const float centx, const float centy, const float centz, const float maxdist, const size_t nWaters, cudaDeviceProp &deviceProp)
{
    // the device arrays
    GPUAtom *dev_inWater = 0;
    char *dev_out = 0;
    cudaError_t cudaStatus;

    // use div because it's more accurrate than the rounding BS
    auto gridDiv = div(nWaters, deviceProp.maxThreadsPerBlock);
    auto gridY = gridDiv.quot;

    // ass backwards way of rounding up (maybe use the same trick as above? It might be "faster")
    if (gridDiv.rem != 0)
        gridY++;

    // find the block and grid size
    auto blockSize = deviceProp.maxThreadsPerBlock;
    int gridSize = min(16 * deviceProp.multiProcessorCount, gridY);

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_out, nWaters * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inWater, nWaters * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_inWater, inWater, nWaters * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // Launch a kernel on the GPU.
    waterFilterKernel << <gridSize, blockSize >> > (dev_out, dev_inWater, centx, centy, centz, maxdist, nWaters);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "dielectric kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching density kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nWaters * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // delete all our device arrays
Error:
    cudaFree(dev_inWater);
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t bondDistCuda(char *out, const GPUAtom *inProteins, const GPUAtom *inWaters, const size_t nProteins, const size_t nWaters, cudaDeviceProp &deviceProp)
{
    // define device arrays
    GPUAtom *dev_protein = 0;
    GPUAtom *dev_water = 0;
    char *dev_out = 0;
    cudaError_t cudaStatus;

    // Setup the kernel dimensions
    int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
    auto blockSize = dim3(blockDim, blockDim);
    //Waters are chosen for x dimension, since CUDA can handle MUCH more data along the x dimension than y.
    auto gridSize = dim3(round((blockDim - 1 + nWaters) / blockDim), round((blockDim - 1 + nProteins) / blockDim));

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nProteins * nWaters * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_protein, nProteins * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_water, nWaters * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_protein, inProteins, nProteins * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_water, inWaters, nWaters * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // Launch a kernel on the GPU.
    bondDistKernel << <gridSize, blockSize >> > (dev_out, dev_protein, dev_water, nProteins, nWaters);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "Distance kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching hbond distance kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nProteins * nWaters * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_protein);
    cudaFree(dev_water);
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t waterToAcceptorCuda(char *out, const GPUAtom *inAcceptor, const GPUAtom *inWater, const size_t nAcceptors, const size_t nWaters, cudaDeviceProp &deviceProp)
{
    // define device arrays
    GPUAtom *dev_acceptor = 0;
    GPUAtom *dev_water = 0;
    char *dev_out = 0;
    cudaError_t cudaStatus;

    // Setup the kernel dimensions
    int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
    auto blockSize = dim3(blockDim, blockDim);
    //Waters are chosen for x dimension, since CUDA can handle MUCH more data along the x dimension than y.
    auto gridSize = dim3(round((blockDim - 1 + nWaters) / blockDim), round((blockDim - 1 + nAcceptors) / blockDim));

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nAcceptors * nWaters * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_acceptor, nAcceptors * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_water, nWaters * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_out, out, nAcceptors * nWaters * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_acceptor, inAcceptor, nAcceptors * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_water, inWater, nWaters * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // Launch a kernel on the GPU.
    waterToAcceptorKernel << <gridSize, blockSize >> > (dev_out, dev_acceptor, dev_water, nAcceptors, nWaters);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "Donor to water angle kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching donor to water angle kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nAcceptors * nWaters * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_acceptor);
    cudaFree(dev_water);
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t donorToWaterCuda(char *out, const GPUAtom *inDonor, const GPUAtom *inLinker, const GPUAtom *inWater, const size_t nDonors, const size_t nLinkers, const size_t nWaters, cudaDeviceProp &deviceProp)
{
    // define device arrays
    GPUAtom *dev_donor = 0;
    GPUAtom *dev_linker = 0;
    GPUAtom *dev_water = 0;
    char *dev_out = 0;
    cudaError_t cudaStatus;

    // Setup the kernel dimensions
    int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
    auto blockSize = dim3(blockDim, blockDim);
    //Waters are chosen for x dimension, since CUDA can handle MUCH more data along the x dimension than y.
    auto gridSize = dim3(round((blockDim - 1 + nWaters) / blockDim), round((blockDim - 1 + nDonors) / blockDim));

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nDonors * nWaters * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_donor, nDonors * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_linker, nLinkers * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_water, nWaters * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_out, out, nDonors * nWaters * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_donor, inDonor, nDonors * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_linker, inLinker, nLinkers * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_water, inWater, nWaters * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // Launch a kernel on the GPU.
    donorToWaterKernel << <gridSize, blockSize >> > (dev_out, dev_donor, dev_linker, dev_water, nDonors, nLinkers, nWaters);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "Water to acceptor angle kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching water to acceptor angle kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nDonors * nWaters * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_donor);
    cudaFree(dev_linker);
    cudaFree(dev_water);
    cudaFree(dev_out);

    return cudaStatus;
}

//--------------------------------------------------------------------------------------------------REVISION BASED KERNELS--------------------------------------------------------------------------------------------------
__global__ void loadTimelineKernel(char * outTimelineChunk, int * inTimeline, int * inLookUp, const int currWater, const int frameOffset, const int framesToProcess)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < framesToProcess)
    {
        int currFrame = frameOffset + i;
        int endTimeline = inLookUp[currFrame + 1];
        for (int posTimeline = inLookUp[currFrame]; posTimeline < endTimeline; posTimeline += 2)
        {
            if (inTimeline[posTimeline + 1] == currWater)
            {
                outTimelineChunk[(inTimeline[posTimeline] * framesToProcess) + i] = 1;
            }
        }
    }
}

cudaError_t loadTimelineCUDA(char * outGlobalTimeline, int * inTimeline, int * inLookUp, const int currWater, const int numTimeline, const int numLookUp,
    const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess, const int blockSize, const int gridSize, cudaDeviceProp &deviceProp)
{
    // the device arrays
    char * dev_outDeviceTimeline = 0;
    int * dev_inTimeline = 0;
    int * dev_inLookUp = 0;

    cudaError_t cudaStatus;
    
    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_outDeviceTimeline, framesToProcess * numAAs * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_outDeviceTimeline)" << endl;
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_inLookUp, numLookUp * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_inLookUp)" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inTimeline, numTimeline * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_inTimeline)" << endl;
        cout << "numTimeline: " << numTimeline;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_inTimeline, inTimeline, numTimeline * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (dev_inTimeline)" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inLookUp, inLookUp, numLookUp * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (dev_inLookUp)" << endl;
        goto Error;
    }

    //Launch the kernel
    loadTimelineKernel <<<gridSize, blockSize>>> (dev_outDeviceTimeline, dev_inTimeline, dev_inLookUp, currWater, frameOffset, framesToProcess);
 
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "Load timeline kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching load timeline kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy(outGlobalTimeline + (frameOffset * numAAs), dev_outDeviceTimeline, framesPerIter * numAAs * sizeof(char), cudaMemcpyDeviceToHost);
    //cudaStatus = cudaMemcpy2D(outMatrix + (offsetY * outWidth) + offsetX, outWidth, dev_tempMat, inWidth, inWidth, inHeight, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy2D(outGlobalTimeline + frameOffset, numFrames, dev_outDeviceTimeline, framesToProcess, framesToProcess, numAAs, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (outGlobalTimeline)" << endl;
        goto Error;
    }
    // delete all our device arrays
Error:
    cudaFree(dev_outDeviceTimeline);
    cudaFree(dev_inTimeline);
    cudaFree(dev_inLookUp);

    return cudaStatus;
}

void occupancyLoadTimeline(int & minGridSize, int & blockSize, const int calculationsRequested)
{
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, loadTimelineKernel, 0, calculationsRequested);
}

__global__ void windowTimelineKernel(char * inDeviceTimeline, char * outCorrectedTimeline, const int window, const int threshold, const int numAAs, const int framesToProcess)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < framesToProcess)
    {
        for (int currAA = 0; currAA < numAAs; ++currAA)
        {
            int boundFrames = 0;

            for (int currWindow = 0; currWindow < window; ++currWindow)
            {
                if (inDeviceTimeline[(currAA * (framesToProcess + window)) + i + currWindow] == 1)
                {
                    ++boundFrames;
                }
            }
            if (boundFrames >= threshold)
            {
                outCorrectedTimeline[(currAA * framesToProcess) + i] = 1;
            }
            else
            {
                outCorrectedTimeline[(currAA * framesToProcess) + i] = 0;
            }
        }
    }
}

cudaError_t windowTimelineCUDA(char * ioGlobalTimeline, const int window, const int threshold, const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess,
    const int blockSize, const int gridSize, cudaDeviceProp &deviceProp)
{
    // the device arrays
    char * dev_outDeviceTimeline = 0;
    char * dev_inDeviceTimeline = 0;

    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_inDeviceTimeline, (framesToProcess + window) * numAAs * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_inDeviceTimeline)" << endl;
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_outDeviceTimeline, framesToProcess * numAAs * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_outDeviceTimeline)" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    //    cudaStatus = cudaMemcpy2D(outGlobalTimeline + frameOffset, numFrames, dev_outDeviceTimeline, framesPerIter, framesPerIter, numAAs, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy2D(dev_inDeviceTimeline, framesToProcess + window, ioGlobalTimeline + frameOffset, numFrames, framesToProcess + window, numAAs, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (dev_inDeviceTimeline)" << endl;
        goto Error;
    }

    //Launch the kernel
    //__global__ void windowTimelineKernel(char * inDeviceTimeline, char * outCorrectedTimeline, const int window, const int threshold, const int numAAs, const int framesToProcess)
    windowTimelineKernel << <gridSize, blockSize >> > (dev_inDeviceTimeline, dev_outDeviceTimeline, window, threshold, numAAs, framesToProcess);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "Window timeline kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching window timeline kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy(outGlobalTimeline + (frameOffset * numAAs), dev_outDeviceTimeline, framesPerIter * numAAs * sizeof(char), cudaMemcpyDeviceToHost);
    //cudaStatus = cudaMemcpy2D(outMatrix + (offsetY * outWidth) + offsetX, outWidth, dev_tempMat, inWidth, inWidth, inHeight, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy2D(ioGlobalTimeline + frameOffset, numFrames, dev_outDeviceTimeline, framesToProcess, framesToProcess, numAAs, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (ioGlobalTimeline)" << endl;
        goto Error;
    }
    // delete all our device arrays
Error:
    cudaFree(dev_outDeviceTimeline);
    cudaFree(dev_inDeviceTimeline);

    return cudaStatus;
}

void occupancyWindowTimeline(int & minGridSize, int & blockSize, const int calculationsRequested)
{
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, windowTimelineKernel, 0, calculationsRequested);
}

__global__ void timelineEventAnalysisKernel(int * outEventAnalysis, char * inTimeline, const int numAAs, const int framesToProcess)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < framesToProcess)
    {
        int eventsum = 0;
        for (int currAA = 0; currAA < numAAs; ++currAA)
        {
            eventsum += inTimeline[(currAA * framesToProcess) + i];
        }
        outEventAnalysis[i] = eventsum;
    }
}

cudaError_t timelineEventAnalysisCUDA(int * outGlobalEventList, char * inGlobalTimeline, const int numFrames, const int numAAs, const int frameOffset, const int framesToProcess,
    const int blockSize, const int gridSize, cudaDeviceProp &deviceProp)
{
    // the device arrays
    int * dev_outDeviceEventList = 0;
    char * dev_inDeviceTimeline = 0;

    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_inDeviceTimeline, framesToProcess * numAAs * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_inDeviceTimeline)" << endl;
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_outDeviceEventList, framesToProcess * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_outDeviceEventList)" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy2D(dev_inDeviceTimeline, framesToProcess, inGlobalTimeline + frameOffset, numFrames, framesToProcess, numAAs, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (dev_inDeviceTimeline)" << endl;
        goto Error;
    }

    //Launch the kernel
    //__global__ void timelineEventAnalysisKernel(int * outEventAnalysis, char * inTimeline, const int numAAs, const int framesToProcess)
    timelineEventAnalysisKernel << <gridSize, blockSize >> > (dev_outDeviceEventList, dev_inDeviceTimeline, numAAs, framesToProcess);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "Timeline event analysis kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching timeline event analysis kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outGlobalEventList + frameOffset, dev_outDeviceEventList, framesToProcess * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (outGlobalEventList)" << endl;
        goto Error;
    }
    // delete all our device arrays
Error:
    cudaFree(dev_outDeviceEventList);
    cudaFree(dev_inDeviceTimeline);

    return cudaStatus;
}

void occupancyTimelineEventAnalysis(int & minGridSize, int & blockSize, const int calculationsRequested)
{
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, timelineEventAnalysisKernel, 0, calculationsRequested);
}

__global__ void timelineVisitAnalysisKernel(char * outVisitedAAs, char * inDeviceTimeline, const int numFrames, const int AAsToProcess)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < AAsToProcess)
    {
        for (int currFrame = 0; currFrame < numFrames; ++currFrame)
        {
            if (inDeviceTimeline[(i * numFrames) + currFrame] == 1)
            {
                outVisitedAAs[i] = 1;
            }
        }
    }
}

cudaError_t timelineVisitAnalysisCUDA(char * outGlobalVisitList, char * inGlobalTimeline, const int numFrames, const int numAAs, const int AAOffset, const int AAsToProcess,
    const int blockSize, const int gridSize, cudaDeviceProp &deviceProp)
{
    // the device arrays
    char * dev_outDeviceAAList = 0;
    char * dev_inDeviceTimeline = 0;

    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_inDeviceTimeline, numFrames * AAsToProcess * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_inDeviceTimeline)" << endl;
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_outDeviceAAList, AAsToProcess * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed! (dev_outDeviceAAList)" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy2D(dev_inDeviceTimeline, numFrames, inGlobalTimeline + (AAOffset * numFrames), numFrames, numFrames, AAsToProcess, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (dev_inDeviceTimeline)" << endl;
        goto Error;
    }

    //Launch the kernel
    //__global__ void timelineVisitAnalysisKernel(char * outVisitedAAs, char * inDeviceTimeline, const int numFrames, const int AAsToProcess)
    timelineVisitAnalysisKernel << <gridSize, blockSize >> > (dev_outDeviceAAList, dev_inDeviceTimeline, numFrames, AAsToProcess);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "Timeline visit analysis kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching timeline visit analysis kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outGlobalVisitList + AAOffset, dev_outDeviceAAList, AAsToProcess * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed! (outGlobalVisitList)" << endl;
        goto Error;
    }
    // delete all our device arrays
Error:
    cudaFree(dev_outDeviceAAList);
    cudaFree(dev_inDeviceTimeline);

    return cudaStatus;
}

void occupancyTimelineVisitAnalysis(int & minGridSize, int & blockSize, const int calculationsRequested)
{
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, timelineVisitAnalysisKernel, 0, calculationsRequested);
}
