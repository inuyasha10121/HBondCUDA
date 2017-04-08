#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <float.h>

#include "CalculationMethods.h"
#include "GPUTypes.h"
#include "PDBProcessor.h"

using namespace std;

float distance(float ax, float ay, float az, float bx, float by, float bz)
{
    return sqrtf(((bx - ax) * (bx - ax)) + ((by - ay) * (by - ay)) + ((bz - az) * (bz - az)));
}

float radtodeg(float rad)
{
    return (rad * (180.0f / M_PI));
}

float dotProd(vector<float> & a, vector<float> & b)
{
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

vector<float> crossProd(vector<float> & a, vector<float> & b)
{
    vector<float> result;
    result.push_back((a[1] * b[2]) - (a[2] * b[1]));
    result.push_back((a[2] * b[0]) - (a[0] * b[2]));
    result.push_back((a[0] * b[1]) - (a[1] * b[0]));
    return result;
}

vector<float> distance(Atom & a, Atom & b)
{
    vector<float> result;
    result.push_back(b.x - a.x);
    result.push_back(b.y - a.y);
    result.push_back(b.z - a.z);
    return result;
}

float magnitude(vector<float> & a)
{
    return sqrtf((a[0] * a[0]) + (a[1] * a[1]) + (a[2] * a[2]));
}

vector<float> normalize(vector<float> & a)
{
    vector<float> result;
    float mag = magnitude(a);
    result.push_back(a[0] / mag);
    result.push_back(a[1] / mag);
    result.push_back(a[2] / mag);
    return result;
}

cudaDeviceProp setupCUDA(int id)
{
    //Make sure we can use the graphics card (This calculation would be unresonable otherwise)
    if (cudaSetDevice(id) != cudaSuccess) {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
    }

    cudaDeviceProp deviceProp;
    cudaError_t cudaResult;
    cudaResult = cudaGetDeviceProperties(&deviceProp, id);

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaGetDeviceProperties failed!" << endl;
    }
    return deviceProp;
}

int getHBondsGPU(vector<GPUAtom> gpuproteindonor, vector<GPUAtom> gpuproteinacceptor, vector<GPUAtom> gpuproteinlinker, vector<GPUAtom> gpuwater, vector<GPUAtom> & gpuclosewaters, vector<vector<int>> & donortowater, vector<vector<int>> & acceptortowater, cudaDeviceProp deviceProp)
{
    //Calculate center of geometry for protein
    float centx = 0.0f;
    float centy = 0.0f;
    float centz = 0.0f;

    for (int i = 0; i < gpuproteindonor.size(); i++)
    {
        centx += gpuproteindonor[i].x;
        centy += gpuproteindonor[i].y;
        centz += gpuproteindonor[i].z;
    }
    for (int i = 0; i < gpuproteinacceptor.size(); i++)
    {
        centx += gpuproteinacceptor[i].x;
        centy += gpuproteinacceptor[i].y;
        centz += gpuproteinacceptor[i].z;
    }
    for (int i = 0; i < gpuproteinlinker.size(); i++)
    {
        centx += gpuproteinlinker[i].x;
        centy += gpuproteinlinker[i].y;
        centz += gpuproteinlinker[i].z;
    }

    centx /= (gpuproteindonor.size() + gpuproteinacceptor.size() + gpuproteinlinker.size());
    centy /= (gpuproteindonor.size() + gpuproteinacceptor.size() + gpuproteinlinker.size());
    centz /= (gpuproteindonor.size() + gpuproteinacceptor.size() + gpuproteinlinker.size());

    //Calculate the filter radius
    float maxdist = 0.0f;
    for (int i = 0; i < gpuproteindonor.size(); i++)
    {
        float dist = distance(centx, centy, centz, gpuproteindonor[i].x, gpuproteindonor[i].y, gpuproteindonor[i].z);
        if (dist > maxdist)
        {
            maxdist = dist;
        }
    }
    for (int i = 0; i < gpuproteinacceptor.size(); i++)
    {
        float dist = distance(centx, centy, centz, gpuproteinacceptor[i].x, gpuproteinacceptor[i].y, gpuproteinacceptor[i].z);
        if (dist > maxdist)
        {
            maxdist = dist;
        }
    }
    for (int i = 0; i < gpuproteinlinker.size(); i++)
    {
        float dist = distance(centx, centy, centz, gpuproteinlinker[i].x, gpuproteinlinker[i].y, gpuproteinlinker[i].z);
        if (dist > maxdist)
        {
            maxdist = dist;
        }
    }

    maxdist += 5.0;

    //Apply filter radius and find new water atoms
    char * filterarray = new char[gpuwater.size()];
    waterFilterCuda(filterarray, &gpuwater[0], centx, centy, centz, maxdist, gpuwater.size(), deviceProp);
    for (int i = 0; i < gpuwater.size(); i += 3)
    {
        /* MOVE THIS OUT OF THIS METHOD!
        if (water[i].element != "O")
        {
            printf("Error: Waters are not formatted properly (O, H, H). \n{Index: %i, Residue: %i, Name: %s, Element: %s}\n\n", i, water[i].resSeq, water[i].name.c_str(), water[i].element.c_str());
            printf("POSSIBLE SOLUTION: If the element information is missing, make sure you\ngenerate the .pdb input file by using\n\"gmx editconf -f INPUT.tpr -o OUTPUT.pdb\".\n");
            printf("NOTE: THIS USED TO NOT WORK IN OLDER VERSIONS OF GROMACS.\nMake sure you manually validate thate the element information column\n(Almost last column) is present in the .pdb after attempting the fix.");
            std::cin.get();
            return 1;
        }
        */
        if (filterarray[i] == true || filterarray[i + 1] == true || filterarray[i + 2] == true)
        {
            gpuclosewaters.push_back(gpuwater[i]);
            gpuclosewaters.push_back(gpuwater[i + 1]);
            gpuclosewaters.push_back(gpuwater[i + 2]);
        }
    }
    delete[] filterarray;
    //TODO: PUT IN MEMORY HANDLING CODE HERE.  THIS COULD SERIOUSLY FUCK UP IF THE INPUT IS TOO BIG

    //CUDA code for finding donor -> water pairs
    char * donorarray = new char[gpuclosewaters.size() * gpuproteindonor.size()];
    int donordistfound = 0;
    bondDistCuda(donorarray, &gpuproteindonor[0], &gpuclosewaters[0], gpuproteindonor.size(), gpuclosewaters.size(), deviceProp);

    donorToWaterCuda(donorarray, &gpuproteindonor[0], &gpuproteinlinker[0], &gpuclosewaters[0], gpuproteindonor.size(), gpuproteinlinker.size(), gpuclosewaters.size(), deviceProp);
    for (int i = 0; i < gpuproteindonor.size() * gpuclosewaters.size(); i += 3)
    {
        if (donorarray[i] == true)
        {
            vector<int> temp;
            temp.push_back(gpuproteindonor[i / gpuclosewaters.size()].resid);
            temp.push_back(gpuclosewaters[i % gpuclosewaters.size()].resid);
            donortowater.push_back(temp);
        }
    }
    delete[] donorarray;

    int acceptordistfound = 0;
    char * acceptorarray = new char[gpuclosewaters.size() * gpuproteinacceptor.size()];
    bondDistCuda(acceptorarray, &gpuproteinacceptor[0], &gpuclosewaters[0], gpuproteinacceptor.size(), gpuclosewaters.size(), deviceProp);

    waterToAcceptorCuda(acceptorarray, &gpuproteinacceptor[0], &gpuclosewaters[0], gpuproteinacceptor.size(), gpuclosewaters.size(), deviceProp);
    for (int i = 0; i < gpuproteinacceptor.size() * gpuclosewaters.size(); i += 3)
    {
        if (acceptorarray[i] == true)
        {
            vector<int> temp;
            temp.push_back(gpuproteinacceptor[i / gpuclosewaters.size()].resid);
            temp.push_back(gpuclosewaters[i % gpuclosewaters.size()].resid);
            acceptortowater.push_back(temp);
        }
    }
    delete[] acceptorarray;

    return 0;
}

int performFlatTimelineAnalysis(char * outGlobalMem, vector<int> & inFlatTimeline, vector<int> & inTLLookup, const int window, const int threshold, const int currWater, const int numAAs,
    const int numFrames, const float cudaMemPercentage, cudaDeviceProp &deviceProp)
{
    //Calculate iteration parameters based on memory
    size_t cudaFreeMem;
    cudaError_t cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL); //Get how much memory is available to use

    int calculationsRequired = numFrames * numAAs;

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaMemGetInfo failed!" << endl;
        printf("\nERROR: CUDA is unable to function.  Double check your installation/device settings.");
        printf("\nExiting...");
        return 1;
    }

    cudaFreeMem *= cudaMemPercentage; //Adjust memory based on command line
    cudaFreeMem -= ((sizeof(int) * inFlatTimeline.size()) + (sizeof(int) * inTLLookup.size())); //Reserve space for timeline information
    //cudaFreeMem -= ((sizeof(char) * numAAs) + (sizeof(int) * 2) + (sizeof(int) * numFrames)); //Reserve space for future outputs (visit list, event information, temp mem for event info building)

    size_t memPerCalc = sizeof(char);
    if (memPerCalc > cudaFreeMem)
    {
        cerr << "ERROR: Not enough memory to process one calculation at one time." << endl;
        cerr << "Exitting..." << endl;
        return 1;
    }

    auto gpuFlatTimeline = &inFlatTimeline[0];
    auto gpuTLLookup = &inTLLookup[0];

    int memCalcsPossible = cudaFreeMem / memPerCalc;
    int memIterReq = (int)ceil((double)calculationsRequired / (double)memCalcsPossible);
    //cout << "Iterations required based on memory: " << memIterReq << "(" << cudaFreeMem << "/" << memPerCalc << ")" << endl;

    //Calculate iterations based on gpu thread parameters
    auto gridDiv = div(calculationsRequired, deviceProp.maxThreadsPerBlock);
    auto hypotheticalgridY = gridDiv.quot;
    if (gridDiv.rem != 0)
    {
        ++hypotheticalgridY;
    }
    auto maxPossibleGrid = (16 * deviceProp.multiProcessorCount);

    auto threadIterReq = (int)ceil((double)hypotheticalgridY / (double)maxPossibleGrid);
    //cout << "Iterations required based on threads: " << threadIterReq << endl;

    //Calculate iteration parameters based on information above
    int iterReq = max(threadIterReq, memIterReq);
    int calcsPerIter = (int)ceil((double)calculationsRequired / (double)iterReq);

    /*
    int dummyA = 0, dummyB = 0;
    getTW1DOccupancyFactors(dummyA, dummyB, 0, calculationsRequired);
    cout << "--------------------------------MY CALC--------------------------------" << endl;
    cout << "Block: " << deviceProp.maxThreadsPerBlock << endl;
    cout << "Hypo Grid: " << hypotheticalgridY << endl;
    cout << "Max Size: " << maxPossibleGrid << endl;
    cout << "Waiting..." << endl;
    cin.get();
    */
    
    //Execute the calculation
    //cout << "Iteration " << 0 << " of " << iterReq;
    for (int currIter = 0; currIter < iterReq; ++currIter)
    {
        //cout << "\rIteration " << currIter + 1 << " of " << iterReq;
        //When we are on the last iteration, we will likely need to do less calculations than calcsPerIter
        //To avoid writing to unallowed memory, calculate ACTUALLY how many calcs are needed
        auto calcsToProcess = calcsPerIter;
        if (currIter == (iterReq - 1))
        {
            calcsToProcess = (calculationsRequired - (calcsPerIter * currIter));
        }
        //auto cudaerror = timelineWindowCUDA1D(outGlobalMem, gpuFlatTimeline, gpuTLLookup, 5, 4, 0, numAAs, numFrames, numFrames, numAAs, calcsToProcess, calcsPerIter * currIter, deviceProp);
        auto cudaerror = timelineWindowCUDA1D(outGlobalMem, gpuFlatTimeline, gpuTLLookup, window, threshold, currWater, numAAs, numFrames, inFlatTimeline.size(), inTLLookup.size(), calcsToProcess, calcsPerIter * currIter, deviceProp);
        if (cudaerror != cudaSuccess) {
            cerr << "Timeline window execution failed!" << endl;
            cin.get();
            return 1;
        }

    }
    return 0;
}

int performTimelineEventAnalysis(int & outTotalEvents, int & outFramesBound, char * inTimeline, const int numAAs, const int numFrames, const float cudaMemPercentage, cudaDeviceProp &deviceProp)
{
    //Calculate iteration parameters based on memory
    size_t cudaFreeMem;
    cudaError_t cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL); //Get how much memory is available to use

    int calculationsRequired = numFrames;

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaMemGetInfo failed!" << endl;
        printf("\nERROR: CUDA is unable to function.  Double check your installation/device settings.");
        printf("\nExiting...");
        return 1;
    }

    cudaFreeMem *= cudaMemPercentage; //Adjust memory based on command line
    //cudaFreeMem -= (sizeof(char) * numAAs * numFrames); //Reserve space for timeline information
    //cudaFreeMem -= (sizeof(int) * numFrames); //Reserve space for output

    size_t memPerCalc = sizeof(int) + (sizeof(char) * numAAs);  //Memory for sum, memory for current AA strip
    if (memPerCalc > cudaFreeMem)
    {
        cerr << "ERROR: Not enough memory to process one calculation at one time." << endl;
        cerr << "Exitting..." << endl;
        return 1;
    }

    int memCalcsPossible = cudaFreeMem / memPerCalc;
    int memIterReq = (int)ceil((double)calculationsRequired / (double)memCalcsPossible);
    //cout << "Iterations required based on memory: " << memIterReq << endl;


    //Calculate iterations based on gpu thread parameters
    auto gridDiv = div(calculationsRequired, deviceProp.maxThreadsPerBlock);
    auto hypotheticalgridY = gridDiv.quot;
    if (gridDiv.rem != 0)
    {
        ++hypotheticalgridY;
    }
    auto maxPossibleGrid = (16 * deviceProp.multiProcessorCount);

    auto threadIterReq = (int)ceil((double)hypotheticalgridY / (double)maxPossibleGrid);
    //cout << "Iterations required based on threads: " << threadIterReq << endl;

    //Calculate iteration parameters based on information above
    int iterReq = max(threadIterReq, memIterReq);
    int calcsPerIter = (int)ceil((double)calculationsRequired / (double)iterReq);

    //Execute the calculation
    auto outGlobalMem = new int[numFrames];

    //cout << "Iteration " << 0 << " of " << iterReq;
    for (int currIter = 0; currIter < iterReq; ++currIter)
    {
        //cout << "\rIteration " << currIter + 1 << " of " << iterReq;
        //When we are on the last iteration, we will likely need to do less calculations than calcsPerIter
        //To avoid writing to unallowed memory, calculate ACTUALLY how many calcs are needed
        auto calcsToProcess = calcsPerIter;
        if (currIter == (iterReq - 1))
        {
            calcsToProcess = (calculationsRequired - (calcsPerIter * currIter));
        }
        auto cudaerror = eventListCUDA1D(outGlobalMem, inTimeline, numAAs, calcsToProcess, calcsPerIter * currIter, deviceProp);
        if (cudaerror != cudaSuccess) {
            cerr << "Timeline event analysis execution failed!" << endl;
            cin.get();
            return 1;
        }

    }
    
    //Process the output from the gpu iterations
    int events = 0;
    int framesbound = 0;
    for (int i = 0; i < numFrames; ++i)
    {
        events += outGlobalMem[i];
        framesbound += (outGlobalMem[i] > 0);
    }

    //Output the values
    outTotalEvents = events;
    outFramesBound = framesbound;

    return 0;
}


//--------------------------------------------------------------------------------------------------REVISION BASED LAUNCHERS--------------------------------------------------------------------------------------------------

int loadTimelineLauncher(char * outGlobalTimeline, int * inTimelineVector, int * inLookupVector, const int currWater, const int numTimeline, const int numLookUp, const int numFrames,
    const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp)
{
    //Calculate iteratons based on memory parameters
    size_t cudaFreeMem;
    cudaError_t cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL); //Get how much memory is available to use

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaMemGetInfo failed!" << endl;
        printf("\nERROR: CUDA is unable to function.  Double check your installation/device settings.");
        printf("\nExiting...");
        return 1;
    }

    cudaFreeMem *= cudaMemPercentage; //Shrink the available memory based on how much memory we specified is available
    cudaFreeMem -= (sizeof(int) * (numTimeline + numLookUp)); //Reserve space for the input timeline information
    size_t memPerCalc = sizeof(char) * numAAs; //How much memory we need for each thread to do its job
    if (memPerCalc > cudaFreeMem)
    {
        cerr << "ERROR: Input too large to handle (loadTimelineLauncher)" << endl;
        cerr << "Exitting...";
        return 1;
    }

    int calculationsPossible = cudaFreeMem / memPerCalc;

    auto memIterReq = (int)ceil((float)numFrames / (float)calculationsPossible);  //Number of iterations needed based on memory

    //Calculate iterations based on thread parameters
    int blockSize = 0;
    int minGridSize = 0;

    occupancyLoadTimeline(minGridSize, blockSize, numFrames); //Use occupancy API to calculate the ideal blockSize
    
    int gridsNeeded = (int)ceil((float)numFrames / (float)blockSize); //Calculate how many grids we need to perform the analysis
    int gridIterReq = (int)ceil((float)gridsNeeded / (float)(deviceProp.maxGridSize[1])); //Number of iterations needed based on memory
    
    int iterReq = max(gridIterReq, memIterReq); //Find out how many iterations we need from both of the calculations above
    int framesPerIter = numFrames / iterReq; //Calculate how many points we can handle per iteration

    //Cycle thourgh until we handle all the points requested
    for (int currIter = 0; currIter < iterReq; ++currIter)
    {
        int framesToProcess = framesPerIter;
        if (currIter == (iterReq - 1)) //If we are on the last iteration, we need to make sure we don't go beyond the scope of the input
        {
            framesToProcess = numFrames - (currIter * framesPerIter);
        }

        //Calculate the grid parameters for this calculation
        auto gridDiv = div(framesToProcess, blockSize);
        auto gridSize = gridDiv.quot;
        if (gridDiv.rem != 0)  //Round up if we have straggling frames
        {
            ++gridSize;
        }

        //Run the CUDA code
        cudaResult = loadTimelineCUDA(outGlobalTimeline, inTimelineVector, inLookupVector, currWater, numTimeline, numLookUp, numFrames, numAAs, currIter * framesPerIter, framesPerIter, blockSize, gridSize, deviceProp);
        if (cudaResult != cudaSuccess)
        {
            cerr << "ERROR: Running load timeline launcher failed!" << endl;
            return 1;
        }
    }
    return 0;
}

int windowTimelineLauncher(char * ioGlobalTimeline, const int window, const int threshold, const int numFrames, const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp)
{
    //Calculate iteratons based on memory parameters
    size_t cudaFreeMem;
    cudaError_t cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL); //Get how much memory is available to use

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaMemGetInfo failed!" << endl;
        printf("\nERROR: CUDA is unable to function.  Double check your installation/device settings.");
        printf("\nExiting...");
        return 1;
    }

    cudaFreeMem *= cudaMemPercentage; //Shrink the available memory based on how much memory we specified is available

    //TODO: This is memory inefficient, but I can't think of an equation to properly calculate this.
    size_t memPerCalc = sizeof(char) * numAAs * 2 * window; //How much memory we need for each thread to do its job

    if (memPerCalc > cudaFreeMem)
    {
        cerr << "ERROR: Input too large to handle (loadTimelineLauncher)" << endl;
        cerr << "Exitting...";
        return 1;
    }

    int calculationsPossible = cudaFreeMem / memPerCalc;

    auto memIterReq = (int)ceil((float)(numFrames - window) / (float)calculationsPossible);  //Number of iterations needed based on memory

    //Calculate iterations based on thread parameters
    int blockSize = 0;
    int minGridSize = 0;

    occupancyWindowTimeline(minGridSize, blockSize, (numFrames - window)); //Use occupancy API to calculate the ideal blockSize

    int gridsNeeded = (int)ceil((float)(numFrames - window) / (float)blockSize); //Calculate how many grids we need to perform the analysis
    int gridIterReq = (int)ceil((float)gridsNeeded / (float)(deviceProp.maxGridSize[1])); //Number of iterations needed based on memory

    int iterReq = max(gridIterReq, memIterReq); //Find out how many iterations we need from both of the calculations above
    iterReq += 4;
    int framesPerIter = (numFrames - window) / iterReq; //Calculate how many points we can handle per iteration

    //Cycle thourgh until we handle all the points requested
    for (int currIter = 0; currIter < iterReq; ++currIter)
    {
        int framesToProcess = framesPerIter;
        if (currIter == (iterReq - 1)) //If we are on the last iteration, we need to make sure we don't go beyond the scope of the input
        {
            framesToProcess = (numFrames - window) - (currIter * framesPerIter);
        }

        //Calculate the grid parameters for this calculation
        auto gridDiv = div(framesToProcess, blockSize);
        auto gridSize = gridDiv.quot;
        if (gridDiv.rem != 0)  //Round up if we have straggling frames
        {
            ++gridSize;
        }

        //Run the CUDA code
        cudaResult = windowTimelineCUDA(ioGlobalTimeline, window, threshold, numFrames, numAAs, currIter * framesPerIter, framesToProcess, blockSize, gridSize, deviceProp);
        if (cudaResult != cudaSuccess)
        {
            cerr << "ERROR: Running load timeline launcher failed!" << endl;
            return 1;
        }
    }
    return 0;
}