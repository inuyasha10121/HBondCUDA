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
        cerr << "ERROR: Input too large to handle (windowTimelineLauncher)" << endl;
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

int timelineEventAnalysisLauncher(int * outGlobalEventList, char * inGlobalTimeline, const int numFrames, const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp)
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
    cudaFreeMem -= sizeof(int) * numFrames;

                                      //TODO: This is memory inefficient, but I can't think of an equation to properly calculate this.
    size_t memPerCalc = sizeof(char) * numAAs + sizeof(int); //How much memory we need for each thread to do its job

    if (memPerCalc > cudaFreeMem)
    {
        cerr << "ERROR: Input too large to handle (timelineEventAnalysisLauncher)" << endl;
        cerr << "Exitting...";
        return 1;
    }

    int calculationsPossible = cudaFreeMem / memPerCalc;

    auto memIterReq = (int)ceil((float)numFrames / (float)calculationsPossible);  //Number of iterations needed based on memory

                                                                                             //Calculate iterations based on thread parameters
    int blockSize = 0;
    int minGridSize = 0;

    occupancyTimelineEventAnalysis(minGridSize, blockSize, numFrames); //Use occupancy API to calculate the ideal blockSize

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
        cudaResult = timelineEventAnalysisCUDA(outGlobalEventList, inGlobalTimeline, numFrames, numAAs, currIter * framesPerIter, framesToProcess, blockSize, gridSize, deviceProp);
        if (cudaResult != cudaSuccess)
        {
            cerr << "ERROR: Running timeline event analysis launcher failed!" << endl;
            return 1;
        }
    }
    return 0;
}

int timelineVisitAnalysisLauncher(char * outGlobalVisitList, char * inGlobalTimeline, const int numFrames, const int numAAs, const float cudaMemPercentage, cudaDeviceProp &deviceProp)
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
    cudaFreeMem -= sizeof(char) * numAAs;

    //TODO: This is memory inefficient, but I can't think of an equation to properly calculate this.
    size_t memPerCalc = sizeof(char) * numFrames + sizeof(char); //How much memory we need for each thread to do its job

    if (memPerCalc > cudaFreeMem)
    {
        cerr << "ERROR: Input too large to handle (timelineVisitAnalysisLauncher)" << endl;
        cerr << "Exitting...";
        return 1;
    }

    int calculationsPossible = cudaFreeMem / memPerCalc;

    auto memIterReq = (int)ceil((float)numAAs / (float)calculationsPossible);  //Number of iterations needed based on memory

                                                                                  //Calculate iterations based on thread parameters
    int blockSize = 0;
    int minGridSize = 0;

    occupancyTimelineVisitAnalysis(minGridSize, blockSize, numAAs); //Use occupancy API to calculate the ideal blockSize

    int gridsNeeded = (int)ceil((float)numAAs / (float)blockSize); //Calculate how many grids we need to perform the analysis
    int gridIterReq = (int)ceil((float)gridsNeeded / (float)(deviceProp.maxGridSize[1])); //Number of iterations needed based on memory

    int iterReq = max(gridIterReq, memIterReq); //Find out how many iterations we need from both of the calculations above
    int AAsPerIter = numAAs / iterReq; //Calculate how many points we can handle per iteration

                                             //Cycle thourgh until we handle all the points requested
    for (int currIter = 0; currIter < iterReq; ++currIter)
    {
        int AAsToProcess = AAsPerIter;
        if (currIter == (iterReq - 1)) //If we are on the last iteration, we need to make sure we don't go beyond the scope of the input
        {
            AAsToProcess = numAAs - (currIter * AAsPerIter);
        }

        //Calculate the grid parameters for this calculation
        auto gridDiv = div(AAsToProcess, blockSize);
        auto gridSize = gridDiv.quot;
        if (gridDiv.rem != 0)  //Round up if we have straggling frames
        {
            ++gridSize;
        }

        //Run the CUDA code
        cudaResult = timelineVisitAnalysisCUDA(outGlobalVisitList, inGlobalTimeline, numFrames, numAAs, currIter * AAsPerIter, AAsToProcess, blockSize, gridSize, deviceProp);
        if (cudaResult != cudaSuccess)
        {
            cerr << "ERROR: Running timeline visit analysis launcher failed!" << endl;
            return 1;
        }
    }
    return 0;
}

void pingPongChecker(int & outNumStates, int & outNumStateChanges, int & outNumPingPongs, char * inTimeline, char * inVisitList, const int numFrames, const int numAAs)
{
    //Setup some initial storage stuff
    auto frameStates = new ptrdiff_t[numFrames];
    int numStateChanges = 0, numPingPongs = 0;
    //Get the list of visited amino acids in a more compressed format
    vector<int> aaList;
    for (int i = 0; i < numAAs; ++i)
    {
        if (inVisitList[i] == 1)
        {
            aaList.push_back(i);
        }
    }

    //Get a list of unique states the water occupies during the frame
    vector<vector<int>> states;
    vector<int> unbound;
    states.push_back(unbound); //Set unbound to the default (0) state.
    for (int i = 0; i < numFrames; ++i)
    {
        //Get list of currently bound amino acids
        vector<int> currAAbound;
        for (int j = 0; j < aaList.size(); ++j)
        {
            if (inTimeline[(aaList[j] * numFrames) + i] == 1)
            {
                currAAbound.push_back(aaList[j]);
            }
        }
        //Find if this is a currently recorded state or not
        auto pos = find(states.begin(), states.end(), currAAbound);
        if (pos == states.end())
        {
            frameStates[i] = states.size();
            states.push_back(currAAbound); //Add it to the list if not
        }
        else
        {
            frameStates[i] = distance(states.begin(), pos);
        }
        int temp = distance(states.begin(), pos);
         //Save the state to the temporary timeline
    }
    outNumStates = states.size();
    //Go through and see if we ever catch a ping-pong state
    auto prevState = frameStates[0];
    //Start searching
    for (int i = 0; i < numFrames - 1; ++i)
    {
        if (frameStates[i] != prevState && prevState != 0) //Check to see if the state has changed (we don't care about free -> bound transition
        {
            ++numStateChanges;
            auto internalPreviousState = frameStates[i]; //Save what state we are in, so that we can see the next change
            auto foundState = frameStates[i];
            //Do an interior search until we see the states change again
            for (int j = i + 1; j < numFrames; ++j)
            {
                if (frameStates[j] != internalPreviousState) //We found the next state change
                {
                    foundState = frameStates[j];  //Save what state this is
                    break;  //and leave the loop
                }
                internalPreviousState = frameStates[j];
            }
            if (prevState == foundState) //Check if we ping-ponged
            {
                ++numPingPongs;
            }
        }
        prevState = frameStates[i]; //Save what the state is for the next round of edge detection
    }
    outNumStateChanges = numStateChanges;
    outNumPingPongs = numPingPongs;
    delete[] frameStates;
}