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

cudaDeviceProp setupCUDA()
{
    //Make sure we can use the graphics card (This calculation would be unresonable otherwise)
    if (cudaSetDevice(0) != cudaSuccess) {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
    }

    cudaDeviceProp deviceProp;
    cudaError_t cudaResult;
    cudaResult = cudaGetDeviceProperties(&deviceProp, 0);

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
    //TODO: REMOVED, RE-ENABLE IS MEMORY IS AN ISSUE
    //vector<Atom>().swap(gpuwater);

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

int getHBondsCPU(vector<Atom> proteindonor, vector<Atom> proteinacceptor, vector<Atom> proteinlinker, vector<Atom> water, vector<Atom> & closewaters, vector<vector<int>> & donortowater, vector<vector<int>> & acceptortowater)
{
    //Calculate center of geometry for protein
    float centx = 0.0f;
    float centy = 0.0f;
    float centz = 0.0f;

    for (int i = 0; i < proteindonor.size(); i++)
    {
        centx += proteindonor[i].x;
        centy += proteindonor[i].y;
        centz += proteindonor[i].z;
    }
    for (int i = 0; i < proteinacceptor.size(); i++)
    {
        centx += proteinacceptor[i].x;
        centy += proteinacceptor[i].y;
        centz += proteinacceptor[i].z;
    }
    for (int i = 0; i < proteinlinker.size(); i++)
    {
        centx += proteinlinker[i].x;
        centy += proteinlinker[i].y;
        centz += proteinlinker[i].z;
    }

    centx /= (proteindonor.size() + proteinacceptor.size() + proteinlinker.size());
    centy /= (proteindonor.size() + proteinacceptor.size() + proteinlinker.size());
    centz /= (proteindonor.size() + proteinacceptor.size() + proteinlinker.size());

    //Calculate the filter radius
    float maxdist = 0.0f;
    for (int i = 0; i < proteindonor.size(); i++)
    {
        float dist = distance(centx, centy, centz, proteindonor[i].x, proteindonor[i].y, proteindonor[i].z);
        if (dist > maxdist)
        {
            maxdist = dist;
        }
    }
    for (int i = 0; i < proteinacceptor.size(); i++)
    {
        float dist = distance(centx, centy, centz, proteinacceptor[i].x, proteinacceptor[i].y, proteinacceptor[i].z);
        if (dist > maxdist)
        {
            maxdist = dist;
        }
    }
    for (int i = 0; i < proteinlinker.size(); i++)
    {
        float dist = distance(centx, centy, centz, proteinlinker[i].x, proteinlinker[i].y, proteinlinker[i].z);
        if (dist > maxdist)
        {
            maxdist = dist;
        }
    }

    maxdist += 5.0;

    //Apply filter radius and find new water atoms
    for (int i = 0; i < water.size(); i += 3)
    {
        if (water[i].element != "O")
        {
            printf("Error: Waters are not formatted properly (O, H, H). {%i, %i, %s, %s}", i, water[i].resSeq, water[i].name.c_str(), water[i].element.c_str());
            std::cin.get();
            return 1;
        }

        if (distance(centx, centy, centz, water[i].x, water[i].y, water[i].z) < maxdist ||
            distance(centx, centy, centz, water[i + 1].x, water[i + 1].y, water[i + 1].z) < maxdist ||
            distance(centx, centy, centz, water[i + 2].x, water[i + 2].y, water[i + 2].z) < maxdist)
        {
            closewaters.push_back(water[i]);
            closewaters.push_back(water[i + 1]);
            closewaters.push_back(water[i + 2]);
        }
    }
    vector<Atom>().swap(water);

    //Get list of donor-acceptor heavy atom pairs based on distance and cutoff angle
    int donordistfound = 0;
    for (int p = 0; p < proteindonor.size(); p++)
    {
        for (int w = 0; w < closewaters.size(); w += 3)
        {
            auto dtoa = distance(proteindonor[p], closewaters[w]); //Construct vector from donor to acceptor
            if (magnitude(dtoa) < 3.5f) //Check if vector length is within H-Bond distance
            {
                donordistfound++;
                int closestindex = -1;
                float mindist = FLT_MAX;  //Equivalent to "c"
                for (int l = 0; l < proteinlinker.size(); l++)
                {
                    if (proteindonor[p].resSeq == proteinlinker[l].resSeq) //Check if the hydrogen in question belongs to the same resiude number as the donor
                    {
                        float dist = magnitude(distance(proteinlinker[l], closewaters[w]));
                        if (dist < mindist)
                        {
                            mindist = dist;
                            closestindex = l;
                        }
                    }
                    else if (proteindonor[p].resSeq < proteinlinker[l].resSeq)
                    {
                        break;
                    }
                }
                if (closestindex == -1)
                {
                    std::cin.get();
                    return 1;
                }
                float a = magnitude(distance(proteindonor[p], proteinlinker[closestindex]));
                float b = magnitude(distance(proteindonor[p], closewaters[w]));
                float theta = radtodeg(acosf(((a * a) + (b*b) - (mindist*mindist)) / (2 * a * b)));
                if (theta <= 30.0f && theta >= 0.0f)  //Check if angle criteria is met.  If so, add to list.
                {
                    vector<int> temp;
                    temp.push_back(p);
                    temp.push_back(w);
                    donortowater.push_back(temp);
                }
            }
        }
    }

    int acceptordistfound = 0;
    for (int p = 0; p < proteinacceptor.size(); p++)
    {
        for (int w = 0; w < closewaters.size(); w += 3)
        {
            auto dtoa = distance(closewaters[w], proteinacceptor[p]); //Construct vector from donor to acceptor
            if (magnitude(dtoa) < 3.5f) //Check if vector length is within H-Bond distance
            {
                acceptordistfound++;
                auto dist1 = magnitude(distance(proteinacceptor[p], closewaters[w + 1]));
                auto dist2 = magnitude(distance(proteinacceptor[p], closewaters[w + 2]));
                int closestindex = -1;
                if (dist1 < dist2)
                {
                    closestindex = w + 1;
                }
                else
                {
                    closestindex = w + 2;
                }
                if (closestindex == -1)
                {
                    printf("Error: No valid hydrogen atom found for donor.");
                    std::cin.get();
                    return 1;
                }
                float a = magnitude(distance(closewaters[w], proteinacceptor[p]));
                float b = magnitude(distance(closewaters[w], closewaters[closestindex]));
                float c = magnitude(distance(proteinacceptor[p], closewaters[closestindex]));
                float theta = radtodeg(acosf(((a * a) + (b*b) - (c*c)) / (2 * a * b)));

                if (theta <= 30.0f && theta >= 0.0f)  //Check if angle criteria is met.  If so, add to list.
                {
                    vector<int> temp;
                    temp.push_back(p);
                    temp.push_back(w);
                    acceptortowater.push_back(temp);
                }
            }
        }
    }
    return 0;
}