//  PFNMR - Estimate FNMR for proteins (to be updated)
//      Copyright(C) 2016 Jonathan Ellis and Bryan Gantt
//
//  This program is free software : you can redistribute it and / or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation, either version 3 of the License.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program.If not, see <http://www.gnu.org/licenses/>.

// More than just GPU types, but still
// structs are good because we don't need any class nonsense

#ifndef __GPUTYPES_H
#define __GPUTYPES_H

#include <string>
#include <cmath>

using namespace std;

typedef struct _GPUAtom {
    float x;
    float y;
    float z;
    short resid;
} GPUAtom;

typedef struct _Atom {
    int serial;
    string name;
    char altLoc;
    string resName;
    char chainID;
    int resSeq;
    char iCode;
    float x;
    float y;
    float z;
    float occupancy;
    float tempFactor;
    string element;
    string charge;
    float vdw;
    char hbondType;
    int hbondListPos;

    GPUAtom getGPUAtom()
    {
        GPUAtom temp;
        temp.x = x;
        temp.y = y;
        temp.z = z;
        temp.resid = resSeq;
        return temp;
    }
} Atom;

typedef struct _GridPoint {
    float x;
    float y;
    float z;
    float dielectric = -1.0f;
} GridPoint;

typedef struct _GPUChargeAtom : _GPUAtom {
    int resid;
    int chainid;
    float charge = 0.0f;
} GPUChargeAtom;

typedef struct {
    int resid;
    int chainid;
    float x;
    float y;
    float z;
    float fieldx = 0.0f;
    float fieldy = 0.0f;
    float fieldz = 0.0f;

    float getTotalField()
    {
        return sqrtf((fieldx * fieldx) + (fieldy * fieldy) + (fieldz * fieldz));
    }
} GPUEFP;
#endif // !__GPUTYPES_H