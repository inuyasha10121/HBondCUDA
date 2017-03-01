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

// C++ code for reading and processing PDB files

#ifndef PDBPROCESSOR_H
#define PDBPROCESSOR_H

#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>

#include "GPUTypes.h"

using namespace std;

class PDBProcessor {

private:
    string pdbPath;
    ifstream pdbStream;
    bool isOpen = false;

public:
    PDBProcessor(string pdbPath);
    PDBProcessor();
    ~PDBProcessor();

    bool is_open() { return isOpen; }

    vector<Atom> getAtomsFromPDB();
    vector<GPUAtom> getGPUAtomsFromAtoms(vector<Atom> & atoms);
    void getProtAndWaterFromPDB(vector<vector<string>> hbondtable, vector<Atom> & water, vector<Atom> & proteinDonor, vector<Atom> & proteinAcceptor, 
        vector<Atom> & proteinLinker);
    void getProtAndWaterFromAtom(vector<Atom> & atoms, vector<vector<string>> & hbondtable, vector<Atom> & water, vector<Atom> & proteinDonor, 
        vector<Atom> & proteinAcceptor, vector<Atom> & proteinLinker);
};

#endif