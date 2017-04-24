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

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <locale>
#include <tuple>

#include "PDBProcessor.h"
#include "GPUTypes.h"

using namespace std;

// trimming stuff for when we read in a file
string & ltrim(string & str)
{
    auto it2 = find_if(str.begin(), str.end(), [](char ch) { return !isspace<char>(ch, locale::classic()); });
    str.erase(str.begin(), it2);
    return str;
}

string & rtrim(string & str)
{
    auto it1 = find_if(str.rbegin(), str.rend(), [](char ch) { return !isspace<char>(ch, locale::classic()); });
    str.erase(it1.base(), str.end());
    return str;
}

string & trim(string & str)
{
    return ltrim(rtrim(str));
}

// Constructor
PDBProcessor::PDBProcessor(string pdbPath)
{
    this->pdbPath = pdbPath;
    pdbStream.open(pdbPath);

    if (pdbStream.is_open())
        isOpen = true;

}

PDBProcessor::PDBProcessor()
{

}

// Deconstructor
PDBProcessor::~PDBProcessor()
{
    if (isOpen)
        pdbStream.close();
}

vector<Atom> PDBProcessor::getAtomsFromPDB()
{
    vector<Atom> atoms;

    // check if the file is open
    if (isOpen)
    {
        string line;

        int residoffset = 0;
        bool toggle = false;

        // read each line
        while (getline(pdbStream, line))
        {
            // read the first 4 characters
            auto begin = line.substr(0, 4);
            if (begin == "ATOM" || begin == "HETA")
            {
                // make an atom and get all the stuff for it
                Atom curAtom;

                // check the element first to see if we
                // need to keep going or not
                string temp = line.substr(76, 2);
                auto element = trim(temp);

                // default vdw is -1.0f, so only check if we need to change it
                // if it's not in the list, just break out (saves a lot of time)
                // TODO: Perhaps read a property file could be read so user-defined vdws could be used, in case we miss some for things such as metaloenzymes.
                if (element == "H")
                    curAtom.vdw = 1.2f;
                else if (element == "ZN")
                    curAtom.vdw = 1.39f;
                else if (element == "F")
                    curAtom.vdw = 1.47f;
                else if (element == "O")
                    curAtom.vdw = 1.52f;
                else if (element == "N")
                    curAtom.vdw = 1.55f;
                else if (element == "C")
                    curAtom.vdw = 1.7f;
                else if (element == "S")
                    curAtom.vdw = 1.8f;

                curAtom.element = element;

                auto name = line.substr(12, 4);
                auto resName = line.substr(17, 3);
                auto charge = line.substr(78, 2);

                // TODO: Slim this down to the bare essentials, since a lot of this information is not needed.
                // Keep it handy somewhere though, since this is useful reference for reading PDBs.
                curAtom.serial = stoi(line.substr(6, 5));
                curAtom.name = trim(name);
                curAtom.altLoc = line.at(16);
                curAtom.resName = trim(resName);
                curAtom.chainID = line.at(21);
                curAtom.resSeq = stoi(line.substr(22, 4)) + residoffset;
                curAtom.iCode = line.at(26);
                curAtom.x = stof(line.substr(30, 8));
                curAtom.y = stof(line.substr(38, 8));
                curAtom.z = stof(line.substr(46, 8));
                curAtom.occupancy = stof(line.substr(54, 6));
                curAtom.tempFactor = stof(line.substr(60, 6));
                curAtom.charge = trim(charge);

                
                if (((curAtom.resSeq % 10000) == 9999))
                {
                    if (toggle)
                    {
                        residoffset += 10000;
                        toggle = false;
                    }
                    
                }
                else
                {
                    toggle = true;
                }
                
                atoms.push_back(curAtom);
            }
        }

        cout << "Found " << atoms.size() << " atoms." << endl;

        return atoms;
    }
    else
    {
        // return an empty vector and check this to see if we
        // found atoms in the main function

        atoms.clear();
        return atoms;
    }
}


void PDBProcessor::getProtAndWaterFromPDB(vector<vector<string>> hbondtable, vector<Atom> & water, vector<Atom> & proteinDonor, vector<Atom> & proteinAcceptor, vector<Atom> & proteinLinker)
{
    // check if the file is open
    if (isOpen)
    {
        string line;

        int residoffset = 0;

        // read each line
        while (getline(pdbStream, line))
        {
            // read the first 4 characters
            auto begin = line.substr(0, 4);
            if (begin == "ATOM" || begin == "HETA")
            {
                // make an atom and get all the stuff for it
                Atom curAtom;

                // check the element first to see if we
                // need to keep going or not
                string temp = line.substr(76, 2);
                curAtom.element = trim(temp);

                auto name = line.substr(12, 4);
                auto resName = line.substr(17, 3);
                auto charge = line.substr(78, 2);

                // TODO: Slim this down to the bare essentials, since a lot of this information is not needed.
                // Keep it handy somewhere though, since this is useful reference for reading PDBs.
                curAtom.serial = stoi(line.substr(6, 5));
                curAtom.name = trim(name);
                curAtom.altLoc = line.at(16);
                curAtom.resName = trim(resName);
                curAtom.chainID = line.at(21);
                curAtom.resSeq = stoi(line.substr(22, 4)) + residoffset;
                curAtom.iCode = line.at(26);
                curAtom.x = stof(line.substr(30, 8));
                curAtom.y = stof(line.substr(38, 8));
                curAtom.z = stof(line.substr(46, 8));
                curAtom.occupancy = stof(line.substr(54, 6));
                curAtom.tempFactor = stof(line.substr(60, 6));
                curAtom.charge = trim(charge);

                if (curAtom.resSeq % 10000 == 9999)
                {
                    residoffset += 10000;
                }

                if (curAtom.resName == "SOL")
                {
                    curAtom.hbondType = 'W';
                    curAtom.hbondListPos = water.size();
                    water.push_back(curAtom);
                }
                else
                {
                    if (curAtom.resSeq == 1 && (curAtom.name == "H1" || curAtom.name == "H2" || curAtom.name == "H3"))
                    {
                        curAtom.hbondType = 'L';
                        curAtom.hbondListPos = proteinLinker.size();
                        proteinLinker.push_back(curAtom);
                    }
                    else
                    {
                        for (int i = 0; i < hbondtable.size(); i++)
                        {
                            if (hbondtable[i][0] == curAtom.resName && hbondtable[i][1] == curAtom.name)
                            {
                                if (hbondtable[i][2] == "D")
                                {
                                    curAtom.hbondType = 'D';
                                    curAtom.hbondListPos = proteinDonor.size();
                                    proteinDonor.push_back(curAtom);
                                }
                                else if (hbondtable[i][2] == "A")
                                {
                                    curAtom.hbondType = 'A';
                                    curAtom.hbondListPos = proteinAcceptor.size();
                                    proteinAcceptor.push_back(curAtom);
                                }
                                else if (hbondtable[i][2] == "L")
                                {
                                    curAtom.hbondType = 'L';
                                    curAtom.hbondListPos = proteinLinker.size();
                                    proteinLinker.push_back(curAtom);
                                }
                                else if (hbondtable[i][2] == "E")
                                {
                                    curAtom.hbondType = 'E';
                                    curAtom.hbondListPos = 0;
                                }
                                else
                                {
                                    cout << "Error: Unrecognized flag (" << hbondtable[i][0] << "," << hbondtable[i][1] << ") - " << hbondtable[i][2] << endl;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // return an empty vector and check this to see if we
        // found atoms in the main function
        cout << "ERROR: File could not be opened!" << endl;
    }
}

vector<GPUAtom> PDBProcessor::getGPUAtomsFromAtoms(vector<Atom> & atoms)
{
    vector<GPUAtom> gpuatoms;
    int natoms = atoms.size();
    for (int i = 0; i < natoms; i++)
    {
        GPUAtom temp;
        temp.x = atoms[i].x;
        temp.y = atoms[i].y;
        temp.z = atoms[i].z;
        temp.resid = atoms[i].resSeq;
        gpuatoms.push_back(temp);
    }

    return gpuatoms;
}

void PDBProcessor::getProtAndWaterFromAtom(vector<Atom> & atoms, vector<vector<string>> & hbondtable, vector<Atom> & water, vector<Atom> & proteinDonor, vector<Atom> & proteinAcceptor, vector<Atom> & proteinLinker)
{
    for (int i = 0; i < atoms.size(); i++)
    {
        atoms[i].hbondType = 'E';
        atoms[i].hbondListPos = 0;
        if (atoms[i].resName == "SOL")
        {
            atoms[i].hbondType = 'W';
            atoms[i].hbondListPos = water.size();
            water.push_back(atoms[i]);
        }
        else
        {
            if (atoms[i].resSeq == 1 && (atoms[i].name == "H1" || atoms[i].name == "H2" || atoms[i].name == "H3"))
            {
                atoms[i].hbondType = 'L';
                atoms[i].hbondListPos = proteinLinker.size();
                proteinLinker.push_back(atoms[i]);
            }
            else
            {
                for (int j = 0; j < hbondtable.size(); j++)
                {
                    if (hbondtable[j][0] == atoms[i].resName && hbondtable[j][1] == atoms[i].name)
                    {
                        if (hbondtable[j][2].find("D") != string::npos)
                        {
                            atoms[i].hbondType = 'D';
                            atoms[i].hbondListPos = proteinDonor.size();
                            proteinDonor.push_back(atoms[i]);
                        }
                        else if (hbondtable[j][2].find("A") != string::npos)
                        {
                            atoms[i].hbondType = 'A';
                            atoms[i].hbondListPos = proteinAcceptor.size();
                            proteinAcceptor.push_back(atoms[i]);
                        }
                        else if (hbondtable[j][2].find("L") != string::npos)
                        {
                            atoms[i].hbondType = 'L';
                            atoms[i].hbondListPos = proteinLinker.size();
                            proteinLinker.push_back(atoms[i]);
                        }
                        else if (hbondtable[j][2].find("E") != string::npos)
                        {
                            
                        }
                        else
                        {
                            cout << "Error: Unrecognized flag (" << hbondtable[j][0] << "," << hbondtable[j][1] << ") - " << hbondtable[j][2] << endl;
                        }
                        break;
                    }
                }
            }
        }
    }
}
