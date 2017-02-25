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

// C++ code used to read the CSV charge file

#ifndef CSVREADER_H
#define CSVREADER_H

#include <string>

using namespace std;

class CSVReader {
private:
    string csvPath;
    ifstream csvStream;
    bool isOpen = false;

public:
    CSVReader(string csvPath);
    ~CSVReader();

    bool is_open() { return isOpen; }

    vector<vector<string>> CSVReader::readCSVFile();
};

#endif