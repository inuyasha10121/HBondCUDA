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

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>

#include "CSVReader.h"

using namespace std;

//Splitting code found on http://stackoverflow.com/questions/30797769/splitting-a-string-but-keeping-empty-tokens-c
void split(const string& str, vector<string>& tokens, const string& delimiters)
{
    // Start at the beginning
    string::size_type lastPos = 0;
    // Find position of the first delimiter
    string::size_type pos = str.find_first_of(delimiters, lastPos);

    // While we still have string to read
    while (string::npos != pos && string::npos != lastPos)
    {
        // Found a token, add it to the vector
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Look at the next token instead of skipping delimiters
        lastPos = pos + 1;
        // Find the position of the next delimiter
        pos = str.find_first_of(delimiters, lastPos);
    }

    // Push the last token
    tokens.push_back(str.substr(lastPos, pos - lastPos));
}

vector<string> split(const string &s, const string& delimiters) {
    vector<string> elems;
    split(s, elems, delimiters);
    return elems;
}

CSVReader::CSVReader(string csvPath)
{
    this->csvPath = csvPath;
    csvStream.open(csvPath);

    if (csvStream.is_open())
        isOpen = true;
}

CSVReader::~CSVReader()
{
    if (isOpen)
        csvStream.close();
}

vector<vector<string>> CSVReader::readCSVFile()
{
    vector<vector<string>> csvcontents;
    //Start reading through the file and adding amino acids to the list
    if (isOpen) //Can we open the file?
    {
        string line;
        while (getline(csvStream, line)) //Read each line through the file
        {
            auto linecontents = split(line, ",");
            csvcontents.push_back(linecontents);
        }
    }
    else
    {
        csvcontents.clear();
    }
    return csvcontents;
}