#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>

#include "xdrfile_trr.h"
#include "xdrfile_xtc.h"

#include "helper_string.h"
#include "CalculationMethods.h"
#include "PDBProcessor.h"
#include "CSVReader.h"

using namespace std;

char * pdbpath = "D:\\CALB\\test.pdb";
char * hbondtablepath = "D:\\CALB\\HBondTableRev2.csv";
char * trajpath = "D:\\CALB\\noPBC_nvt.trr";
char * outpath = "D:\\CALB\\hbondlog.txt";
char * csvpath = "D:\\CALB\\analysis.csv";
int dt = 1;
int hbondwindow = 5; //MUST BE ODD
int windowthreshold = 4; //Inclusive

//---------------------------------------------MAIN CODE BODY---------------------------------------------
int index3d(int z, int y, int x, int xmax, int ymax)
{
    return (z * xmax * ymax) + (y * xmax) + xmax;
}

int performTimelineAnalysis(char * logpath, cudaDeviceProp deviceProp);

int main(int argc, char **argv)
{   
    setbuf(stdout, NULL);
    //Command line arguements to make the code a bit more versatile
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h") ||
        argc < 2)
    {
        cout << "Usage: " << argv[0] << " -pdb = PDB File Path (Required if not jumping directly to analysis)" << endl;
        cout << "      -hbt = Hydrogen Bond Lookup Table (Required if not jumping directly to analysis)" << endl;
        cout << "      -trj = Trajectory Path (Required if not jumping directly to analysis)" << endl;
        cout << "      -ol = Outpath for Hydrogen Bond Timeline File (Required ALWAYS)" << endl << endl;
        cout << "      -oa = Outpath for Analysis CSV file (Required ALWAYS)" << endl;
        cout << "      -dt = Frame skip parameter (Optional, Default 1)" << endl;
        cout << "      -window = Window frame size for bond analysis (MUST BE ODD) (Optional, Default 5)" << endl;
        cout << "      -wint = Window threshold for bond analysis (Optional, Default 4)" << endl;
        cout << "      -analysisonly = Jumps to analysis of timeline file (Optional)" << endl << endl;

        return 0;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "ol"))
    {
        getCmdLineArgumentString(argc, (const char**)argv, "ol", &outpath);
    }
    else
    {
        cout << "An output log (-ol) file MUST be specified." << endl;
        cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

        return 1;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "oa"))
    {
        getCmdLineArgumentString(argc, (const char**)argv, "oa", &csvpath);
    }
    else
    {
        cout << "An output analysis csv (-oa) file MUST be specified." << endl;
        cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

        return 1;
    }

    cudaDeviceProp deviceProp = setupCUDA();

    if (checkCmdLineFlag(argc, (const char**)argv, "analysisonly"))
    {
        return performTimelineAnalysis(outpath, deviceProp);
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "pdb"))
    {
        getCmdLineArgumentString(argc, (const char**)argv, "pdb", &pdbpath);
    }
    else
    {
        cout << "A pdb file (-pdb) MUST be specified." << endl;
        cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

        return 1;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "hbt"))
    {
        getCmdLineArgumentString(argc, (const char**)argv, "hbt", &hbondtablepath);
    }
    else
    {
        cout << "A trajectory file (-hbt) MUST be specified." << endl;
        cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

        return 1;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "trj"))
    {
        getCmdLineArgumentString(argc, (const char**)argv, "trj", &trajpath);
    }
    else
    {
        cout << "A trajectory file (-trj) MUST be specified." << endl;
        cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

        return 1;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "dt"))
    {
        dt = getCmdLineArgumentInt(argc, (const char**)argv, "dt");

        if (dt < 1)
        {
            cout << "Error: dt must be greater than 0." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "window"))
    {
        hbondwindow = getCmdLineArgumentInt(argc, (const char**)argv, "window");

        if (dt < 1)
        {
            cout << "Error: window must be greater than 0." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }

        if (dt % 2 != 1)
        {
            cout << "Error: window must be an odd value." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "wint"))
    {
        windowthreshold = getCmdLineArgumentInt(argc, (const char**)argv, "wint");

        if (dt < 1)
        {
            cout << "Error: wint must be greater than 0." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }

        if (windowthreshold > hbondwindow)
        {
            cout << "Error: wint must be less than or equal to window." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }
    }

    //Just some initial stuff we will need later
    PDBProcessor pdbprocessor(pdbpath);
    CSVReader csvreader(hbondtablepath);
    printf("Reading bond identities from hbond csv table...");
    auto hbondtable = csvreader.readCSVFile();
    printf("Done.\n");

    //Inputs of frame hbond function
    vector<Atom> water;
    vector<Atom> proteinacceptor;
    vector<Atom> proteindonor;
    vector<Atom> proteinlinker;

    //Read the input PDB to get the atom system information
    printf("Reading protein and water from PDB file...\n");
    auto atoms = pdbprocessor.getAtomsFromPDB();
    pdbprocessor.getProtAndWaterFromAtom(atoms, hbondtable, water, proteindonor, proteinacceptor, proteinlinker);
    printf("Done.\n");
    printf("Found %i donors, %i acceptors, %i linkers, and %i water atoms.\n", proteindonor.size(), proteinacceptor.size(), proteinlinker.size(), water.size());

    //Setup some stuff for CUDA, just in case
    vector<GPUAtom> gpuwater = pdbprocessor.getGPUAtomsFromAtoms(water);
    vector<GPUAtom> gpuproteinacceptor = pdbprocessor.getGPUAtomsFromAtoms(proteinacceptor);
    vector<GPUAtom> gpuproteindonor = pdbprocessor.getGPUAtomsFromAtoms(proteindonor);
    vector<GPUAtom> gpuproteinlinker = pdbprocessor.getGPUAtomsFromAtoms(proteinlinker);

    //Time to start processing the trajetory
    //First, determine what kind of file it is
    string trjp(trajpath);
    auto extension = trjp.substr(trjp.find_last_of('.'), trjp.length() - trjp.find_last_of('.'));
    if (extension != ".xtc" && extension != ".trr")
    {
        printf("ERROR: Trajectory file extension not recognized.  It must be a .xtc or .trr file! (Found %s)\n", extension.c_str());
        cin.get();
        return 1;
    }
    auto is_xtc = extension == ".xtc";

    //Start the reading process
    XDRFILE *xd_read;
    vector<char> treatedpath(trjp.begin(), trjp.end());
    treatedpath.push_back('\0');

    xd_read = xdrfile_open(&treatedpath[0], "r");
    if (xd_read == NULL)
    {
        printf("ERROR: Could not open trajectory file for reading: %s\n", trjp.c_str());
        cin.get();
        return 1;
    }
    
    //Read out relevant header information
    int result_xdr, xdr_natoms, xdr_step;
    matrix xdr_box;
    rvec *xdr_coords, *trr_vels, *trr_fs;
    float xdr_time, xtc_prec = 1000.0f, trr_lambda = 0.0f;

    //Read the number of atoms in each frame
    if (is_xtc)
    {
        result_xdr = read_xtc_natoms(&treatedpath[0], &xdr_natoms);
    }
    else
    {
        result_xdr = read_trr_natoms(&treatedpath[0], &xdr_natoms);
        trr_vels = new rvec[xdr_natoms];
        trr_fs = new rvec[xdr_natoms];
    }
    if (exdrOK != result_xdr)
    {
        printf("ERROR: Reading trajectory num atoms.  Code: %i\n", result_xdr);
        cin.get();
        return 1;
    }

    //Allocate memory for a frame of atomic coordinates
    xdr_coords = new rvec[xdr_natoms];

    //Setup logger
    FILE *logger;
    char * logpath = &outpath[0];
    logger = fopen(logpath, "w");
    if (logger == NULL)
    {
        printf("Error: Log file could not be opened for writing.");
        std::cin.get();
        return 1;
    }

    //Keep reading the trajectory file until we hit the end
    int currentframe = 0;
    int framesprocessed = 0;
    long averagetime = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    //clock_t begin = clock();
    //clock_t end;

    //[FRAME][# interaction][Prot/Water res id]
    vector<vector<vector<int>>> timeline;

    printf("Processing frame: %i", currentframe);
    do
    {
        //Read a frames worth of coordinate data
        if (is_xtc)
        {
            result_xdr = read_xtc(xd_read, xdr_natoms, &xdr_step, &xdr_time, xdr_box, xdr_coords, &xtc_prec);
        }
        else
        {
            result_xdr = read_trr(xd_read, xdr_natoms, &xdr_step, &xdr_time, &trr_lambda, xdr_box, xdr_coords, trr_vels, trr_fs);
        }

        if (currentframe % dt == 0)
        {
            timeline.push_back(vector<vector<int>>());
            //Setup storage variables
            vector<GPUAtom> gpuclosewaters;
            vector<vector<int>> gpudonortowater;
            vector<vector<int>> gpuacceptortowater;

            //Load the new coordinate information into the lists
            for (int i = 0; i < xdr_natoms; i++)
            {
                switch (atoms[i].hbondType)
                {
                case 'W':
                    gpuwater[atoms[i].hbondListPos].x = xdr_coords[i][0] * 10.0f;
                    gpuwater[atoms[i].hbondListPos].y = xdr_coords[i][1] * 10.0f;
                    gpuwater[atoms[i].hbondListPos].z = xdr_coords[i][2] * 10.0f;
                    break;
                case 'A':
                    gpuproteinacceptor[atoms[i].hbondListPos].x = xdr_coords[i][0] * 10.0f;
                    gpuproteinacceptor[atoms[i].hbondListPos].y = xdr_coords[i][1] * 10.0f;
                    gpuproteinacceptor[atoms[i].hbondListPos].z = xdr_coords[i][2] * 10.0f;
                    break;
                case 'D':
                    gpuproteindonor[atoms[i].hbondListPos].x = xdr_coords[i][0] * 10.0f;
                    gpuproteindonor[atoms[i].hbondListPos].y = xdr_coords[i][1] * 10.0f;
                    gpuproteindonor[atoms[i].hbondListPos].z = xdr_coords[i][2] * 10.0f;
                    break;
                case 'L':
                    gpuproteinlinker[atoms[i].hbondListPos].x = xdr_coords[i][0] * 10.0f;
                    gpuproteinlinker[atoms[i].hbondListPos].y = xdr_coords[i][1] * 10.0f;
                    gpuproteinlinker[atoms[i].hbondListPos].z = xdr_coords[i][2] * 10.0f;
                    break;
                case 'E':
                    break;
                default:
                    printf("ERROR: Internal hbond lookup table is mangled somehow! (Found: %c)\n", atoms[i].hbondType);
                    cin.get();
                    return 1;
                }
            }

            //Run the calculation
            getHBondsGPU(gpuproteindonor, gpuproteinacceptor, gpuproteinlinker, gpuwater, gpuclosewaters, gpudonortowater, gpuacceptortowater, deviceProp);

            //Record the data out in some meaningful manner
            //TODO: Maybe replace this with faster stuff?
            fprintf(logger, "FRAME: %i\n", xdr_step);

            for (int i = 0; i < gpudonortowater.size(); i++)
            {
                fprintf(logger, "%i,%i\n", gpudonortowater[i][0], gpudonortowater[i][1]);
                timeline[framesprocessed].push_back(vector<int>());
                timeline[framesprocessed][i].push_back(gpudonortowater[i][0]);
                timeline[framesprocessed][i].push_back(gpudonortowater[i][1]);
            }
            for (int i = 0; i < gpuacceptortowater.size(); i++)
            {
                fprintf(logger, "%i,%i\n", gpuacceptortowater[i][0], gpuacceptortowater[i][1]);
                timeline[framesprocessed].push_back(vector<int>());
                timeline[framesprocessed][gpudonortowater.size() + i].push_back(gpuacceptortowater[i][0]);
                timeline[framesprocessed][gpudonortowater.size() + i].push_back(gpuacceptortowater[i][1]);
            }
            

            //For timing
            auto t2 = std::chrono::high_resolution_clock::now();
            auto elapsedtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            averagetime += elapsedtime;
            printf("\r(avg: %.2f ms/ps, curr: %ld ms, step: %i, simtime: %.2f)", ((float)averagetime / (xdr_time + 0.2f)), elapsedtime, xdr_step, xdr_time);
            framesprocessed++;
            t1 = t2;
        }
        currentframe++;
    } while (result_xdr == 0);

    //Release the allocated memory to prevent a memory leak.
    delete[] xdr_coords;
    if (!is_xtc)
    {
        delete[] trr_vels;
        delete[] trr_fs;
    }


    fclose(logger);
    return performTimelineAnalysis(outpath, deviceProp);
}

//Splitting code found on http://stackoverflow.com/questions/30797769/splitting-a-string-but-keeping-empty-tokens-c
void splits(const string& str, vector<string>& tokens, const string& delimiters)
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

vector<string> splits(const string &s, const string& delimiters) {
    vector<string> elems;
    splits(s, elems, delimiters);
    return elems;
}

int performTimelineAnalysis(char * logpath, cudaDeviceProp deviceProp)
{
    vector<int> tllookup;
    vector<int> flattimeline;

    //Read the log file
    printf("Opening timeline log file for analysis...\n");
    ifstream logfile;
    logfile.open(logpath);
    if (!logfile.is_open())
    {
        printf("Error opening timeline file for analysis.");
        return 1;
    }
    int currline = 0;
    string line;
    while (getline(logfile,line))
    {
        printf("\rCurrent line: %i", currline);
        if (line.find("FRAME") != string::npos)
        {
            tllookup.push_back(flattimeline.size());
        }
        else if (line.find(',') != string::npos)
        {
            auto values = splits(line, ",");
            flattimeline.push_back(stoi(values[0]));
            flattimeline.push_back(stoi(values[1]));
        }
        currline++;
    }
    tllookup.push_back(flattimeline.size());
    logfile.close();
    printf("\nDone!\n");
    //Start doing analysis
    //Get a list of all the waters that participated in hydrogen bonding
    printf("Getting lists of waters and amino acids involved...");
    vector<int> boundwaters;
    vector<int> boundAAs;

    for (int i = 0; i < flattimeline.size(); i++)
    {
        if (i % 2 == 0)
        {
            if (!(find(boundAAs.begin(), boundAAs.end(), flattimeline[i]) != boundAAs.end()))
            {
                boundAAs.push_back(flattimeline[i]);
            }
        }
        else
        {
            if (!(find(boundwaters.begin(), boundwaters.end(), flattimeline[i]) != boundwaters.end()))
            {
                boundwaters.push_back(flattimeline[i]);
            }
        }
    }
    sort(boundwaters.begin(), boundwaters.end());
    sort(boundAAs.begin(), boundAAs.end());
    printf("Done!");
    printf("\nNumber waters involved in hydrogen bonding: %i\n", boundwaters.size());
    printf("Number AAs involved in hydrogen bonding: %i\n", boundAAs.size());

    //Start processing the timeline information
    FILE *csvout;
    char * csv = &csvpath[0];
    csvout = fopen(csv, "w");
    if (csvout == NULL)
    {
        printf("Error: csv file could not be opened for writing.");
        std::cin.get();
        return 1;
    }

    //Water ID:, Bridger?:, Bulk?:, #AAs:, # Events:
    printf("Performing analysis.  This may take a while...\n");
    fprintf(csvout, "Water ID:,Bridger?:,Bulk?:,# AAs:,# Events:\n");

    //GPU METHOD
    auto gpuAAs = &boundAAs[0];
    auto gputimeline = &flattimeline[0];
    auto gputllookup = &tllookup[0];

    char * timelinemap = new char[boundAAs.size() * (tllookup.size() - 1)];
    char * bridgers = new char[tllookup.size() - 1];
    char * visited = new char[boundAAs.size()];
    int * framesbound = new int[tllookup.size() - 1];

    for (int currwater = 0; currwater < boundwaters.size(); currwater++)
    {
        printf("\rProcessing water %i of %i", currwater + 1, boundwaters.size());
        fill(timelinemap, timelinemap + (boundAAs.size() * (tllookup.size() - 1)), false);
        fill(bridgers, bridgers + tllookup.size() - 1, false);
        fill(visited, visited + boundAAs.size(), false);
        fill(framesbound, framesbound + (tllookup.size() - 1), 0);

        timelineMapCuda(timelinemap, gputimeline, gputllookup, gpuAAs, boundwaters[currwater], hbondwindow, windowthreshold, tllookup[tllookup.size() - 1], tllookup.size() - 1, boundAAs.size(), deviceProp);
        visitAndBridgerAnalysisCuda(bridgers, visited, framesbound, timelinemap, tllookup.size() - 1, boundAAs.size(), deviceProp);

        bool bridger = false;
        int visits = 0;
        int totalframes = 0;
        for (int i = 0; i < tllookup.size() - 1; i++)
        {
            bridger = bridger || (bool)bridgers[i];
            totalframes += (bool)framesbound[i];
        }
        for (int i = 0; i < boundAAs.size(); i++)
        {
            visits += (bool)visited[i];
        }
        fprintf(csvout, "%i,%s,%s,%i,%i\n", boundwaters[currwater], bridger ? "true" : "false", (visits > 1) ? "false" : "true", visits, -1);
    }
    delete[] timelinemap;
    delete[] bridgers;
    delete[] visited;
    delete[] framesbound;
    //OLD CPU METHOD
    /*
    //Binding edges descibes how many transition events happened, which roughly quantifies how frequently the water participated in hbonding
    vector<int> bindingedges;
    bindingedges.resize(boundAAs.size());
    vector<bool> bridger;
    bridger.resize(boundwaters.size());
    fill(bridger.begin(), bridger.end(), false); //Likely un-necessary, but just in case.
    for (int nwater = 0; nwater < boundwaters.size(); nwater++)
    {
        int numevents = 0;
        vector<int> visitedlist;
        fill(bindingedges.begin(), bindingedges.end(), 0);
        printf("\rProcessing water %i of %i", nwater + 1, boundwaters.size());
        for (int nframe = 0; nframe < timeline.size() - hbondwindow; nframe++)
        {
            int boundcount = 0;
            for (int nprot = 0; nprot < boundAAs.size(); nprot++)
            {
                int boundframes = 0;
                for (int nwindow = 0; nwindow < hbondwindow; nwindow++)
                {
                    for (int nsearch = 0; nsearch < timeline[nframe + nwindow].size(); nsearch++)
                    {
                        if ((timeline[nframe + nwindow][nsearch][0] == boundAAs[nprot]) && (timeline[nframe + nwindow][nsearch][1] == boundwaters[nwater]))
                        {
                            boundframes++;
                        }
                    }
                }
                if (((boundframes >= 4) && (bindingedges[nprot] % 2 == 0)) || ((boundframes < 4) && (bindingedges[nprot] % 2 == 1)))
                {
                    bindingedges[nprot]++;
                    numevents++;
                }
                if (bindingedges[nprot] % 2 == 1)
                {
                    boundcount++;
                    if (visitedlist.empty())
                    {
                        visitedlist.push_back(boundAAs[nprot]);
                    }
                    else
                    {
                        if (find(visitedlist.begin(), visitedlist.end(), boundAAs[nprot]) == visitedlist.end())
                        {
                            visitedlist.push_back(boundAAs[nprot]);
                        }
                    }
                }
            }

            if (boundcount > 1)
            {
                bridger[nwater] = true;
            }
            if (boundcount > 4)
            {
                printf("WARNING: More than 4 partners found on a water.\n");
            }
        }
        fprintf(csvout, "%i,%s,%s,%i,%i\n", boundwaters[nwater], bridger[nwater] ? "true" : "false", (visitedlist.size() > 1) ? "false" : "true", visitedlist.size(), numevents);
    }
    */
    printf("\n\nDone with analysis!\n");
    cin.get();
    
    return 0;
}