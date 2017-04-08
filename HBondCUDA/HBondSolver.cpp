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
float cudaMemPercentage = 0.75f;
int gpuid = 0;

//---------------------------------------------MAIN CODE BODY---------------------------------------------
int index3d(int z, int y, int x, int xmax, int ymax)
{
    return (z * xmax * ymax) + (y * xmax) + xmax;
}

int performTimelineAnalysis(char * logpath, cudaDeviceProp deviceProp);
int testfunc(cudaDeviceProp deviceProp);

int main(int argc, char **argv)
{   
    setbuf(stdout, NULL);
    //Command line arguements to make the code a bit more versatile
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h") ||
        argc < 2)
    {
        cout << "Usage: " << argv[0] << " -pdb = PDB File Path (Required if not jumping directly to analysis)" << endl;
        cout << "      -hbt : Hydrogen Bond Lookup Table (Required if not jumping directly to analysis)" << endl;
        cout << "      -trj : Trajectory Path (Required if not jumping directly to analysis)" << endl;
        cout << "      -ol : Outpath for Hydrogen Bond Timeline File (Required ALWAYS)" << endl << endl;
        cout << "      -oa : Outpath for Analysis CSV file (Required ALWAYS)" << endl;
        cout << "      -dt=<ARG> : Frame skip parameter (Optional, Default 1)" << endl;
        cout << "      -window=<ARG> : Window frame size for bond analysis (Optional, Default 5)" << endl;
        cout << "      -wint=<ARG> : Window threshold for bond analysis (Optional, Default 4)" << endl;
        cout << "      -gpumem=<ARG> : Percentage of total per-graphics card memory allowed to be used (Optional, 0 to 1, Default 0.15)" << endl;
        cout << "      -gpuid=<ARG> : ID value of which graphics card to use (Optional, Default 0)" << endl;
        cout << "      -analysisonly = Jumps to analysis of timeline file (Optional)" << endl;
        cout << endl;

        return 0;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "gpumem"))
    {
        cudaMemPercentage = getCmdLineArgumentFloat(argc, (const char**)argv, "gpumem");
        if (cudaMemPercentage > 1.0f || cudaMemPercentage < 0.0f)
        {
            cout << "Error: gpumem must be between 0 and 1!" << endl;
            cout << "Exiting..." << endl;

            return 1;
        }
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "gpuid"))
    {
        gpuid = getCmdLineArgumentInt(argc, (const char**)argv, "gpuid");
        //TODO: Do some checking here to see if the ID given exists.  
    }

    cudaDeviceProp deviceProp = setupCUDA(gpuid);

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
    vector<int> boundwaters;
    vector<int> boundAAs;

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

    //TODO: Do this in a binary fashion, hopefully to speed things up.  
    while (getline(logfile,line))
    {
        if (line.find("FRAME") != string::npos)
        {
            printf("\rCurrent line: %i", currline);
            tllookup.push_back(flattimeline.size());
        }
        else if (line.find(',') != string::npos)
        {
            auto values = splits(line, ",");
            auto resAA = stoi(values[0]);
            auto reswater = stoi(values[1]);
            auto AAsearch = find(boundAAs.begin(), boundAAs.end(), resAA);
            auto watersearch = find(boundwaters.begin(), boundwaters.end(), reswater);

            if (AAsearch == boundAAs.end())
            {
                flattimeline.push_back(boundAAs.size());
                boundAAs.push_back(resAA);
            }
            else
            {
                flattimeline.push_back(distance(boundAAs.begin(), AAsearch));
            }

            if (watersearch == boundwaters.end())
            {
                flattimeline.push_back(boundwaters.size());
                boundwaters.push_back(reswater);
            }
            else
            {
                flattimeline.push_back(distance(boundwaters.begin(), watersearch));
            }
        }
        currline++;
    }
    tllookup.push_back(flattimeline.size());
    logfile.close();
    printf("\nDone!\n");

    //Start doing analysis
    
    int numAAs = boundAAs.size();
    int numFrames = (tllookup.size() - 1);
    int numWaters = boundwaters.size();

    printf("Number waters involved in hydrogen bonding: %i\n", numWaters);
    printf("Number AAs involved in hydrogen bonding: %i\n", numAAs);
    printf("Number of frames found: %i\n", numFrames);

    //Start processing the timeline information
    FILE *csvout;
    char * csv = &csvpath[0];
    csvout = fopen(csv, "w");
    if (csvout == NULL)
    {
        printf("Error: csv file could not be opened for writing.");
        return 1;
    }

    //Water ID:, Bridger?:, Bulk?:, #AAs:, # Events:
    printf("Performing analysis.  This may take a while...\n");

    //-------------------------------------------------------------------GPU METHOD-------------------------------------------------------------------
    size_t cudaFreeMem;
    cudaError_t cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL); //Get how much memory is available to use

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaMemGetInfo failed!" << endl;
        printf("\nERROR: CUDA is unable to function.  Double check your installation/device settings.");
        printf("\nExiting...");
        return 1;
    }

    cudaFreeMem *= cudaMemPercentage; //Adjust memory based on command line
    cudaFreeMem -= ((sizeof(int) * flattimeline.size()) + (sizeof(int) * tllookup.size())); //Reserve space for timeline
    cudaFreeMem -= ((sizeof(char) * numAAs) + (sizeof(int) * 2) + (sizeof(int) * numFrames)); //Reserve space for output (visit list, event information, temp mem for event info building)
    if (cudaFreeMem < 1)
    {
        cerr << "ERROR: Not enough memory to process the trajectory." << endl;
        cerr << "Exitting..." << endl;
        return 1;
    }
    auto gpuFlatTimeline = &flattimeline[0];
    auto gpuTLLookup = &tllookup[0];

    size_t memPerWater = sizeof(char) * numFrames * numAAs;
    if (memPerWater > cudaFreeMem)
    {
        cerr << "ERROR: Not able to process one water at one time." << endl;
        cerr << "Exitting..." << endl;
        return 1;
    }
    cout << "Processing waters..." << endl;

    //Write the csv file header
    fprintf(csvout, "Water ID:,Bridger?,# Events:, # Frames Bound:,Visit List:,\n");

    //Process each water
    auto gpuVisitedList = new char[numAAs]; //List of visited amino acids
    auto gpuLoadedTimeline = new char[numAAs * numFrames]; //Temporary matrix for the loaded timeline
    auto gpuFrameEventInfo = new int[numFrames];  //Temporary matrix of hydrogen bond information over frames

    auto totaltime = 0;


    for (int currWater = 0; currWater < numWaters; ++currWater)
    {
        cout << "\rProcessing water " << currWater + 1 << " of " << numWaters;
        auto t1 = std::chrono::high_resolution_clock::now();

        //Prep the timeline memory for the upcoming load
        fill(gpuLoadedTimeline, gpuLoadedTimeline + (numAAs * numFrames), '0');

        //Step 1: Load the timeline information into memory
        loadTimelineLauncher(gpuLoadedTimeline, gpuFlatTimeline, gpuTLLookup, currWater, flattimeline.size(), tllookup.size(), numFrames, numAAs, cudaMemPercentage, deviceProp);

        //DEBUG: Check if the timeline got loaded properly
        /*
        for (int frame = 0; frame < tllookup.size() - 1; frame++)
        {
            int searchEnd = tllookup[frame + 1];
            for (int searchPos = tllookup[frame]; searchPos < searchEnd; searchPos += 2)
            {
                if (flattimeline[searchPos + 1] == currWater)
                {
                    if (gpuLoadedTimeline[(flattimeline[searchPos] * numFrames) + frame] != 1)
                    {
                        cout << "ERROR! Frame: " << frame << "\tAA:" << boundAAs[flattimeline[searchPos]] << "\tWater: " << boundwaters[currWater] << endl;
                        cin.get();
                    }
                }
            }
        }
        */
        //DEBUG: Save previous array for analysis later
        auto oldTimeline = new char[numFrames * numAAs];
        copy(gpuLoadedTimeline, gpuLoadedTimeline + (numFrames * numAAs), oldTimeline);

        //Step 2: Perform sliding window analysis of the timeline to smooth high frequency fluctuations
        windowTimelineLauncher(gpuLoadedTimeline, hbondwindow, windowthreshold, numFrames, numAAs, cudaMemPercentage, deviceProp);
        //DEBUG: Check if the timeline was processed properly
        /*
        for (int currFrame = 0; currFrame < numFrames - hbondwindow; ++currFrame)
        {
            for (int currAA = 0; currAA < numAAs; ++currAA)
            {
                int boundFrames = 0;
                for (int currWindow = 0; currWindow < hbondwindow; ++currWindow)
                {
                    if (oldTimeline[(currAA * numFrames) + currFrame + currWindow] == 1)
                    {
                        ++boundFrames;
                    }
                }
                if (boundFrames >= windowthreshold)
                {
                    if (gpuLoadedTimeline[(currAA * numFrames) + currFrame] != 1)
                    {
                        cout << "ERROR! Frame: " << currFrame << "\tAA:" << boundAAs[currAA] << "\tWater: " << boundwaters[currWater] << endl;
                        cin.get();
                    }
                }
                else
                {
                    if (gpuLoadedTimeline[(currAA * numFrames) + currFrame] != 0)
                    {
                        cout << "ERROR! Frame: " << currFrame << "\tAA:" << boundAAs[currAA] << "\tWater: " << boundwaters[currWater] << endl;
                        cin.get();
                    }
                }
            }
        }
        */
        //Perform analysis kernels
        /*
        cudaResult =  timelineWindowCUDA(gpuTemp2DMatrix, gpuFlatTimeline, gpuTLLookup, hbondwindow, windowthreshold, currWater, numAAs, numFrames, flattimeline.size(), tllookup.size(), deviceProp);
        if (cudaResult != cudaSuccess) {
            cerr << "ERROR: timelineWindowCUDA failed!" << endl;
            cerr << "Exitting..." << endl;
            return 1;
        }

        cudaResult = eventListCUDA(gpuFrameEventInfo, gpuTemp2DMatrix, numAAs, numFrames, deviceProp);
        if (cudaResult != cudaSuccess) {
        cerr << "ERROR: eventListCUDA failed!" << endl;
        cerr << "Exitting..." << endl;
        return 1;
        }

        
        int errorcode = performFlatTimelineAnalysis(gpuCorrectedTimeline, flattimeline, tllookup, hbondwindow, windowthreshold, currWater, numAAs, numFrames, cudaMemPercentage, deviceProp);
        if (errorcode == 1)
        {
            cout << "Something went wrong with the flat timeline analysis.  " << endl;
            cout << "Exitting..." << endl;
            return 1;
        }
        int totalEvents = 0, framesBound = 0;

        errorcode = performTimelineEventAnalysis(totalEvents, framesBound, gpuCorrectedTimeline, numAAs, numFrames, cudaMemPercentage, deviceProp);
        if (errorcode == 1)
        {
            cout << "Something went wrong with the timeline event analysis.  " << endl;
            cout << "Exitting..." << endl;
            return 1;
        }

        //TODO: This needs to be replaced with a similar method to the ones above.  
        cudaResult = visitListCUDA(gpuVisitedList, gpuCorrectedTimeline, numAAs, numFrames, deviceProp);
        if (cudaResult != cudaSuccess) {
            cerr << "ERROR: visitListCUDA failed!" << endl;
            cerr << "Exitting..." << endl;
            return 1;
        }
                
        //Harvest the data and print the results to file
        /*
        for (int currFrame = 0; currFrame < numFrames; ++currFrame)
        {
            totalEvents += gpuFrameEventInfo[currFrame];
            framesBound += (gpuFrameEventInfo[currFrame] > 0) ? 1 : 0;
        }
        
        //    fprintf(csvout, "Water ID:,Bridger?,# Events:,# Frames Bound:,Visit List:,\n");
        fprintf(csvout, "%i,%s,%i,%i,", boundwaters[currWater], (totalEvents != framesBound) ? "true" : "false", totalEvents, framesBound);
        
        //TODO: This feels wrong as fuck.  
        for (int currAA = 0; currAA < numAAs; ++currAA)
        {
            if (gpuVisitedList[currAA] == 1)
            {
                fprintf(csvout, "%i,", boundAAs[currAA]);
            }
        }
        fprintf(csvout, "\n");
        fflush(csvout);
        */
        auto t2 = std::chrono::high_resolution_clock::now();
        auto elapsedtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        totaltime += elapsedtime;
        auto predictedtime = (totaltime / (currWater + 1) * (numWaters - currWater));

        int seconds = (int)(predictedtime / 1000) % 60;
        int minutes = (int)((predictedtime / (1000 * 60)) % 60);
        int hours = (int)((predictedtime / (1000 * 60 * 60)) % 24);
        printf("\tPredicted time remaining: %03i:%02i:%02i", hours, minutes, seconds);
    }
    delete[] gpuVisitedList;
    delete[] gpuFrameEventInfo;
    delete[] gpuLoadedTimeline;

    printf("\n\nDone with analysis!\n");

    return 0;
}

int testfunc(cudaDeviceProp deviceProp)
{
    size_t cudaFreeMem;
    cudaError_t cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL); //Get how much memory is available to use

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaMemGetInfo failed!" << endl;
        printf("\nERROR: CUDA is unable to function.  Double check your installation/device settings.");
        printf("\nExiting...");
        return 1;
    }

    cudaFreeMem *= cudaMemPercentage; //Adjust memory based on command line

    //Exit
    cout << "DONE" << endl;
    cin.get();
    return 0;
}