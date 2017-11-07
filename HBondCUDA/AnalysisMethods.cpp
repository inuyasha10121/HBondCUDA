#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>

#include "AnalysisMethods.h"
#include "CalculationMethods.h"
#include "PDBProcessor.h"
#include "CSVReader.h"
#include "xdrfile_trr.h"
#include "xdrfile_xtc.h"

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

int hbondTrajectoryAnalysis(char * pdbpath, char * hbtpath, char * trjpath, char * outlog, char * outanalysiscsv, char * outbridgercsv, int windowsize, int windowthreshold, int dt, float gpumem, int gpuid, cudaDeviceProp deviceProp)
{
	//Just some initial stuff we will need later
	PDBProcessor pdbprocessor(pdbpath);
	CSVReader csvreader(hbtpath);
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
	string trjp(trjpath);
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
	char * logpath = &outlog[0];
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
	return hbondTimelineAnalysis(outlog, outanalysiscsv, outbridgercsv, windowsize, windowthreshold, dt, gpumem, gpuid, deviceProp);
}


int hbondTimelineAnalysis(char * outlog, char * outanalysiscsv, char * outbridgercsv, int windowsize, int windowthreshold, int dt, float gpumem, int gpuid, cudaDeviceProp deviceProp)
{
	vector<int> tllookup;
	vector<int> flattimeline;
	vector<int> boundwaters;
	vector<int> boundAAs;

	//Read the log file
	printf("Opening timeline log file for analysis...\n");
	ifstream logfile;
	logfile.open(outlog);
	if (!logfile.is_open())
	{
		printf("Error opening timeline file for analysis.");
		return 1;
	}
	int currline = 0;
	string line;

	//TODO: Do this in a binary fashion, hopefully to speed things up.  
	while (getline(logfile, line))
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
	char * csv = &outanalysiscsv[0];
	csvout = fopen(csv, "w");
	if (csvout == NULL)
	{
		printf("Error: csv file could not be opened for writing.");
		return 1;
	}

	FILE *bridgeout;
	char * blpath = &outbridgercsv[0];
	bridgeout = fopen(blpath, "w");
	if (bridgeout == NULL)
	{
		printf("Error: Bridger log file could not be opened for writing.");
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

	cudaFreeMem *= gpumem; //Adjust memory based on command line
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
	fprintf(csvout, "Water ID:,# Bridging:,# States:,# State Changes:,# Ping-Pong,Ping-Pong?,# Events:, # Frames Bound:,Visit List:,\n");

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
		fill(gpuLoadedTimeline, gpuLoadedTimeline + (numAAs * numFrames), 0);

		//Step 1: Load the timeline information into memory
		auto errorcheck = loadTimelineLauncher(gpuLoadedTimeline, gpuFlatTimeline, gpuTLLookup, currWater, flattimeline.size(), tllookup.size(), numFrames, numAAs, gpumem, deviceProp);
		if (errorcheck == 1)
		{
			return 1;
		}

		//Step 2: Perform sliding window analysis of the timeline to smooth high frequency fluctuations
		errorcheck = windowTimelineLauncher(gpuLoadedTimeline, windowsize, windowthreshold, numFrames, numAAs, gpumem, deviceProp);
		if (errorcheck == 1)
		{
			return 1;
		}
		//Do zero-fill on last-unanalyzable section
		for (int j = 0; j < numAAs; ++j)
		{
			for (int i = numFrames - windowsize; i < numFrames; ++i)
			{
				gpuLoadedTimeline[(j * numFrames) + i] = 0;
			}
		}

		//Step 3: Perform analysis of timeline events
		int totalEvents = 0, boundFrames = 0, bridgeFrames = 0;
		errorcheck = timelineEventAnalysisLauncher(gpuFrameEventInfo, gpuLoadedTimeline, numFrames, numAAs, gpumem, deviceProp);
		if (errorcheck == 1)
		{
			return 1;
		}

		fprintf(bridgeout, "Water: %i\n", boundwaters[currWater]); //Make a log entry in the bridger file
		for (int currFrame = 0; currFrame < (numFrames); ++currFrame)
		{
			totalEvents += gpuFrameEventInfo[currFrame];
			boundFrames += (gpuFrameEventInfo[currFrame] > 0) ? 1 : 0;
			bridgeFrames += (gpuFrameEventInfo[currFrame] > 1) ? 1 : 0;
			if (gpuFrameEventInfo[currFrame] > 1) //Check if bridging is occuring this frame
			{
				fprintf(bridgeout, " ,Frame %i:,", currFrame);
				//Go through amino acids, see which ones are bound, and record
				for (int currAA = 0; currAA < numAAs; ++currAA)
				{
					if (gpuLoadedTimeline[(currAA * numFrames) + currFrame] == 1)
					{
						fprintf(bridgeout, "%i,", boundAAs[currAA]);
					}
				}
				fprintf(bridgeout, "\n");
				fflush(bridgeout);
			}
		}

		//Step 4: Perform amino acid visit analysis
		errorcheck = timelineVisitAnalysisLauncher(gpuVisitedList, gpuLoadedTimeline, numFrames, numAAs, gpumem, deviceProp);
		if (errorcheck == 1)
		{
			return 1;
		}

		//Step 5: Check for ping-ponging, if a bridger
		string pingpong = "N/A";
		int numStateChanges = 0, numPingPongs = 0, numStates = 0;
		if (bridgeFrames > 0)
		{
			pingPongChecker(numStates, numStateChanges, numPingPongs, gpuLoadedTimeline, gpuVisitedList, numFrames, numAAs);
			if (numPingPongs > 0)
			{
				pingpong = "YES";
			}
			else
			{
				pingpong = "NO";
			}
		}

		//Print the results to the log .csv file
		fprintf(csvout, "%i,%i,%i,%i,%i,%s,%i,%i,", boundwaters[currWater], bridgeFrames, numStates, numStateChanges, numPingPongs, pingpong.c_str(), totalEvents, boundFrames);

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

	fclose(csvout);
	fclose(bridgeout);

	printf("\n\nDone with analysis!\n");

	return 0;
}

int velocityAnalysis(char * pdbpath, char * trjpath, char * velocitycsv, char * neighborcsv, char * avgvelcsv, float cutoffdist, int dt, float gpumem, int gpuid, cudaDeviceProp deviceProp)
{
	cout << "Performing velocity based analysis" << endl;

	//Setup initial data
	PDBProcessor pdbProcessor(pdbpath);
	auto atoms = pdbProcessor.getAtomsFromPDB();
	vector<GPUAtom> waters, protein;
	pdbProcessor.getGPUHeavyProtWaterFromAtom(atoms, waters, protein);
	auto numwaters = waters.size();
	auto numprotein = protein.size();

	auto gpuprotein = &protein[0];
	auto gpuwater = &waters[0];

	//Start processing the trajectory
	string trjp(trjpath);
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

	//Setup loggers
	/*
	FILE *vellogger;
	vellogger = fopen(velocitycsv, "w");
	if (vellogger == NULL)
	{
		printf("Error: Velocity log file could not be opened for writing.");
		return 1;
	}

	FILE *neilogger;
	neilogger = fopen(neighborcsv, "w");
	if (neilogger == NULL)
	{
		printf("Error: Neighbor log file could not be opened for writing.");
		return 1;
	}

	//Load the first frame of coordinates in, and setup the headers for the output files
	fprintf(vellogger, "Time:,");
	fprintf(neilogger, "Time:,");
	*/
	if (is_xtc)
	{
		result_xdr = read_xtc(xd_read, xdr_natoms, &xdr_step, &xdr_time, xdr_box, xdr_coords, &xtc_prec);
	}
	else
	{
		result_xdr = read_trr(xd_read, xdr_natoms, &xdr_step, &xdr_time, &trr_lambda, xdr_box, xdr_coords, trr_vels, trr_fs);
	}
	for (int i = 0; i < xdr_natoms; i++)
	{
		switch (atoms[i].hbondType)
		{
		case 'P':
			protein[atoms[i].hbondListPos].x = xdr_coords[i][0] * 10.0f;
			protein[atoms[i].hbondListPos].y = xdr_coords[i][1] * 10.0f;
			protein[atoms[i].hbondListPos].z = xdr_coords[i][2] * 10.0f;
			break;
		case 'W':
			//fprintf(vellogger, "%i,", waters[atoms[i].hbondListPos].resid);
			//fprintf(neilogger, "%i,", waters[atoms[i].hbondListPos].resid);
			waters[atoms[i].hbondListPos].x = xdr_coords[i][0] * 10.0f;
			waters[atoms[i].hbondListPos].y = xdr_coords[i][1] * 10.0f;
			waters[atoms[i].hbondListPos].z = xdr_coords[i][2] * 10.0f;
			break;
		case 'H': //Ignore hydrogens
			break;
		default:
			printf("ERROR: Internal flag lookup table is mangled somehow! (Found: %c)\n", atoms[i].hbondType);
			return 1;
		}
	}
	//fprintf(vellogger, "\n");
	//fprintf(neilogger, "\n");

	//Keep reading the trajectory file until we hit the end
	int currentframe = 1;
	int framesprocessed = 1;
	long averagetime = 0;
	auto t1 = std::chrono::high_resolution_clock::now();

	auto mindist = new float[numwaters];
	auto nearres = new int[numwaters];
	auto avgvel = new float[numprotein];
	fill(avgvel, avgvel + numprotein, 0);
	auto avginstances = new int[numprotein];
	fill(avginstances, avginstances + numprotein, 0);
	auto prevsimtime = xdr_time;

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
			if (xdr_time == prevsimtime)  //Get around the potential issue that the last frame has the same time as the last-1 frame
				break;
			//fprintf(vellogger, "%f,", xdr_time);
			//fprintf(neilogger, "%f,", xdr_time);
			//Do neighbor analysis
			neighborAnalysisLauncher(nearres, mindist, gpuwater, gpuprotein, numprotein, numwaters, gpumem, deviceProp);
			//Load the new coordinate information into the lists, while calculating velocities
			for (int i = 0; i < xdr_natoms; i++)
			{
				switch (atoms[i].hbondType)
				{
				case 'P':
					protein[atoms[i].hbondListPos].x = xdr_coords[i][0] * 10.0f;
					protein[atoms[i].hbondListPos].y = xdr_coords[i][1] * 10.0f;
					protein[atoms[i].hbondListPos].z = xdr_coords[i][2] * 10.0f;
					break;
				case 'W':
				{
					//TODO: Maybe do this on GPU?  It might be faster
					int currwater = atoms[i].hbondListPos;
					float newx = xdr_coords[i][0] * 10.0f;
					float newy = xdr_coords[i][1] * 10.0f;
					float newz = xdr_coords[i][2] * 10.0f;

					//fprintf(vellogger, "%f,", velocity);
					//fprintf(neilogger, "%i,", protein[nearres[currwater]].resid);
					//If the water is close to a residue atom, calculate it's velocity and save it for averaging
					if (mindist[currwater] < cutoffdist)
					{
						float timespan = xdr_time - prevsimtime;
						float velocity = sqrtf(((waters[currwater].x - newx) * (waters[currwater].x - newx)) +
							((waters[currwater].y - newy) * (waters[currwater].y - newy)) +
							((waters[currwater].z - newz) * (waters[currwater].z - newz))) / timespan;

						avgvel[nearres[currwater]] += velocity;
						++avginstances[nearres[currwater]];
					}

					//Update coordinates
					waters[currwater].x = newx;
					waters[currwater].y = newy;
					waters[currwater].z = newz;
				}
					break;
				case 'H': //Ignore hydrogen
					break;
				default:
					printf("ERROR: Internal hbond lookup table is mangled somehow! (Found flag: %c)\n", atoms[i].hbondType);
					return 1;
				}
			}
			prevsimtime = xdr_time;
			//fprintf(vellogger, "\n");
			//fprintf(neilogger, "\n");
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

	//fclose(vellogger);
	//fclose(neilogger);

	//Release the allocated memory to prevent a memory leak.
	delete[] mindist;
	delete[] xdr_coords;
	if (!is_xtc)
	{
		delete[] trr_vels;
		delete[] trr_fs;
	}

	//Calculate the average velocities
	vector<int> reslist;
	for (int i = 0; i < numprotein; ++i)
	{
		if (find(reslist.begin(), reslist.end(), protein[i].resid) == reslist.end())
		{
			reslist.push_back(protein[i].resid);
		}
	}
	auto resvel = new float[reslist.size()];
	fill(resvel, resvel + reslist.size(), -1.0f);
	for (int i = 0; i < numprotein; ++i)
	{
		if (avginstances[i] != 0)
		{
			int respos = std::distance(reslist.begin(), find(reslist.begin(), reslist.end(), nearres[i]));
			resvel[respos] = avgvel[i] / avginstances[i];
		}
	}

	//Print residue velocities
	FILE *avgvelloger;
	avgvelloger = fopen(avgvelcsv, "w");
	if (avgvelloger == NULL)
	{
		printf("Error: Average residue velocity log file could not be opened for writing.");
		return 1;
	}
	fprintf(avgvelloger, "Res ID:,Velocity:,\n");
	for (int i = 0; i < reslist.size(); ++i)
	{
		fprintf(avgvelloger, "%i,%f,\n", reslist[i], resvel[i]);
	}

	fclose(avgvelloger);
	delete[] resvel;
	delete[] nearres;
	delete[] avgvel;
	delete[] avginstances;

	return 0;
}