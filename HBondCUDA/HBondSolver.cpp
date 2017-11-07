#include <string>

#include "AnalysisMethods.h"
#include "helper_string.h"
#include "CalculationMethods.h"
#include "PDBProcessor.h"
#include "CSVReader.h"

using namespace std;

//---------------------------------------------MAIN CODE BODY---------------------------------------------
int main(int argc, char **argv)
{   
    setbuf(stdout, NULL);
    //Command line arguements to make the code a bit more versatile
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h") ||
        argc < 1)
    {
        cout << "Usage: " << argv[0] << endl;
		cout << "      -analysistype <bridgertimeline, bridgeanalysisonly, velocity>: Specifies analysis type (Default \"bridgertimeline\")" << endl;
		cout << "NOTE: All parameters are mandatory unless designted \"Optional\"" << endl;
		cout << endl;

		cout << "COMMON AMONGST ALL ANALYSIS TYPES:" << endl;
		cout << "      -gpumem=<FLOAT> : Percentage of total per-graphics card memory allowed to be used (Optional, 0 to 1, Default 0.15)" << endl;
		cout << "      -gpuid=<INT> : ID value of which graphics card to use (Optional, Default 0)" << endl;
		cout << "      -dt=<INT> : Frame skip parameter (Optional, Default 1)" << endl;

		cout << endl;
		cout << "ANALYSIS TYPE: hbondtimeline" << endl;
		cout << "Performs analysis on a trajectory for hydrogen bond behavior" << endl;
		cout << "      -pdb = PDB File Path" << endl;
		cout << "      -hbt : Hydrogen Bond Lookup Table" << endl;
		cout << "      -trj : Trajectory Path" << endl;
		cout << "      -ol : Outpath for Hydrogen Bond Timeline File" << endl << endl;
		cout << "      -oa : Outpath for Analysis CSV file" << endl;
		cout << "      -ob : Outpath for Bridger CSV file" << endl;
		cout << "      -window=<INT> : Window frame size for bond analysis (Optional, Default 5)" << endl;
		cout << "      -wint=<INT> : Window threshold for bond analysis (Optional, Default 4)" << endl;


		cout << endl;
		cout << "ANALYSIS TYPE: hbondanalysisonly" << endl;
		cout << "Performs analysis on a pre-made timeline log from \"hbondtimeline\" analysis" << endl;
		cout << "      -ol : Input path for Hydrogen Bond Timeline File" << endl << endl;
		cout << "      -oa : Outpath for Analysis CSV file" << endl;
		cout << "      -ob : Outpath for Bridger CSV file" << endl;
		cout << "      -window=<INT> : Window frame size for bond analysis (Optional, Default 5)" << endl;
		cout << "      -wint=<INT> : Window threshold for bond analysis (Optional, Default 4)" << endl;

		cout << endl;
		cout << "Performs analysis based on water velocities" << endl;
		cout << "ANALYSIS TYPE: velocity" << endl;
		cout << "      -pdb = PDB File Path" << endl;
		cout << "      -trj : Trajectory Path" << endl;
		cout << "      -vl : Outpath for velocity based CSV log" << endl;
		cout << "      -nl : Outpath for neighbor CSV log" << endl;
		cout << "      -al : Outpath for average velocity near protein residue" << endl;
		cout << "      -vc=<FLOAT> : Cutoff distance for considering a water \"close\" to a residue. (Optional, default 5.0)" << endl;

		cout << endl;
        return 0;
    }

	char * analysistype = "hbondtimeline";
	float cudaMemPercentage = 0.75f;
	int gpuid = 0;
	int dt = 1;

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

	if (checkCmdLineFlag(argc, (const char**)argv, "analysistype"))
	{
		getCmdLineArgumentString(argc, (const char**)argv, "analysistype", &analysistype);
	}


	if (strcmp(analysistype , "hbondtimeline") == 0)
	{
		char * pdbpath;
		char * hbondtablepath;
		char * trajpath;
		char * outpath;
		char * csvpath;
		char * bridgerpath;
		int hbondwindow = 5; //MUST BE ODD
		int windowthreshold = 4; //Inclusive

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

		if (checkCmdLineFlag(argc, (const char**)argv, "ob"))
		{
			getCmdLineArgumentString(argc, (const char**)argv, "ob", &bridgerpath);
		}
		else
		{
			cout << "An output bridger log (-ob) file MUST be specified." << endl;
			cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

			return 1;
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

	}
	else if (strcmp(analysistype, "hbondanalysisonly") == 0)
	{
		char * outpath;
		char * csvpath;
		char * bridgerpath;
		int hbondwindow = 5; //MUST BE ODD
		int windowthreshold = 4; //Inclusive

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

		if (checkCmdLineFlag(argc, (const char**)argv, "ob"))
		{
			getCmdLineArgumentString(argc, (const char**)argv, "ob", &bridgerpath);
		}
		else
		{
			cout << "An output bridger log (-ob) file MUST be specified." << endl;
			cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

			return 1;
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
		hbondTimelineAnalysis(outpath, csvpath, bridgerpath, hbondwindow, windowthreshold, dt, cudaMemPercentage, gpuid, deviceProp);
	}
	else if (strcmp(analysistype, "velocity") == 0)
	{
		char * pdbpath;
		char * trajpath;
		char * velcsv;
		char * neighborcsv;
		char * avgcsv;
		float velcutoff = 5.0f;

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
		if (checkCmdLineFlag(argc, (const char**)argv, "vl"))
		{
			getCmdLineArgumentString(argc, (const char**)argv, "vl", &velcsv);
		}
		else
		{
			cout << "A velocity CSV file (-vl) MUST be specified." << endl;
			cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

			return 1;
		}
		if (checkCmdLineFlag(argc, (const char**)argv, "nl"))
		{
			getCmdLineArgumentString(argc, (const char**)argv, "nl", &neighborcsv);
		}
		else
		{
			cout << "A neighbor CSV file (-nl) MUST be specified." << endl;
			cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

			return 1;
		}
		if (checkCmdLineFlag(argc, (const char**)argv, "al"))
		{
			getCmdLineArgumentString(argc, (const char**)argv, "al", &avgcsv);
		}
		else
		{
			cout << "An average residue velocity CSV file (-al) MUST be specified." << endl;
			cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

			return 1;
		}

		if (checkCmdLineFlag(argc, (const char**)argv, "vc"))
		{
			velcutoff = getCmdLineArgumentFloat(argc, (const char**)argv, "vc");
		}

		velocityAnalysis(pdbpath, trajpath, velcsv, neighborcsv, avgcsv, velcutoff, dt, cudaMemPercentage, gpuid, deviceProp);
	}
	else
	{
		cout << "Error: Analysis type not recognized (Must be \"bridgertimeline\", \"bridgeranalysisonly\", or \"velocity\")" << endl;
		return 1;
	}
}