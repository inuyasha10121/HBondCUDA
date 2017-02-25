#include "TrajectoryProcessor.h"

int TrajectoryProcessor::openTrajectory(string trajpath)
{
    //Initial checks to see if we are getting what we expect
    string ext = trajpath.substr(trajpath.find_last_of(".") + 1);
    if (ext != "xtc" && ext != "trr")
    {
        printf("ERROR: File is not a trr or xtc file...\n");
        return 1;
    }

    isxtc = (ext == "xtc");
    char* processedpath = (char*)trajpath.c_str();

    //Setup all the initial junk for reading the file
    

    //Try to open the file for reading
    xd_read = xdrfile_open(processedpath, "r");

    if (xd_read == NULL)
    {
        printf("ERROR: Unable to read trr...\n");
        return 1;
    }

    //Get the number of atoms in the trajectory from the header
    if (isxtc) //Process the file with xtc functions, otherwise use trr functions
    {
        result = read_xtc_natoms(processedpath, &natoms);  //Get the number of atoms
    }
    else
    {
        result = read_trr_natoms(processedpath, &natoms);
    }
    if (result != exdrOK)
    {
        printf("ERROR: Could not read number of atoms from header...\n");
        return 1;
    }
    coords = (rvec *)calloc(natoms, sizeof(coords[0]));
    return 0;
}

void TrajectoryProcessor::closeTrajectory()
{
    xdrfile_close(xd_read);
}

int TrajectoryProcessor::readFrame(vector<Atom> & atoms)
{
    //Make sure out input array matches the atom number in the trajectory
    if (atoms.size() != natoms)
    {
        printf("ERROR: atoms in array does not equal atoms in frame...\n");
        return 1;
    }

    //Use proper reading method
    if (isxtc) 
    {
        result = read_xtc(xd_read, natoms, &step, &time, box, coords, &prec_xtc);
    }
    else
    {
        result = read_trr(xd_read, natoms, &step, &time, &lambda, box, coords, velocities, unknown_f);
    }

    //Make sure we haven't hit the end of the file yet
    if (result == 0)  
    {
        if (exdrOK != result)
        {
            printf("ERROR: Couldn't read trajectory properly...\n");
            return 1;
        }

        //Load in the new coordinate information to the given atom array
        for (int i = 0; i < natoms; i++)
        {
            atoms[i].x = coords[i][0];
            atoms[i].y = coords[i][1];
            atoms[i].z = coords[i][2];
        }
    }
    else
    {
        return -1;  //"Error code" for "We hit the end of the file"
    }

}