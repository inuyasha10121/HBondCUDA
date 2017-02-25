#ifndef TRAJECTORYPROCESSOR_H
#define TRAJECTORYPROCESSOR_H

#include <string>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "GPUTypes.h"
#include "xdrfile.h"
#include "xdrfile_trr.h"
#include "xdrfile_xtc.h"

using namespace std;

class TrajectoryProcessor {
private:
    XDRFILE *xd_read;
    bool isxtc;
    int result;
    int natoms;
    int step;
    float time;
    matrix box;
    rvec *coords, *velocities, *unknown_f; // , *x_trr, *v_trr, *f_trr
    float prec_xtc = 1000.0;
    float lambda = 0.0;

public:
    int TrajectoryProcessor::openTrajectory(string trajpath);
    void TrajectoryProcessor::closeTrajectory();
    int TrajectoryProcessor::readFrame(vector<Atom> & atoms);
};

#endif