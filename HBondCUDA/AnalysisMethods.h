#ifndef ANALYSISNMETHODS_H
#define ANALYSISMETHODS_H

#include "kernel.cuh"

int hbondTrajectoryAnalysis(char * pdbpath, char * hbtpath, char * trjpath, char * outlog, char * outanalysiscsv, char * outbridgercsv, int windowsize, int windowthreshold, int dt, float gpumem, int gpuid, cudaDeviceProp deviceProp);
int hbondTimelineAnalysis(char * outlog, char * outanalysiscsv, char * outbridgercsv, int windowsize, int windowthreshold, int dt, float gpumem, int gpuid, cudaDeviceProp deviceProp);
int velocityAnalysis(char * pdbpath, char * trjpath, char * velocitycsv, char * neighborcsv, char * avgvelcsv, float cutoffdist, int dt, float gpumem, int gpuid, cudaDeviceProp deviceProp);
#endif