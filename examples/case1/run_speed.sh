#!/bin/bash

export SPEED_PROGRAM=SPEED
export OUTPUT=speed.out 
export TOTAL_MPI_PROCESSES=4
export THREADS=1
 
let NCORES=$TOTAL_MPI_PROCESSES
echo "Running on $NCORES cores"
echo "Job started at `date`"

# Create output directories if they don't exist
mkdir -p VTKOUT FILES_MPI MONITOR POST-PROC 

# Clean up previous output files
rm -rf VTKOUT/*
rm -rf FILES_MPI/*
rm -rf MONITOR/*
rm -rf POST-PROC/*

time mpirun -np $TOTAL_MPI_PROCESSES  -x OMP_NUM_THREADS=$THREADS  $SPEED_PROGRAM >& $OUTPUT
#time $SPEED_PROGRAM >& $OUTPUT
wait

