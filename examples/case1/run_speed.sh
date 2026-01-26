#!/bin/bash

export SPEED_PROGRAM=SPEED
export OUTPUT=speed.out 
export TOTAL_MPI_PROCESSES=1
export THREADS=1
 
let NCORES=$TOTAL_MPI_PROCESSES
echo "Running on $NCORES cores"
echo "Job started at `date`"

mkdir -p VTKOUT FILES_MPI MONITOR POST-PROC 
rm -rf VTKOUT/*
#time mpirun -np $TOTAL_MPI_PROCESSES  -x OMP_NUM_THREADS=$THREADS  $SPEED_PROGRAM >& $OUTPUT
time $SPEED_PROGRAM >& $OUTPUT
wait

