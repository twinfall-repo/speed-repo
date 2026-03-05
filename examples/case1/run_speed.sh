#!/bin/bash

export SPEED_PROGRAM=SPEED
export OUTPUT=speed.out 
export THREADS=1

if [[ -n "$1" ]]; then
	if [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -ge 1 ]]; then
		export TOTAL_MPI_PROCESSES="$1"
	else
		echo "Error: first argument must be a positive integer (TOTAL_MPI_PROCESSES)." >&2
		exit 1
	fi
else
	export TOTAL_MPI_PROCESSES=1
fi
 
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

# Clean the archive files if they exist
rm -f VTKOUT.tar.gz
rm -f MONITOR.tar.gz

time mpirun -np $TOTAL_MPI_PROCESSES  -x OMP_NUM_THREADS=$THREADS  $SPEED_PROGRAM >& $OUTPUT
#time $SPEED_PROGRAM >& $OUTPUT

tar -czf VTKOUT.tar.gz VTKOUT
tar -czf MONITOR.tar.gz MONITOR

wait

