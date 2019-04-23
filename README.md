# TSP
Multiple implementations of the Travelling Salesman Problem
Project developed in the course IE555 - Programing for analytics with Dr. Murray


This script will solve a TSP via 4 different methods:  nearest neighbor (NN) heuristic, IP,
simulated annealing (SA), or genetic algorithm.


Inputs:
 	locationsFolder:	For example, practice_3
	objectiveType:		1 --> Minimize Distance, 2 --> Minimize Time
	solveNN				1 --> Solve using NN
	solveIP				1 --> Solve using IP
	solveSA				1 --> Solve using SA
   solveGA             1 --> Solve using GA
 	IPcutoffTime:		-1 --> No time limit, o.w., max number of seconds for Gurobi to run
 	turnByTurn:			1 --> Use MapQuest for detailed routes. 0 --> Just draw straight lines.

 How to run:
 	python new.py practice_25 1 0 0 0 1 120 1
