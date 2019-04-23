# File:  solve_tsp_3.py
#
# This script will solve a TSP via 3 different methods:  nearest neighbor (NN) heuristic, IP, or simulated annealing (SA)
#
# Inputs:
# 	locationsFolder:	For example, practice_3
#	objectiveType:		1 --> Minimize Distance, 2 --> Minimize Time
#	solveNN				1 --> Solve using NN
#	solveIP				1 --> Solve using IP
#	solveSA				1 --> Solve using SA
# 	IPcutoffTime:		-1 --> No time limit, o.w., max number of seconds for Gurobi to run
# 	turnByTurn:			1 --> Use MapQuest for detailed routes. 0 --> Just draw straight lines between nodes.
#
# How to run:
# 	python solve_tsp_3.py practice_3 1 1 1 1 120 1

import sys			# Allows us to capture command line arguments
import csv
import folium		# https://github.com/python-visualization/folium

import urllib2
import json
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import time
from pandas.io.json import json_normalize
from matplotlib import animation 

from collections import defaultdict

from gurobipy import *


# -----------------------------------------
mapquestKey = '5euWLXfmSp1GXg1mw6Y1tgUdvT87AP11'			# Visit https://developer.mapquest.com/ to get a free API key

# Put your SA parameters here:
Tzero = 1     #Tzero: Initial temperature for SA
I = 40	   #I:Number of iterations per temperature for SA
delta = 0.1	   #delta:Cooling schedule temp reduction for SA
Tfinal = 0        #Tfinal:Minimum allowable temp for SA
SAcutoffTime = 10  #SAcutoffTime: Number of seconds to allow your SA heuristic to run
NumSubtours = 5    #NumSubtours: Number of subtour reversals generated
PlotsFlag = False  #Flag to choose whether to Plot the SA progress
# -----------------------------------------


# http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python
def make_dict():
	return defaultdict(make_dict)

class make_node:
	def __init__(self, nodeName, isDepot, latDeg, lonDeg, demand):
		# Set node[nodeID]
		self.nodeName 	= nodeName
		self.isDepot	= isDepot
		self.latDeg		= latDeg
		self.lonDeg		= lonDeg
		self.demand		= demand

def genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'
	# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)
	routeTypeStr = 'routeType:%s' % transportMode
	
	# Assemble query URL
	myUrl = 'http://www.mapquestapi.com/directions/v2/routematrix?'
	myUrl += 'key={}'.format(mapquestKey)
	myUrl += '&inFormat=json&json={locations:['
	
	# Insert coordinates into the query:
	n = len(coordList)
	for i in range(0,n):
		if i != n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}},'.format(coordList[i][0], coordList[i][1])
		elif i == n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}}'.format(coordList[i][0], coordList[i][1])
	myUrl += '],options:{{{},{},{},{},doReverseGeocode:false}}}}'.format(routeTypeStr, all2allStr,one2manyStr,many2oneStr)
	
	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl
	
	
	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)	
	data = json.loads(response.read())
	
	# print "\nHere's what MapQuest is giving us:"
	# print data

	# This info is hard to read.  Let's store it in a pandas dataframe.
	# We're goint to create one dataframe containing distance information:
	distance_df = json_normalize(data, "distance")
	# print "\nHere's our 'distance' dataframe:"
	# print distance_df	

	# print "\nHere's the distance between the first and second locations:"
	# print distance_df.iat[0,1]	
	
	# Our dataframe is a nice table, but we'd like the row names (indexes)and column names to match our location IDs.
	# This would be important if our locationIDs are [1, 2, 3, ...] instead of [0, 1, 2, 3, ...]
	distance_df.index = locIDlist
	distance_df.columns = locIDlist
		
	# Now, we can find the distance between location IDs 1 and 2 as:
	# print "\nHere's the distance between locationID 1 and locationID 2:"
	# print distance_df.loc[1,2]
		
	
	# We can create another dataframe containing the "time" information:
	time_df = json_normalize(data, "time")

	# print "\nHere's our 'time' dataframe:"
	# print time_df
	
	# Use our locationIDs as row/column names:
	time_df.index = locIDlist
	time_df.columns = locIDlist

	
	# We could also create a dataframe for the "locations" information (although we don't need this for our problem):
	#print "\nFinally, here's a dataframe for 'locations':"
	#df3 = json_normalize(data, "locations")
	#print df3
	
	return(distance_df, time_df)
	
def computeTourCost(objectiveType, distance_df, time_df, tour):
	# print 'Input Tour=', tour
	TourCost = 0
	CurrentNode = tour[0]
	if objectiveType == 1:
		for i in range(1, len(tour)):
			TourCost += distance_df.loc[CurrentNode,tour[i]]
			CurrentNode = tour[i]
	else:
		for i in range(1, len(tour)):
			TourCost += time_df.loc[CurrentNode,tour[i]]
			CurrentNode = i
	return TourCost

def subtourReversal(objectiveType, distance_df, time_df,tour,NumSubtours):
	# Generate 5 subtours and draw the best.
	ReversedTours = []
	CostofTours = []
	# Generate 5 distinct ReversedTours
	while len(ReversedTours) < NumSubtours:
		#Generate subtour reversal
		# raw_input('Press Enter to Continue\n')
		# print 'InputTour', tour
		a = rd.randint(1, len(tour)-3)
		b = rd.randint(a+1, len(tour)-2)
		body1 = list(tour[a:b+1])
		head1 = list(tour[0:a])
		tail1 = list(tour[b+1:])
		reversedTour = head1 + body1[::-1] + tail1
		# print 'reversedTour', reversedTour
		# # Test if the the generated tour was not generated before
		if reversedTour not in ReversedTours:
			ReversedTours.append(reversedTour)
			CostofTours.append(computeTourCost(objectiveType, distance_df, time_df, reversedTour))
		
	# Find the one with best cost
	
	minCostTour = CostofTours.index(min(CostofTours))
	BestTour = list(ReversedTours[minCostTour])
	# print 'ReversedTours = ', ReversedTours
	# print 'BestTour=', BestTour
	return BestTour

def genShapepoints(startCoords, endCoords):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'		# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)

	# assemble query URL 
	myUrl = 'http://www.mapquestapi.com/directions/v2/route?key={}&routeType={}&from={}&to={}'.format(mapquestKey, transportMode, startCoords, endCoords)
	myUrl += '&doReverseGeocode=false&fullShape=true'

	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl
	
	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)	
	data = json.loads(response.read())
	
	# print "\nHere's what MapQuest is giving us:"
	# print data
		
	# retrieve info for each leg: start location, length, and time duration
	lats = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lat'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	lngs = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lng'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	secs = [data['route']['legs'][0]['maneuvers'][i]['time'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	dist = [data['route']['legs'][0]['maneuvers'][i]['distance'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]

	# print "\nHere are all of the lat coordinates:"
	# print lats

	# create list of dictionaries (one dictionary per leg) with the following keys: "waypoint", "time", "distance"
	legs = [dict(waypoint = (lats[i],lngs[i]), time = secs[i], distance = dist[i]) for i in range(0,len(lats))]

	# create list of waypoints (waypoints define legs)
	wayPoints = [legs[i]['waypoint'] for i in range(0,len(legs))]
	# print wayPoints

	# get shape points (each leg has multiple shapepoints)
	shapePts = [tuple(data['route']['shape']['shapePoints'][i:i+2]) for i in range(0,len(data['route']['shape']['shapePoints']),2)]
	# print shapePts
					
	return shapePts

def solve_tsp_nn(objectiveType, distance_df, time_df, node):
	# We're going to find a route and a cost for this route
	nn_route = []
	
	# Create a list of locations we haven't visited yet, 
	# excluding the depot (which is our starting/ending location)
	unvisitedLocations = []
	startNode = -1
	for nodeID in node:
		if (node[nodeID].isDepot):
			startNode = nodeID
		else:	
			unvisitedLocations.append(nodeID) 

	if (startNode == -1):
		# We didn't find a depot in our list of locations.
		# We'll let our starting node be the first node
		startNode = min(node)
		print "I couldn't find a depot.  Starting the tour at locationID %d" % (startNode)

		
	# From our starting location, find the nearest unvisited location.
	# We'll also keep track of the total distance (or time)
	i = startNode		# Here's where our tour starts
	nn_route.append(i)
	nn_cost = 0.0

	while len(unvisitedLocations) > 0:
		# Find the nearest neighbor to customer i
		if (objectiveType == 1):
			# Find the location with the minimum DISTANCE to location 1
			# (excluding any visited locations).
			# Break ties by choosing the location with the smallest ID.
			closestNeighbor = distance_df.loc[i][distance_df.loc[i,:] == distance_df.loc[i,unvisitedLocations].min()].index.values[0]
			cost2neighbor = distance_df.loc[i][distance_df.loc[i,:] == distance_df.loc[i,unvisitedLocations].min()].values[0]

		else:
			# Find the location with the minimum TIME to location 1
			# (excluding any visited locations).
			# Break ties by choosing the location with the smallest ID.
			closestNeighbor = time_df.loc[i][time_df.loc[i,:] == time_df.loc[i,unvisitedLocations].min()].index.values[0]
			cost2neighbor = time_df.loc[i][time_df.loc[i,:] == time_df.loc[i,unvisitedLocations].min()].values[0]
		# Add this neighbor to our tour:
		nn_route.append(closestNeighbor)
		
		# Update our tour cost thus far:
		nn_cost += cost2neighbor
		
		# Remove the closestNeighbor from our list of unvisited locations:
		unvisitedLocations.remove(closestNeighbor)
		
		# Virtually move our salesperson to the neighbor we just found:
		i = closestNeighbor
		
	# Finish the tour by returning to the start location:
	nn_route.append(startNode)	
	if (objectiveType == 1):
		nn_cost += distance_df.loc[i,startNode]
	else:
		nn_cost += time_df.loc[i,startNode]
							
	return (nn_route, nn_cost)
	
def solve_tsp_ip(objectiveType, distance_df, time_df, node, cutoffTime):
	ip_route = []
	
	N = []
	q = 0
	for nodeID in node:
		N.append(nodeID)
		if (node[nodeID].isDepot == 0):
			q += 1

	c = defaultdict(make_dict)
	decvarx = defaultdict(make_dict)
	decvaru = {}
	

	# GUROBI
	m = Model("TSP")

	# The objective is to minimize the total travel distance.
	m.modelSense = GRB.MINIMIZE
		
	# Give Gurobi a time limit:
	if (cutoffTime > 0):
		m.params.TimeLimit = cutoffTime
	
	# Define our decision variables (with their objective function coefficients:			
	for i in N:
		decvaru[i] = m.addVar(lb=1, ub=q+2, obj=0, vtype=GRB.CONTINUOUS, name="u.%d" % (i))
		for j in N:
			if (i != j):
				if (objectiveType == 1):
					# We want to minimize distance
					decvarx[i][j] = m.addVar(lb=0, ub=1, obj=distance_df.loc[i,j], vtype=GRB.BINARY, name="x.%d.%d" % (i,j))
				else:
					# We want to minimize time
					decvarx[i][j] = m.addVar(lb=0, ub=1, obj=time_df.loc[i,j], vtype=GRB.BINARY, name="x.%d.%d" % (i,j))

	# Update model to integrate new variables:
	m.update()
	
	# Build our constraints:
	# Constraint (2)
	for i in N:
		m.addConstr(quicksum(decvarx[i][j] for j in N if j != i) == 1, "Constr.2.%d" % i)
		
	# Constraint (3)
	for j in N:
		m.addConstr(quicksum(decvarx[i][j] for i in N if i != j) == 1, "Constr.3.%d" % j)
	
	# Constraint (4)
	for i in range(1, q+1):
		for j in N:
			if (j != i):
				m.addConstr(decvaru[i] - decvaru[j] + 1 <= (q + 1)*(1 - decvarx[i][j]), "Constr.4.%d.%d" % (i,j))
				
	# Solve
	m.optimize()


	if (m.Status == GRB.INFEASIBLE):
		# NO FEASIBLE SOLUTION EXISTS
		
		print "Sorry, Guroby is telling us this problem is infeasible."
		
		ip_cost = -999	# Infeasible
	
	elif ((m.Status == GRB.TIME_LIMIT) and (m.objVal > 1e30)):
		# NO FEASIBLE SOLUTION WAS FOUND (maybe one exists, but we ran out of time)

		print "Guroby can't find a feasible solution.  It's possible but one exists."
		
		ip_cost = -888	# Possibly feasible, but no feasible solution found.
			
	else:
		# We found a feasible solution
		if (m.objVal == m.ObjBound):
			print "Hooray...we found an optimal solution."
			print "\tOur objective function value:  %f" % (m.objVal)
		else:
			print "Good News:  We found a feasible solution."
			print "Bad News:  It's not provably optimal."
			print "\tOur objective function value:  %f" % (m.objVal)
			print "\tGurobi's best bound: %f" % (m.ObjBound)

		if (m.Status == GRB.TIME_LIMIT):
			print "\tGurobi reached it's time limit"


		# Let's use the values of the x decision variables to create a tour:
		startNode = -1
		for nodeID in node:
			if (node[nodeID].isDepot):
				startNode = nodeID
	
		if (startNode == -1):
			# We didn't find a depot in our list of locations.
			# We'll let our starting node be the first node
			startNode = min(node)
			print "I couldn't find a depot.  Starting the tour at locationID %d" % (startNode)

		allDone = False
		i = startNode
		ip_route.append(i)
		while (not allDone):
			for j in N:
				if (i != j):
					if (decvarx[i][j].x > 0.9):
						# We traveled from i to j
						ip_route.append(j)
						i = j
						break	# We found the link from i to j.  Stop the "for" loop
			if (len(ip_route) == len(N)):
				# We found all the customers
				allDone = True
		
		# Now, add a link back to the start
		ip_route.append(startNode)
		
		ip_cost = m.objVal
	
	return (ip_route, ip_cost)

def solve_tsp_sa(objectiveType, distance_df, time_df,node):
	# Procedure
	[nn_route, nn_cost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
	Xinitial = list(nn_route)
	Xcurr = list(Xinitial)
	Zcurr = nn_cost
	Tcurr = Tzero
	Xbest = list(Xcurr)
	Zbest = Zcurr
	itnum = 0
	timeout = time.time() + SAcutoffTime
	timecount = time.time()
	allCurrSol = []

	while (Tcurr > Tfinal):
		plt.figure(1)
		for i in range(1,I):
			itnum += 1
			# TemperatureP, = plt.plot(itnum,Tcurr,'ko') 
			CandidateTour = subtourReversal(objectiveType, distance_df, time_df,Xbest,NumSubtours)
			CandidateTourCost = computeTourCost(objectiveType, distance_df, time_df, CandidateTour)
			if (CandidateTourCost < Zcurr):
				Xcurr = list(CandidateTour)
				Zcurr = CandidateTourCost
				DirectAccept, = plt.plot(itnum,CandidateTourCost,'bo-') # Plot the accepted solutions
			else:
				# print 'Did Not accept directly'
				# print 'CandidateTourCost=', CandidateTourCost
				DeltaC = CandidateTourCost - Zcurr
				if rd.random() <= np.exp(-DeltaC/Tcurr):
					# print 'Accepted worse solution on the prob test'
					Xcurr = list(CandidateTour)
					Zcurr = CandidateTourCost
					WorseAccept, = plt.plot(itnum,CandidateTourCost,'rD') #Worse But accept
				else:
					# print 'Rejected worse on the probability test'
					WorseReject, = plt.plot(itnum,CandidateTourCost,'g^',markersize=3) # Worse and rejected
			allCurrSol.append(Zcurr)
			if Zcurr < Zbest:
				Zbest = Zcurr
				Xbest = list(Xcurr)
	# Update the temperature
		Tcurr -= delta
		# Check the stopping criteria
		if (time.time() > timeout):
			print 'Exceeded the specified runtime... Exiting now'
			break

		# raw_input('Press Enter to Continue\n')
	if PlotsFlag:
		plt.legend([DirectAccept,WorseAccept,WorseReject], ["Directly accepted solutions","Worse but accepted solutions", "Worse and rejected solutions"])
		plt.title('SA progress')
		plt.ylabel('Objective Function Value')
		plt.xlabel('Total Iteration Number')
		# plt.figure(2)
		# plt.plot(allCurrSol,'g-')
		# plt.title('SA progress 2 - Z_curr')
		# plt.ylabel('Objective Function Value')
		# plt.xlabel('Total Iteration Number')
		# plt.figure(3)
		# plt.plot(allCurrSol,'g-')
		# plt.title('SA progress 3 - Best Values')
		# plt.ylabel('Objective Function Value')
		# plt.xlabel('Total Iteration Number')
		plt.show()

	sa_route = list(Xbest)
	sa_cost = computeTourCost(objectiveType, distance_df, time_df, Xbest)
	print "Total Runtime for the SA", (time.time() - timecount)	
	# print "Total Iterations", itnum
	# print "SA Cost ", sa_cost 
	return (sa_route, sa_cost)

def solve_tsp_ga(objectiveType, distance_df, time_df,node):
	# Define Parameters of the GA
	MaxGenerations = 50	# Maximun number of generations 400
	PopSize = 100  			# Population size 800
	MutationRate = 0.01		# Mutation probability
	CrossOverRate = 0.70	# Crossover probability

	ProbSize = len(distance_df)
	ZBest = -1e30
	XBest = []

	# Make the population size an even number
	if (PopSize % 2) > 0:
		PopSize +=1

	def GeneratePopulation(ProbSize,PopSize):
		# Generate population using random permutations
		Population = []
		for component in range(PopSize):
			temp = range(1,ProbSize)
			individual = rd.sample(temp, len(temp))
			Population.append(individual)
		# print Population
		return Population

	def GeneratePopulation2(nn_route,PopSize,objectiveType, distance_df, time_df, node):
		# Function to generate initial population based on the Nearest Neighbour Solution
		[nn_route, nn_cost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
		del nn_route[0]
		del nn_route[-1]
		Population = []
		for ind in range(PopSize):
			newInd = Mutate(nn_route)
			Population.append(newInd)
		return Population
	
	def Fitness(indiv, objectiveType, distance_df, time_df):
		# Compute Fitness of each component
		tempIndiv = list(indiv)
		tempIndiv = [0] + tempIndiv + [0] #Remove the node depot for simplifying the calculations
		tempCost = computeTourCost(objectiveType, distance_df, time_df, tempIndiv)
		tempCost = 1.0/tempCost
		return tempCost

	def PMXCrossover(Mother, Father):
		# Crossover following the heuristic: D.E. Goldberg and R. Lingle, Alleles, loci and the traveling salesman problem, 1985
		crossPoint = rd.randint(1, len(Parents[1])-1)
		Child1 = list(Mother)
		Child2 = list(Father)
		for i in range(0, crossPoint):
			# Swap the components
			TempChild1 = list(Child1)
			TempChild1[i] = Father[i]
			TempChild2 = list(Child2)
			TempChild2[i] = Mother[i]
			# Correct the child so that the tour is valid
			for j in range(0,len(Mother)):
				if j != i:
					if TempChild1[i] == TempChild1[j]:
						TempChild1[j] = Child1[i]
						break
			Child1 = list(TempChild1)
			for j in range(0,len(Mother)):
				if j != i:
					if TempChild2[i] == TempChild2[j]:
						TempChild2[j] = Child2[i]
						break
			Child2 = list(TempChild2)
		
		return Child1, Child2	

	def Mutate(tour):
		# Generate a subtour reversal and draw the best.
		a = rd.randint(0, len(tour)-2)
		b = rd.randint(a+1, len(tour)-1)
		body1 = list(tour[a:b+1])
		head1 = list(tour[0:a])
		tail1 = list(tour[b+1:])
		reversedTour = head1 + body1[::-1] + tail1
		return reversedTour

	def Roulette(ZCumm, EvaluatedPop):
		# http://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
		Roulette = rd.uniform(0, ZCumm)
		tempsum = 0
		for indiv in EvaluatedPop:
			tempsum += indiv[1]
			if tempsum > Roulette:
				selected = indiv[0]
				break
		return selected
	
	################ Generate initial Population ################ 
	Population = GeneratePopulation(ProbSize,PopSize)
	# Population = GeneratePopulation2(nn_route,PopSize)
	
	################ Starts the Process #########################
	# plt.ion()
	fig1 = plt.figure()
	# ax1 = fig1.add_subplot(211)
	ax2 = fig1.add_subplot(212)

	# ax1.set_xlim([0, MaxGenerations])
	# ax1.set_ylim([0, 200])
	# ax1.set_ylabel('Average OFV')
	# ax1.set_title('Average Population')
	ax2.set_xlim([0, MaxGenerations])
	ax2.set_ylim([0, 200])
	ax2.set_xlabel("Generation")
	ax2.set_ylabel('Best OFV')
	ax2.set_title('Best Individual in each generation')
	global PopAverage, ZBestIndiv 
	def init():
	    # ax1.plot([], [])
	    ax2.plot([], [])
	    # return ax1,ax2
	    return ax2
	    
	def animate(i):
		global PopAverage, ZBestIndiv 
		# ax1.plot(i,1.0/PopAverage)
		ax2.plot(i,1.0/ZBestIndiv)
		# return ax1,ax2
		return ax2
		
	for i in range(MaxGenerations):
		######## Compute Statistics of the population ##########
		EvaluatedPop = []
		ZCumm = 0
		for indiv in Population:
			ZIndiv = Fitness(indiv, objectiveType, distance_df, time_df)
			ZCumm += ZIndiv
			# Place the the pair ( X, Z(X) ) in a tuple
			EvaluatedPop.append((indiv, ZIndiv))
		
		BestIndiv_ = max(EvaluatedPop, key = lambda t: t[1])
		BestIndiv = BestIndiv_[0] 
		ZBestIndiv = BestIndiv_[1]
		PopAverage = ZCumm/PopSize

		# Retain the best element
		if ZBestIndiv > ZBest:
			ZBest = ZBestIndiv
			XBest = [0] + list(BestIndiv) + [0] # Add the depot node back to the solution

		# Print and Plot the stats (curr population)
		print 'PopAverage=', 1.0/PopAverage


		anim = animation.FuncAnimation(fig1, animate, init_func=init,
                               frames=200, interval=20, blit=True)
		anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

		plt.show()

		# ax2.plot(i,1.0/ZBestIndiv,'bo')
		# ax1.plot(i,1.0/PopAverage,'ko')
		print 'ZBestIndiv=', 1.0/ZBestIndiv

		############# Selection #################
		Parents = []
		while len(Parents) < PopSize:
			Par = Roulette(ZCumm, EvaluatedPop)
			Parents.append(Par)

		############## Crossover ################
		newPopulation = []
		while len(newPopulation) < PopSize:
			for i in range(0,len(Parents),2):
				if Parents[i] != Parents[i+1]:
					if rd.random() < CrossOverRate:
						Child1, Child2 = PMXCrossover(Parents[i], Parents[i+1])
						# print 'Crossover'
					else:
						Child1 = Parents[i]
						Child2 = Parents[i+1]
					newPopulation.append(Child1)
					newPopulation.append(Child2)
				
		############## Mutation ################
		for indiv in range(0,len(newPopulation)):
			if MutationRate > rd.random():
				newPopulation[indiv] = Mutate(newPopulation[indiv]) 
				# print 'Mutate'
		Population = list(newPopulation)
		
		# plt.pause(0.0005)

	print 1.0/ZBest
	print XBest

	ga_cost = -1
	ga_route = []
	return (ga_route, ga_cost)


# Capture command line inputs:
if (len(sys.argv) == 8):
	locationsFolder		= str(sys.argv[1])		# Ex:  practice_3
	objectiveType		= int(sys.argv[2])		# 1 --> Minimize Distance, 2 --> Minimize Time
	solveNN				= int(sys.argv[3])		# 1 --> Solve NN.  0 --> Don't solve NN
	solveIP				= int(sys.argv[4])		# 1 --> Solve IP.  0 --> Don't solve IP
	solveSA				= int(sys.argv[5])		# 1 --> Solve SA.  0 --> Don't solve SA	
	IPcutoffTime		= float(sys.argv[6])	# -1 --> No time limit, o.w., max number of seconds for Gurobi to run
	turnByTurn			= int(sys.argv[7])		# 1 --> Use MapQuest for detailed routes.  0 --> Just draw straight lines connecting nodes.
	if (objectiveType not in [1,2]):
		print 'ERROR:  objectiveType %d is not recognized.' % (objectiveType)
		print 'Valid numeric options are:  1 (minimize distance) or 2 (minimize time)'
		quit()
else:
	print 'ERROR: You passed', len(sys.argv)-1, 'input parameters.'
	quit()


# Initialize a dictionary for storing all of our locations (nodes in the network):
node = {}


# Read location data
locationsFile = 'Problems/%s/tbl_locations.csv' % locationsFolder
# Read problem data from .csv file
# NOTE:  We are assuming that the .csv file has a pre-specified format.
#	 Column 0 -- nodeID
# 	 Column 1 -- nodeName
#	 Column 2 -- isDepot (1 --> This node is a depot, 0 --> This node is a customer
#	 Column 3 -- lat [degrees]
#	 Column 4 -- lon [degrees]
#	 Column 5 -- Customer demand
with open(locationsFile, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
		if (row[0][0] != '%'):
			nodeID = int(row[0])
			nodeName = str(row[1])
			isDepot = int(row[2])
			latDeg = float(row[3])
			lonDeg = float(row[4])
			demand = float(row[5])
	
			node[nodeID] = make_node(nodeName, isDepot, latDeg, lonDeg, demand)

# Use MapQuest to generate two pandas dataframes.
# One dataframe will contain a matrix of travel distances, 
# the other will contain a matrix of travel times.
coordList = []
locIDlist = []
for i in node:
	coordList.append([node[i].latDeg, node[i].lonDeg])
	locIDlist.append(i)

all2allStr	= 'allToAll:true' 
one2manyStr	= 'oneToMany:false'
many2oneStr	= 'manyToOne:false'
	
[distance_df, time_df] = genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr)
[ga_cost,ga_route] = solve_tsp_ga(objectiveType, distance_df, time_df,node)
# [ip_route, ip_cost] = solve_tsp_ip(objectiveType, distance_df, time_df, node, IPcutoffTime)

quit()

# print len(time_df)
# print
# print len(distance_df)
# quit()
# solutions = [] 
# for runnum in range(3):
# [sa_route, sa_cost] = solve_tsp_sa(objectiveType, distance_df, time_df,node)
# minCostTour = CostofTours.index(min(CostofTours))
# BestSA = list(ReversedTours[minCostTour])

# solutions.append(sa_cost)
# print solutio
# print 'SA Solution'
# print sa_cost
# quit()
# # print ''
# quit()

# Now, solve the TSP:
[nn_route, ip_route, sa_route, nn_cost, ip_cost, sa_cost] = [[], [], [], -1, -1, -1]

if (solveNN):
	# Solve the TSP using nearest neighbor
	[nn_route, nn_cost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
if (solveIP):
	# Solve the TSP using the IP model
	[ip_route, ip_cost] = solve_tsp_ip(objectiveType, distance_df, time_df, node, IPcutoffTime)
if (solveSA):
	# Solve the TSP using simulated annealing
	[sa_route, sa_cost] = solve_tsp_sa(objectiveType, distance_df, time_df,node)

# Create a map of our solution
mapFile = 'Problems/%s/osm.html' % locationsFolder
map_osm = folium.Map(location=[node[0].latDeg, node[0].lonDeg], zoom_start=10)

# Plot markers
for nodeID in node:
	if (node[nodeID].isDepot):		
		folium.Marker([node[nodeID].latDeg, node[nodeID].lonDeg], icon = folium.Icon(color ='red'), popup = node[nodeID].nodeName).add_to(map_osm)
	else:
		folium.Marker([node[nodeID].latDeg, node[nodeID].lonDeg], icon = folium.Icon(color ='blue'), popup = node[nodeID].nodeName).add_to(map_osm)
		
if (turnByTurn):
	# (PRETTY COOL) Plot turn-by-turn routes using MapQuest shapepoints:
	if (nn_cost > 0):
		# a) nearest neighbor:
		i = nn_route[0]
		for j in nn_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)
			
			myShapepoints = genShapepoints(startCoords, endCoords)	       
		
			folium.PolyLine(myShapepoints, color="blue", weight=8.5, opacity=0.5).add_to(map_osm)	
	
			i = j
			
	if (ip_cost > 0):
		# b) ip:
		i = ip_route[0]
		for j in ip_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)
			
			myShapepoints = genShapepoints(startCoords, endCoords)	       
		
			folium.PolyLine(myShapepoints, color="green", weight=8.5, opacity=0.5).add_to(map_osm)	
	
			i = j

	if (sa_cost > 0):
		# c) simmulated annealing:
		i = sa_route[0]
		for j in sa_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)
			
			myShapepoints = genShapepoints(startCoords, endCoords)	       
		
			folium.PolyLine(myShapepoints, color="red", weight=4.5, opacity=0.5).add_to(map_osm)	
	
			i = j
else:
	# (BORING) Plot polylines connecting nodes with simple straight lines:
	if (nn_cost > 0):
		# a) nearest neighbor:
		points = []
		for nodeID in nn_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="red", weight=8.5, opacity=0.5).add_to(map_osm)	
	if (ip_cost > 0):
		# b) ip:
		points = []
		for nodeID in ip_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="green", weight=4.5, opacity=0.5).add_to(map_osm)	
	if (sa_cost > 0):
		# b) simmulated annealing:
		points = []
		for nodeID in sa_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="green", weight=4.5, opacity=0.5).add_to(map_osm)		
map_osm.save(mapFile)

print "\nThe OSM map is saved in: %s" % (mapFile)

if (solveNN):
	print "\nNearest Neighbor Route:"
	print nn_route
	print "Nearest Neighbor 'Cost':"
	print nn_cost
if (solveIP):	
	print "\nIP Route:"
	print ip_route
	print "IP 'cost':"
	print ip_cost
if (solveSA):	
	print "\nSA Route:"
	print sa_route
	print "SA 'cost':"
	print sa_cost









































