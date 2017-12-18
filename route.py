#!/usr/bin/env python
"""
Iplemented the following cost functions:
Distance
Time
Segments
Longtour

Answers to question 1 
1. The uniform cost search turned out to be the fastest for all three costs distance, time and segments and also gave the most optimal route for the selected cost
	Astar also gave the most optimal route but it was slower than uniform vecause of heuristic computations in the code. 
	
2. A sample test case was tried. start city = Bloomington,_Indiana and end city = Alton,_California and distance as the cost:
	The resulting time in seconds are shown 
	astar	uniform  bfs	 dfs
	4.14	3.59	 3.88	 6.08
	Uniform cost search was the fastest followed by bfs astar and dfs. However one important point to be noted here is that even though astar took a 
	long time, it searched through fewer states (100 - 200 states fewer)than uniform. Thus it can be inferred that heuristic does lessen than number of 
	search states. However the heristic calculation itself slowed down the  code.
	
	According to the discussions in the class DFS may converge faster than BFS. But it is not so because DFS may spend time exprloring unnecessary states 
	for this problem.
  
   
3. The memory usage for the four algorithms  in bytes is shown below. The same test case as above was used.
	astar	    uniform	     bfs	        dfs
	81690624	81735680	81477632	84062208
	
	bfs used the least memory followed by astar uniform and dfs. 


4. The following heuristic was used if cost is distance:
	(haversine distance between the city and the end city )/10 
	Since only 1/10 of the distance is used it is an admissible heuristic and will never over estimate the cost to goal which is the sum of all road lengths from the city to end city.
	Roads are not built in the linear fashion and the summation  of road lengths will be greater than the actual haversine distance. 
	Hence one tenth of haversine distance will always be an underestimation.
	In case of junctions,
	(heuristic of it's parent city - distance from parent city) /10 is used. 
	
	for time as the cost function:
	(haversine distance to the end city/65) - 1 is used.
	65 is the max speed in the road segments file. Hence 65 is used. In order to prevent overestimtion 1 is subtracted for the time thus obtained. 
	
	for segments as the cost function:
	The heuristic is taken as zero. Since the number of segments to the end city is atleast 1 a heuristic of zero is admissible. 
"""
import pandas as pd
import os
import math as mt
# Create the Graph for search
from collections import defaultdict
import sys
import time
import numpy as np
import copy
import heapq

t0 = time.time()
city_gps=pd.read_csv("city-gps.txt", sep=" ", names=["City_name", "Lat","Long"])
road_segs = pd.read_csv("road-segments.txt", sep=" ", names=["first_city","Second_city" ,"Length","Speed", "Name of Highway"])

start_city = sys.argv[1]

end_city = sys.argv[2]

cost = sys.argv[4]

routing_algo = sys.argv[3]


# Replace NaN
road_segs["Speed"] = road_segs["Speed"].fillna(max(road_segs["Speed"]))

#replace zero values
road_segs[road_segs["Speed"]==0] = road_segs[road_segs["Speed"]==0].replace(0,max(road_segs["Speed"]))


def distance_calc(gps_1, gps_2):
    #Function to calculate the distance given coordinates
    gps_1_rad = [ mt.radians(item) for item in gps_1]
    gps_2_rad = [ mt.radians(item) for item in gps_2]
    lon = gps_2_rad[1]- gps_1_rad[1]
    lat = gps_2_rad[0]- gps_1_rad[0]
    x = mt.sin(lat/2)**2 + mt.cos(gps_1_rad[0]) * mt.cos(gps_2_rad[0]) * mt.sin(lon/2)**2
    y = 2 * mt.atan2(mt.sqrt(x), mt.sqrt(1-x))
    rad = 3956 
    return y * rad
  
dist = []
for cities in zip(road_segs[road_segs["Length"]==0].first_city, road_segs[road_segs["Length"]==0].Second_city):
    if np.size(city_gps[city_gps["City_name"]==cities[0]].Lat.values)>0 and np.size(city_gps[city_gps["City_name"]==cities[0]].Long.values) >0:
        gps_1 =(city_gps[city_gps["City_name"]==cities[0]].Lat.values[0], city_gps[city_gps["City_name"]==cities[0]].Long.values[0])
        if np.size(city_gps[city_gps["City_name"]==cities[1]].Lat.values) >0 and np.size(city_gps[city_gps["City_name"]==cities[1]].Long.values) >0:
            gps_2 =(city_gps[city_gps["City_name"]==cities[1]].Lat.values[0], city_gps[city_gps["City_name"]==cities[1]].Long.values[0])
            dist.append(np.around(distance_calc(gps_1, gps_2),0))
    else:
        dist.append(0)
road_segs[road_segs["Length"]==0]["Length"] = dist 
road_segs["time"]= road_segs["Length"]/road_segs["Speed"]


# Create the Graph for search
from collections import defaultdict
import sys
import time
import numpy as np
import copy

t0 = time.time()

class Graph:
    def __init__(self):
        # default dictionary to store graph
        self.graph_dict = defaultdict(list)
        
    def add_graph_edge(self,parent, child, dist, time,hw):
        node_set = ((child, dist,time),hw)
        self.graph_dict[parent].append(node_set) if node_set not in self.graph_dict[parent] else self.graph_dict[parent]
        #return self.graph_dict[parent].append(child)
        
    
    def print_graph(self):
        print self.graph_dict
        
    def give_child_nodes(self, node):
		return self.graph_dict[node]
    
g = Graph()

for i in range(len(road_segs)):
    g.add_graph_edge(road_segs['first_city'][i], road_segs['Second_city'][i], road_segs['Length'][i],road_segs['time'][i],road_segs["Name of Highway"][i])
    g.add_graph_edge(road_segs['Second_city'][i], road_segs['first_city'][i],road_segs['Length'][i],road_segs['time'][i],road_segs["Name of Highway"][i])

city_gps_dict = city_gps.set_index('City_name').T.apply(tuple).to_dict()

def successors_bfs_dfs(city_list, visited_city_list):
	succ = [] 
	for sec_city in g.give_child_nodes(city_list[0].split(" ")[-1]):
		if sec_city[0][0] not in visited_city_list and sec_city[0][0] !=  start_city :
			succ.append(add_city(city_list,sec_city[0]) )
	#print "succ",succ
	return succ  

def add_city(city_array,sec_city):
    return  (city_array[0] +" " + sec_city[0],city_array[1] +sec_city[1],city_array[2] +sec_city[2])
	

def add_city2(city_array,sec_city):
	return  (city_array[1][0] +" " + sec_city[0] , city_array[1][1] +sec_city[1] , city_array[1][2] +sec_city[2])

# BFS and DFS   
# Solve for the cities

def solve_bfs_dfs(start_city,end_city ):
    fringe = [(start_city,0,0)]
    visited_city_list = []
    while len(fringe) > 0:
		if routing_algo == "bfs":
			path = fringe.pop(0)
		if routing_algo == "dfs":
			path = fringe.pop()
		if path[0].split(" ")[-1] not in visited_city_list:
			visited_city_list.append(path[0].split(" ")[-1]) 
			succ_arr = successors_bfs_dfs(path, visited_city_list)
			for iter in range (len(succ_arr)):
				if succ_arr[iter][0].split(" ")[-1] == end_city:
					return( succ_arr[iter]) 
				fringe.append( succ_arr[iter])
# Uniform
def successors_uniform(city_list, visited_city_list):
    succ = [] 
    pri = city_list[0]
    for sec_city in g.give_child_nodes(city_list[1][0].split(" ")[-1]):
        if sec_city[0][0] not in visited_city_list and sec_city[0][0] !=  start_city :
			temp = add_city2(city_list,sec_city[0])
			if cost == "distance":
				succ.append((temp[1],temp))
			if cost == "time":
				succ.append((temp[2],temp))
			if cost == "segments":
				succ.append((pri+1,temp))
			if cost == "longtour":
				succ.append((-temp[1],temp))
    return succ  

# Solve for the cities
def solve_uniform(start_city,end_city ):
	fringe = []
	heapq.heappush(fringe, (0, (start_city,0,0)))
	visited_city_list = []
	while len(fringe) > 0:
		path = heapq.heappop(fringe)
		if path[1][0].split(" ")[-1] not in visited_city_list:
			visited_city_list.append(path[1][0].split(" ")[-1])
			if path[1][0].split(" ")[-1] == end_city:
				return( path[1]) 
			succ_arr = successors_uniform(path, visited_city_list)
			for iter in range (len(succ_arr)):
				#if succ_arr[iter][1][0].split(" ")[-1] == end_city:
				#	return( succ_arr[iter][1]) 
				heapq.heappush(fringe, succ_arr[iter])
				


def successors_astar(city_list, visited_city_list):
    succ = [] 
    pri = city_list[0]
    parent = city_list[1][0].split(" ")[-1]
    heuristic=0
    for sec_city in g.give_child_nodes(parent):
		if sec_city[0][0] not in visited_city_list and sec_city[0][0] !=  start_city :
			temp = add_city2(city_list,sec_city[0])
			if cost in ("distance", "time"):
				if sec_city[0][0] in gps_cities and checker ==1:       
					heuristic =  distance_calc(city_gps_dict[end_city],city_gps_dict[sec_city[0][0]])
				else:
					heuristic = (pri - city_list[1][1])*10  - (sec_city[0][1] ) 
			if cost == "distance":
				succ.append((temp[1] + ((heuristic)/10),temp))
			if cost == "time": 
				succ.append((temp[2] + (heuristic/65)-1 ,temp))
			if cost == "segments":
				succ.append((pri+1 + heuristic,temp))
			if cost == "longtour":
				succ.append(( -(temp[1] + ((heuristic)/10)),temp))
			
    return succ  
            
# Solve for the cities
def solve_astar(start_city,end_city ):
	fringe = []
	heapq.heappush(fringe, (0 + distance_calc(city_gps_dict[start_city],city_gps_dict[end_city])  , (start_city,0,0)))
	visited_city_list = []
	while len(fringe) > 0:
		path = heapq.heappop(fringe)
		if path[1][0].split(" ")[-1] not in visited_city_list:
			visited_city_list.append(path[1][0].split(" ")[-1])
			if path[1][0].split(" ")[-1] == end_city:
				return( path[1]) 
			succ_arr = successors_astar(path, visited_city_list)
			for iter in range (len(succ_arr)):

				heapq.heappush(fringe, succ_arr[iter])    
 

gps_cities = list(city_gps["City_name"])
checker =0
if end_city in gps_cities:
    checker =1
if routing_algo in ("bfs", "dfs"):
	output_df = solve_bfs_dfs(start_city,end_city )
elif routing_algo == "uniform":
	output_df =  solve_uniform(start_city,end_city )
elif routing_algo == "astar":
	output_df =  solve_astar(start_city,end_city )
#tot_dist , tot_time , tot_path = solve2(start_city,end_city )

adj_cities = output_df[0].split(" ")
final_list = []
for i in range(len(adj_cities)-1):
	for child in g.give_child_nodes(adj_cities[i]):
		if child[0][0] == adj_cities[i+1]:
			#print child
			final_list.append((child[1],child[0][1],np.around(child[0][2],2)))
print pd.DataFrame(final_list,columns=["Highway","Distance in miles","Time in hours"])
print output_df[1], np.around(output_df[2],4) , output_df[0] 
#t1 = time.time()
#print "total time taken",t1-t0