# To find the best path between two cities
Four different search algorithms were implemented to find the best path across US. 
1. Breadth First Search
2. Depth First Search
3. Uniform Search
4. Astar Search

In addtition to this, there is an option to use three different cost functions
1. Distance
2. Time
3. Number of road segments

The file can be run in the following manner

./route.py [start-city] [end-city] [routing-algorithm] [cost-function]

[routing-algorithm] can take the following values:
..1. bfs
..2. dfs
..3. uunifrom
..4. astar

[cost_function] can take the following values:
..1. segments
..2. distance
..3. time

The code finally outputs the route to be taken from the start city to end city.
