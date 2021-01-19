PERFORMANCE STUDY OF COLLECTIVE INTELLIGENCE TECHNIQUES 
FOR THE TSP APPLIED TO VLSI CIRCUITS

Author: Luna Jimenez Fernandez
Multiagent Systems - MUIA 2020/21

# 1 - Contents

This assignment contains the following scripts:

- aco_tsp.py: Implementation of Ant-Colony Optimization in Python.
- genetic_tsp.py: Implementation of Genetic Algorithm in Python.
- pso_tsp.py: Implementation of Particle Swarm Optimization in Python.
- shared_methods.py: General methods shared by the three previous implementations.
                     This file cannot be directly launched.
					 
In addition, the following folders are included:

- Graphs: Contains the graphs used in the paper in higher resolution, as well as the
          script used to construct them (and the text filed used to do so).
- Problems: Contains the three problems to be solved using the algorithms as .tsp files.

Due to size constraints, the final results for all algorithms are not included.
However, they are available in the following public repository:
https://github.com/MoonDollLuna/vlsi_tsp_comparison

# 2 - Use of the scripts

The algorithms can be directly run in the CMD using:

	python <script_name> Problems\<problem_name>.tsp

The scripts have only been tested in Windows 10, so there may be problems running them in other OSs.
By using the following script:
	
	python <script_name> -h
	
A list of all optional parameters is shown. Most parameters for each script can be adjusted this way.
In addition, some scripts include some extra options not included in the final paper:

- ACO: A hybrid approach using 2-opt is also included.
- PSO: The simpler version proposed in the reference paper is also implemented.
- GA: A simpler version of the script using two point crossover is also implemented.