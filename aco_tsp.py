# COURSE PROJECT - Comparison of collective intelligence techniques performances
# Multiagent Systems - MUIA 2020/21
#
# ANT COLONY OPTIMIZATION
#
# Author: Luna Jimenez Fernandez
#
# This file contains the implementation for an Ant Colony Optimization implementation used to
# solve the Travelling Salesman Problem
#
# An implementation of MAX-MIN ACO is provided, with the option to improve solutions using 2-opt for local-search
# optimization

###########
# IMPORTS #
###########

import argparse
import sys
import time
import math

import shared_methods

import numpy as np

##################
# DEFAULT VALUES #
##################

# path - Path to the file to process. This argument MUST be specified
path = None

# These values are used by default by the algorithm, and can be modified by arguments

# ant_count - Number of ants to use
ant_count = 25

# iteration_count - How many iterations to perform
iteration_count = 300

# best_ant_percentage - What percentage of the iterations the best global tour is used for pheromones
# Note - the first (1 - best_ant_percentage)% iterations use the best iteration tour is used for pheromones
best_ant_percentage = 0.25

# total_executions - How many times the algorithm is executed
total_executions = 5

# pheromone_decay - Percentage by which pheromones decay after every iteration
pheromone_decay = 0.2

# alpha - Importance given to the pheromones by the algorithm
alpha = 1

# beta = Importance given to the heuristic values by the algorithm
beta = 1

# local_search = Boolean used to specify whether local search is performed or not
local_search = False

# seed = Seed used for random events, used for reproducibility
seed = 0


###########
# CLASSES #
###########

class Ant:
    """
    An ant used in the ACO algorithm

    The ant uses MAX-MIN Ant System ACO as its algorithm, and improves its
    solution found by applying a local search algorithm.

    Ants use the inverse of the distance between edges as a heuristic
    """

    def __init__(self, node_list, distance_matrix, pheromone_matrix, alpha, beta, local_search):
        """
        Ant constructor.

        :param node_list: List containing all the cities in the problem
        :param distance_matrix: Matrix containing the distance between all cities
        :param pheromone_matrix: Matrix containing the pheromones in the trail between all cities
        :param alpha: Importance given to the pheromones
        :param beta: Importance given to the heuristics
        :param local_search: Whether the ants use local search to further improve their tours
        """

        # Store all the values
        self.node_list = node_list
        self.distance_matrix = distance_matrix
        self.pheromone_matrix = pheromone_matrix
        self.alpha = alpha
        self.beta = beta
        self.local_search = local_search

        # Store the INDEX of the current path and the current node
        self.current_path = []
        self.current_node = None

    # PRIVATE METHODS #

    def _compute_probabilities(self, possible_visits):
        """
        Given the list of possible visits, computes all the probabilities
        of travelling to each city

        The heuristic used is the inverse of the distance between the current and the considered city

        :return: A list containing all the probabilities
        """

        # Compute the list of pheromones (already pondered by alpha)
        pheromones = [(self.pheromone_matrix[self.current_node][city]) ** self.alpha for city in possible_visits]

        # Compute the list of heuristics (already pondered by beta)
        heuristics = [(1 / self.distance_matrix[self.current_node][city]) ** self.beta for city in possible_visits]

        # Compute the actual denominator
        denominator = [pheromones[index] * heuristics[index] for index in range(len(pheromones))]
        denominator = sum(denominator)

        # Compute the list of probabilities
        probabilities = [(pheromones[index] * heuristics[index])/denominator for index in range(len(possible_visits))]

        return probabilities

    # PUBLIC METHODS #

    # Auxiliary #

    def get_tour(self):
        """
        Returns the tour found and its length
        """

        return self.current_path, shared_methods.compute_tour_length(self.node_list, self.current_path)

    # Main

    def find_path(self):
        """
        Generates a path using Ant Swarm
        """

        # Choose a random node to start
        self.current_node = np.random.choice(len(node_list))
        self.current_path.append(self.current_node)

        # Until all nodes have been visited by the ant
        while len(self.current_path) < len(self.node_list):

            # CONSTRUCTION STEP
            # Get all possible nodes to be visited
            possible_visits = [index for index in np.arange(len(self.node_list)) if index not in self.current_path]

            # Get the probabilities of visiting each node
            probabilities = self._compute_probabilities(possible_visits)

            # Randomly choose the next city to visit based on the probabilities
            chosen_city = np.random.choice(possible_visits, p=probabilities)

            # Once the city is chosen, update the information
            self.current_node = chosen_city
            self.current_path.append(chosen_city)

        # If local search is active, also performs a local search optimization
        if self.local_search:
            self.current_path = shared_methods.two_opt(self.current_path, self.node_list)


####################
# AUXILIAR METHODS #
####################

def update_pheromones(pheromone_matrix, node_list, tour, max_pheromone):
    """
    Given a tour, updates the pheromone matrix with the information of said tour

    :param pheromone_matrix: Matrix containing the pheromones between two cities
    :param node_list: List with all the nodes of the problem
    :param tour: Tour by the ant to update pheromones
    :param max_pheromone: Upper limit to pheromones
    :return: Updated pheromone matrix
    """

    # Compute the length of the tour
    length = shared_methods.compute_tour_length(node_list, tour)

    # Update the pheromones
    # NOTE - Pheromones are only updated in the direction followed by the ant. Graphs are assumed to be directed
    for i in range(len(tour) - 1):
        pheromone_matrix[tour[i]][tour[i+1]] += 1 / length
        if pheromone_matrix[tour[i]][tour[i+1]] > max_pheromone:
            pheromone_matrix[tour[i]][tour[i + 1]] = max_pheromone

    # Connect the last city with the first

    pheromone_matrix[tour[-1]][tour[0]] += 1 / length
    if pheromone_matrix[tour[-1]][tour[0]] > max_pheromone:
        pheromone_matrix[tour[-1]][tour[0]] = max_pheromone

    return pheromone_matrix


######################
# MAIN LOOP AND CODE #
######################

def ant_colony_optimization(node_list, distance_matrix, initial_best_route_length, ant_count,
                            iteration_count, best_ant_percentage, pheromone_decay, alpha, beta,
                            local_search, file):
    """
    Performs Ant Colony Optimization on the TSP, in order to obtain a tour

    MAX MIN Ant System is used as the ACO algorithm

    :param node_list: List containing all nodes (cities) in the problem
    :param distance_matrix: Matrix containing the distance between all pairs of cities
    :param initial_best_route_length: Length of the initial best route length (used for MAX MIN AS)
    :param ant_count: Number of ants to use
    :param iteration_count: Number of iterations to perform
    :param best_ant_percentage: Percentage of iterations in which the best route is used for pheromones
    :param pheromone_decay: Rate at which pheromones decay
    :param alpha: Alpha value
    :param beta: Beta value
    :param local_search: Whether ants use local search or not to improve their solutions
    :param file: File used to store all the values

    :return: The best path obtained and its length. In addition, returns lists with the best tour length for each iteration
    (both the best globally and the best for that iteration)
    """

    # Compute the initial max and min pheromone values
    max_pheromone = (1 / pheromone_decay) * (1 / initial_best_route_length)
    min_pheromone = max_pheromone / 2 * len(node_list)

    # Create the pheromones matrix
    pheromone_matrix = np.full(distance_matrix.shape, max_pheromone)

    # Store the global best tour
    global_best_path = None
    global_best_length = math.inf

    # Store the best tour length globally and of the iteration for each iteration
    list_best_lengths = []
    list_iteration_lengths = []

    # Compute in how many iterations the best iteration will be used for pheromones
    best_iteration_pheromones = iteration_count - (iteration_count * best_ant_percentage)

    # Start the iterations
    for iteration in range(iteration_count):

        # Store the iteration best tour
        iteration_best_path = None
        iteration_best_length = math.inf

        # Process each ant
        for ant_id in range(ant_count):
            ant = Ant(node_list, distance_matrix, pheromone_matrix, alpha, beta, local_search)
            ant.find_path()
            ant_path, ant_length = ant.get_tour()

            # If the path found is better, store it for the iteration
            if ant_length < iteration_best_length:
                iteration_best_path = ant_path
                iteration_best_length = ant_length

        # Ants are executed - post-processing

        # Update the global best if required
        if iteration_best_length < global_best_length:
            global_best_path = iteration_best_path
            global_best_length = iteration_best_length

        # Perform pheromone decay
        for y in range(pheromone_matrix.shape[0]):
            for x in range(pheromone_matrix.shape[1]):
                pheromone_matrix[y][x] = (1 - pheromone_decay) * pheromone_matrix[y][x]

                # Pheromones cannot go below the minimum value
                if pheromone_matrix[y][x] < min_pheromone:
                    pheromone_matrix[y][x] = min_pheromone

        # Update the pheromones of the appropriate ant
        # Check if the best global solution needs to be used
        if iteration + 1 <= best_iteration_pheromones:
            # Use the best iteration ant
            pheromone_matrix = update_pheromones(pheromone_matrix, node_list, iteration_best_path, max_pheromone)
        else:
            # Use the best global ant
            pheromone_matrix = update_pheromones(pheromone_matrix, node_list, global_best_path, max_pheromone)

        # Update the pheromone limits
        max_pheromone = (1 / pheromone_decay) * (1 / global_best_length)
        min_pheromone = max_pheromone / (2 * len(node_list))

        # Store the results
        shared_methods.print_iteration(iteration + 1, iteration_best_length, global_best_length, file)

        list_best_lengths.append(global_best_length)
        list_iteration_lengths.append(iteration_best_length)

    return global_best_path, global_best_length, list_best_lengths, list_iteration_lengths


# MAIN METHOD #

if __name__ == "__main__":

    # ARGUMENT PARSER #
    # Create the parser
    parser = argparse.ArgumentParser(description="Ant-Colony Optimization applied to the Travelling Salesman Problem")

    # Create all the arguments

    # Positional arguments

    # path - Path to the TSP file containing the problem
    parser.add_argument('path',
                        help="Path to the TSP file. This argument is REQUIRED and must be specified as:\n"
                             "python aco_tsp.py <path to the file>")

    # Optional arguments

    # ant_count - Numbers of ants to use
    parser.add_argument('-ants',
                        '--ant_count',
                        type=int,
                        help="Number of ants to use per iteration. The number of ants must be greater than 0. "
                             "DEFAULT: " + str(ant_count))

    # iteration_count - Number of iterations performed by the algorithm
    parser.add_argument('-ic',
                        '--iteration_count',
                        type=int,
                        help="Number of iterations to be performed. This number must be greater than 0. "
                             "DEFAULT: " + str(iteration_count))

    # best_ant_percentage
    parser.add_argument('-ba',
                        '--best_ant_percentage',
                        type=float,
                        help="Specifies the percentage of iterations that the best global route will be used for "
                             "pheromones. Note that the best global route is used only the last <percentage>%% of "
                             "iterations, while the (1-<percentage>)%% remaining iterations use the best iteration "
                             "route. This value must be between 0.0 and 1.0. DEFAULT: " + str(best_ant_percentage))

    # total_executions - How many times the algorithm is performed
    parser.add_argument('-te',
                        '--total_executions',
                        type=int,
                        help="Number of times the whole algorithm is repeated to obtain the average results. "
                             "This number must be greater than 0. DEFAULT: " + str(total_executions))

    # pheromone_decay - Rate at which the pheromones decay after every iteration
    parser.add_argument('-pd',
                        '--pheromone_decay',
                        type=float,
                        help="Rate at which the pheromones decay after every iteration. "
                             "This number must be between 0.0 and 1.0. DEFAULT: " + str(pheromone_decay))

    # alpha - Weight given to the pheromones when computing probabilities
    parser.add_argument('-alpha',
                        '--alpha',
                        type=float,
                        help="Weight given to the pheromones when computing probabilities. "
                             "This number must be greater or equal than 0.0. DEFAULT: " + str(alpha))

    # beta - Weight given to the heuristic when computing probabilities
    parser.add_argument('-beta',
                        '--beta',
                        type=float,
                        help="Weight given to the heuristic when computing probabilities. "
                             "This number must be greater or equal than 0.0. DEFAULT: " + str(beta))

    # local_search - Whether ants use local search to further improve their solutions
    parser.add_argument('-ls',
                        '--local_search',
                        action='store_true',
                        help='If chosen, ants also perform local search to improve their tours. The '
                             'algorithm used is 2-opt.')

    # seed - Seed used for random events
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        help="Seed used for random events.")

    # Parse the arguments

    arguments = vars(parser.parse_args())

    if arguments["path"]:
        path = arguments["path"]

    if arguments["ant_count"]:
        if arguments["ant_count"] <= 0:
            print("ERROR: Ant count must be greater than 0")
            sys.exit()
        else:
            ant_count = arguments["ant_count"]

    if arguments["iteration_count"]:
        if arguments["iteration_count"] <= 0:
            print("ERROR: Iteration count must be greater than 0")
            sys.exit()
        else:
            iteration_count = arguments["iteration_count"]

    if arguments["best_ant_percentage"]:
        if arguments["best_ant_percentage"] < 0.0 or arguments["best_ant_percentage"] > 1.0:
            print("ERROR: Best ant percentage must be between 0.0 and 1.0")
            sys.exit()
        else:
            best_ant_percentage = arguments["best_ant_percentage"]

    if arguments["total_executions"]:
        if arguments["total_executions"] <= 0:
            print("ERROR: Total executions must be greater than 0")
            sys.exit()
        else:
            total_executions = arguments["total_executions"]

    if arguments["pheromone_decay"]:
        if arguments["pheromone_decay"] < 0.0 or arguments["pheromone_decay"] > 1.0:
            print("ERROR: Pheromone decay must be between 0.0 and 1.0")
            sys.exit()
        else:
            pheromone_decay = arguments["pheromone_decay"]

    if arguments["alpha"]:
        if arguments["alpha"] < 0.0:
            print("ERROR: Alpha must be greater or equal than 0.0")
            sys.exit()
        else:
            alpha = arguments["alpha"]

    if arguments["beta"]:
        if arguments["beta"] < 0.0:
            print("ERROR: Beta must be greater or equal than 0.0")
            sys.exit()
        else:
            beta = arguments["beta"]

    if arguments["local_search"]:
        local_search = True

    if arguments["seed"]:
        seed = arguments["seed"]

    # Set the seed
    np.random.seed(seed)

    # PROBLEM PREPARATION #
    # Extract the nodes and generate the distances
    node_list, file_name = shared_methods.read_file(path)
    distances_matrix = shared_methods.generate_distances_matrix(node_list)

    # Specifies if local search is active
    local_search_message = ""
    if local_search:
        local_search_message = "_ls"

    # Create a file to store all the results
    file, folder = shared_methods.create_file("aco_{0}{1}".format(file_name, local_search_message))

    # PROBLEM EXECUTION #

    # Store a list with the best paths and their distances, and the times for each execution
    best_paths = []
    best_distances = []
    execution_times = []

    # Perform several executions
    for execution in range(total_executions):

        # Print the title for the current execution
        shared_methods.print_execution_title(execution + 1, file)

        # Generate a random path and use it as the initial distance
        random_start = np.arange(len(node_list))
        np.random.shuffle(random_start)
        initial_distance = shared_methods.compute_tour_length(node_list, random_start)

        # Execute the algorithm - timing it
        execution_time = time.time()
        best_path, best_distance, list_best, list_iteration = ant_colony_optimization(node_list,
                                                                                      distances_matrix,
                                                                                      initial_distance,
                                                                                      ant_count,
                                                                                      iteration_count,
                                                                                      best_ant_percentage,
                                                                                      pheromone_decay,
                                                                                      alpha,
                                                                                      beta,
                                                                                      local_search,
                                                                                      file)
        execution_time = time.time() - execution_time

        # Add the information to the global lists
        best_paths.append(best_path)
        best_distances.append(best_distance)
        execution_times.append(execution_time)

        # Print the final results and the found path
        shared_methods.print_execution_final_information(best_distance, iteration_count, execution_time, file)
        shared_methods.print_final_solution_graph(node_list, best_path, execution + 1, folder)

        # Store at the end of the file the best tours each iteration
        shared_methods.print_tour_lengths(list_best, list_iteration, file)

    # Print the total results of all executions
    shared_methods.print_global_information(best_distances, execution_times, file)

    # Close the file
    file.close()
