# COURSE PROJECT - Comparison of collective intelligence techniques performances
# Multiagent Systems - MUIA 2020/21
#
# GENETIC ALGORITHM
#
# Author: Luna Jimenez Fernandez
#
# This file contains the implementation for a Genetic Algorithm used to
# solve the Travelling Salesman Problem
#
# The following elements are used:
#   * Genome: Encoding of the tour
#   * Selection: Rank based selection
#   * Crossover: SCX crossover operator / 2PCS crossover operator
#   * Mutation: Random swap
#   * Replacement: Elitism

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

# population_size - Size of the population
population_size = 200

# crossover_chance - Chance to perform crossover on each pair of elements of the sampled population
crossover_chance = 1.0

# mutation_chance - Chance to perform mutation on each element of the sampled population
mutation_chance = 0.1

# crossover_method - Method used for crossover.
# Valid options:
#   * "2pcs" - Two points crossover
#   * "scx" - Sequential constructive crossover
crosssover_method = "2pcs"

# iteration_count - Number of iterations
iteration_count = 300

# total_executions - Number of executions of the algorithm
total_executions = 5

# seed = Seed used for random events, used for reproducibility
seed = 0


#############################
# GENETIC ALGORITHM METHODS #
#############################

def fitness(individual, node_list):
    """
    Computes the fitness of an individual

    Fitness is computed as the inverse of the tour length
    """

    return 1 / shared_methods.compute_tour_length(node_list, individual)


def initial_population(node_list, population_size):
    """
    Generates an initial population randomly

    :param node_list: List of cities in the problem
    :param population_size: Population size
    :return: List containing the population
    """

    # Store the individuals
    population = []

    # Generate the population
    for i in range(population_size):

        # All individuals are PERMUTATIONS, so a shuffle can be used
        # Generate an initial permutation and shuffle it
        permutation = np.arange(len(node_list))
        np.random.shuffle(permutation)
        population.append(permutation.tolist())

    return population


def selection(population, node_list):
    """
    From the population, performs selection

    The method used is rank-based selection WITH replacement, to ensure that
    better individuals appear more times

    The selection has the same size as the population

    :param population: Full population of the problem
    :param node_list: List containing all the nodes in a city
    :return: Selection from the population
    """

    # Compute the fitness of the whole population
    fitnesses = [fitness(individual, node_list) for individual in population]

    # Order the fitness indexes from greater to smaller
    ranking = np.flip(np.argsort(fitnesses))

    # Compute the probability for each individual
    population_size = len(population)
    denominator = sum(np.arange(1, population_size + 1))

    probabilities = [(population_size - index + 1) / denominator for index, _ in enumerate(ranking)]

    # Normalize the probabilities
    probabilities_sum = sum(probabilities)
    probabilities = [probability / probabilities_sum for probability in probabilities]

    # Perform a random sample with replacement
    choices = np.random.choice(ranking, len(population), replace=True, p=probabilities)

    # Create the final selection
    selection = []
    for choice in choices:
        selection.append(population[choice])

    return selection


def crossover_2pcs(parent_1, parent_2):
    """
    Performs crossover between the two parents, creating two children

    The crossover operator is 2PCS (2 points centre crossover)

    :param parent_1: Parent 1
    :param parent_2: Parent 2
    :return: Two offsprings from the parents
    """

    # Choose two positions, l and r, randomly
    # l must be smaller than r

    l = np.random.choice(len(parent_1) - 2)
    r = np.random.choice(np.arange(l + 1, len(parent_1)))

    # Create the offspring

    # OFFSPRING 1
    offspring_1 = []
    # From 0 to l - 1, get parent 1 in parent 1 order
    offspring_1.extend(parent_1[0:l])
    # From l to r - 1, get parent 1 in parent 2 order
    parent_1_elements = parent_1[l:r]
    for element in parent_2:
        if element in parent_1_elements:
            offspring_1.append(element)
    # From r to the end, get parent 1 in parent 1 order
    offspring_1.extend(parent_1[r:])

    # OFFSPRING 2
    offspring_2 = []
    # From 0 to l - 1, get parent 2 in parent 2 order
    offspring_2.extend(parent_2[0:l])
    # From l to r - 1, get parent 2 in parent 1 order
    parent_2_elements = parent_2[l:r]
    for element in parent_1:
        if element in parent_2_elements:
            offspring_2.append(element)
    # From r to the end, get parent 2 in parent 2 order
    offspring_2.extend(parent_2[r:])

    return offspring_1, offspring_2


def crossover_scx(parent_1, parent_2, distance_matrix):
    """
    Performs crossover between the two parents, creating a single child

    The crossover operator is SCX (sequential constructive crossover operator)

    :param parent_1: Parent 1
    :param parent_2: Parent 2
    :param distance_matrix: Matrix containing all the distances between nodes
    :return: A SINGLE offspring from both parents
    """

    # List containing all the nodes not yet in the offspring
    not_visited = np.arange(len(parent_1)).tolist()

    # The offspring is built step by step
    offspring = []

    # Randomly choose between the initial node from both parents
    current_node = np.random.choice([parent_1[0], parent_2[0]])
    not_visited.remove(current_node)
    offspring.append(current_node)

    # Construct the solution
    while len(offspring) != len(parent_1):

        # Find the legitimate node for each parent
        # PARENT 1
        parent_1_index = parent_1.index(current_node)
        parent_1_node = None
        for i in range(parent_1_index + 1, len(parent_1)):
            # If there is a legitimate node, store it and stop looping
            if parent_1[i] not in offspring:
                parent_1_node = parent_1[i]
                break

        # If no legitimate node was found, get the first node from the not visited list
        if not parent_1_node:
            parent_1_node = not_visited[0]

        # PARENT 2
        parent_2_index = parent_2.index(current_node)
        parent_2_node = None
        for i in range(parent_2_index + 1, len(parent_2)):
            # If there is a legitimate node, store it and stop looping
            if parent_2[i] not in offspring:
                parent_2_node = parent_2[i]
                break

        # If no legitimate node was found, get the first node from the not visited list
        if not parent_2_node:
            parent_2_node = not_visited[0]

        # Both legitimate nodes are found
        # If they are the same, directly add it to the offspring
        if parent_1_node == parent_2_node:
            offspring.append(parent_1_node)
            not_visited.remove(parent_1_node)
        else:
            # If they are not the same, get the one with the shortest path
            if distance_matrix[current_node][parent_1_node] < distance_matrix[current_node][parent_2_node]:
                offspring.append(parent_1_node)
                not_visited.remove(parent_1_node)
            else:
                offspring.append(parent_2_node)
                not_visited.remove(parent_2_node)

    # A single offspring is returned
    return offspring


def mutation(individual):
    """
    Performs mutation on an individual by randomly swapping two indexes
    """

    # Get the indexes
    indexes = np.random.choice(len(individual), 2, replace=False)

    # Swap the indexes
    individual[indexes[0]], individual[indexes[1]] = individual[indexes[1]], individual[indexes[0]]

    return individual


def replacement(original_population, processed_population, node_list):
    """
    Creates a new population from the original and the processed population

    The strategy used is ELITISM (keeping the best n individuals from each population)

    :param original_population: Original population (before selection, crossover and mutation)
    :param processed_population: New population after selection, crossover and mutation
    :param node_list: List of cities in the problem
    :return: Final population from both populations
    """

    # Get the half size of the populations
    half_size = len(original_population) // 2

    # Store the final population
    final_population = []

    # Process the original population
    original_fitness = [fitness(individual, node_list) for individual in original_population]
    original_ordered_indexes = np.flip(np.argsort(original_fitness))
    for i in range(half_size):
        final_population.append(original_population[original_ordered_indexes[i]])

    # Process the original population
    processed_fitness = [fitness(individual, node_list) for individual in
                        processed_population]
    processed_ordered_indexes = np.flip(np.argsort(processed_fitness))
    for i in range(half_size):
        final_population.append(processed_population[processed_ordered_indexes[i]])

    return final_population


######################
# MAIN LOOP AND CODE #
######################

def genetic_algorithm(node_list, distance_matrix, population_size, crossover_chance, mutation_chance,
                      iteration_count, crossover_method, file):
    """
    Performs a Genetic Algorithm to solve the TSP

    The following elements are used:
        * Genome encoding: A tour of all the cities (permutation)
        * Initial population: Randomly generated
        * Selection: Rank-based selection
        * Crossover: SCX operator
        * Mutation: Gene swapping
        * Replacement: Elitism

    :param node_list: List containing all nodes (cities) in the problem
    :param distance_matrix: Matrix containing the distance between all pairs of cities
    :param population_size: Size of the population - population_size times the number of cities
    :param crossover_chance: Chance to perform crossover
    :param mutation_chance: Chance to perform mutation
    :param iteration_count: Number of iterations to perform
    :param crossover_method: Crossover method to use ("2pcs" or "scx")
    :param file: File used to store all the values
    :return: The best path obtained and its length. In addition, returns lists with the best tour length for each iteration
    (both the best globally and the best for that iteration)
    """

    # Store the global best tour
    global_best_path = None
    global_best_length = math.inf

    # Store the best tour length globally and of the iteration for each iteration
    list_best_lengths = []
    list_iteration_lengths = []

    # Generate the initial population
    population = initial_population(node_list, population_size)

    # Start the iterations
    for iteration in range(iteration_count):

        # Perform selection
        selected_population = selection(population, node_list)

        # Perform crossover
        crossover_population = []

        # Perform the appropriate crossover depending on the crossover choice
        if crossover_method == "scx":
            # SCX
            for i in range(len(population) - 1):

                # Check if crossover has to be done
                random = np.random.random_sample()
                if random < crossover_chance:
                    crossover_population.append(crossover_scx(selected_population[i], selected_population[i+1], distance_matrix))
                else:
                    crossover_population.append(selected_population[i])

            # Perform crossover between the last and first element (if needed)
            random = np.random.random_sample()
            if random < crossover_chance:
                crossover_population.append(crossover_scx(selected_population[-1], selected_population[0], distance_matrix))
            else:
                crossover_population.append(selected_population[-1])
        else:
            # 2PCS
            for i in range(0, len(selected_population), 2):
                # Check if crossover is necessary
                random = np.random.random_sample()
                if random < crossover_chance:
                    # If necessary, crossover both elements
                    offspring_1, offspring_2 = crossover_2pcs(selected_population[i], selected_population[i + 1])
                    crossover_population.append(offspring_1)
                    crossover_population.append(offspring_2)
                else:
                    # Keep both individuals
                    crossover_population.append(selected_population[i])
                    crossover_population.append(selected_population[i + 1])

        # Perform mutation
        mutation_population = []
        for i in range(len(population)):

            # Check if mutation has to be done
            random = np.random.random_sample()
            if random < mutation_chance:
                mutation_population.append(mutation(crossover_population[i]))
            else:
                mutation_population.append(crossover_population[i])

        # Perform replacement
        population = replacement(population, mutation_population, node_list)

        # Find the best individual in the population
        population_lengths = [shared_methods.compute_tour_length(node_list, individual) for individual in population]
        best_individual_index = np.argmax(population_lengths)

        # Get the best individual of the iteration
        iteration_best_tour = population[best_individual_index]
        iteration_best_length = population_lengths[best_individual_index]

        # If the best solution of the iteration improves the global solution, store it
        if iteration_best_length < global_best_length:
            global_best_path = iteration_best_tour
            global_best_length = iteration_best_length

        # Store the best lengths for the iteration
        list_iteration_lengths.append(iteration_best_length)
        list_best_lengths.append(global_best_length)

        # Print relevant information
        shared_methods.print_iteration(iteration+1,iteration_best_length, global_best_length, file)

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

    # population_size - Size of the population
    parser.add_argument('-ps',
                        '--population_size',
                        type=int,
                        help="Size of the population. Must be greater than 0. DEFAULT: {}".format(population_size))

    # crossover_chance - Chance of performing crossover for each pair of elements in the population
    parser.add_argument('-cc',
                        '--crossover_chance',
                        type=float,
                        help="Chance of performing crossover during the algorithm. Must be between 0.0 and 1.0. "
                             "DEFAULT:{}".format(crossover_chance))

    # mutation_chance - Chance of performing mutation to each element of the population
    parser.add_argument('-mc',
                        '--mutation_chance',
                        type=float,
                        help="Chance of performing mutation during the algorithm. Must be between 0.0 and 1.0. "
                             "DEFAULT:{}".format(crossover_chance))

    # crossover_method - Crossover operator used
    parser.add_argument('-cm',
                        '--crossover_method',
                        choices=["2pcs", "scx"],
                        help="Crossover operator to be used. DEFAULT: {}".format(crosssover_method))

    # iteration_count - Number of iterations performed by the algorithm
    parser.add_argument('-ic',
                        '--iteration_count',
                        type=int,
                        help="Number of iterations to be performed. This number must be greater than 0. "
                             "DEFAULT: " + str(iteration_count))

    # total_executions - How many times the algorithm is performed
    parser.add_argument('-te',
                        '--total_executions',
                        type=int,
                        help="Number of times the whole algorithm is repeated to obtain the average results. "
                             "This number must be greater than 0. DEFAULT: " + str(total_executions))

    # seed - Seed used for random events
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        help="Seed used for random events.")

    # Parse the arguments

    arguments = vars(parser.parse_args())

    if arguments["path"]:
        path = arguments["path"]

    if arguments["population_size"]:
        if arguments["population_size"] <= 0:
            print("ERROR: Population size must be greater than 0")
            sys.exit()
        else:
            population_size = arguments["population_size"]

    if arguments["crossover_chance"]:
        if arguments["crossover_chance"] < 0.0 or arguments["crossover_chance"] > 1.0:
            print("ERROR: Crossover chance must be between 0.0 and 1.0")
            sys.exit()
        else:
            crossover_chance = arguments["crossover_chance"]

    if arguments["mutation_chance"]:
        if arguments["mutation_chance"] < 0.0 or arguments["mutation_chance"] > 1.0:
            print("ERROR: Mutation chance must be between 0.0 and 1.0")
            sys.exit()
        else:
            mutation_chance = arguments["mutation_chance"]

    if arguments["crossover_method"]:
        crosssover_method = arguments["crossover_method"]

    if arguments["iteration_count"]:
        if arguments["iteration_count"] <= 0:
            print("ERROR: Iteration count must be greater than 0")
            sys.exit()
        else:
            iteration_count = arguments["iteration_count"]

    if arguments["total_executions"]:
        if arguments["total_executions"] <= 0:
            print("ERROR: Total executions must be greater than 0")
            sys.exit()
        else:
            total_executions = arguments["total_executions"]

    if arguments["seed"]:
        seed = arguments["seed"]

    # Set the seed
    np.random.seed(seed)

    # PROBLEM PREPARATION #
    # Extract the nodes and generate the distances
    node_list, file_name = shared_methods.read_file(path)
    distances_matrix = shared_methods.generate_distances_matrix(node_list)

    # Create a file to store all the results
    file, folder = shared_methods.create_file("ga_{0}".format(file_name))

    # PROBLEM EXECUTION #

    # Store a list with the best paths and their distances, and the times for each execution
    best_paths = []
    best_distances = []
    execution_times = []

    # Perform several executions
    for execution in range(total_executions):

        # Print the title for the current execution
        shared_methods.print_execution_title(execution + 1, file)

        # Execute the algorithm - timing it
        execution_time = time.time()
        best_path, best_distance, list_best, list_iteration = genetic_algorithm(node_list,
                                                                                distances_matrix,
                                                                                population_size,
                                                                                crossover_chance,
                                                                                mutation_chance,
                                                                                iteration_count,
                                                                                crosssover_method,
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
