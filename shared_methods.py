# COURSE PROJECT - Comparison of collective intelligence techniques performances
# Multiagent Systems - MUIA 2020/21
#
# SHARED METHODS
#
# Author: Luna Jimenez Fernandez
#
# This file contains the methods that are shared by all collective intelligence techniques implemented
# (mostly utility methods such as reading files and writing results)

###########
# IMPORTS #
###########

import os
import sys
import math
import shutil

import numpy as np
import matplotlib.pyplot as plt


###########
# METHODS #
###########

# MATHEMATICS #

def euclidean_distance(point_1, point_2):
    """
    Computes the euclidean distance between 2 points.

    :param point_1: Point 1 in (x,y) format
    :param point_2: Point 2 in (x,y) format
    :return: Distance as a float
    """

    node1_x, node1_y = point_1
    node2_x, node2_y = point_2

    return math.sqrt(((node1_x - node2_x) ** 2) + ((node1_y - node2_y) ** 2))


# LOCAL SEARCH

def two_opt(initial_tour, node_list):
    """
    Implementation of the 2-opt local search algorithm to improve solutions found
    by other metaheuristics

    :param initial_tour: Tour provided by the metaheuristic
    :return: Improved tour by using 2-opt
    """

    # Store the current best tour
    best_tour = initial_tour
    best_length = compute_tour_length(node_list, initial_tour)

    # Boolean used to store improvements
    improved = True

    # Repeat until no improvements are found
    while improved:
        improved = False

        # Store the current route
        current_tour = best_tour

        # Explore all posible swaps
        for i in range(1, len(current_tour) - 1):
            for j in range(i+1, len(current_tour)):
                # If j - i is 1, no swap would be performed, so it is skipped
                if j-i == 1:
                    continue
                # Compute the new path
                new_tour = current_tour[:]
                new_tour[i:j] = current_tour[j-1:i-1:-1]

                # Check if the solution improves
                if compute_tour_length(node_list, new_tour) < best_length:
                    # If it improves, store it
                    best_tour = new_tour
                    best_length = compute_tour_length(node_list, new_tour)
                    improved = True

    return best_tour


# TSP PROCESSING #

def read_file(file_path):
    """
    Given a valid path to a TSP file, extracts all the position information
    for a Travelling Salesman Problem

    This method is responsible for the error management in case of an invalid position

    :param file_path: A valid path to a TSP file
    :return: A list containing all positions as tuples (x,y) and the file name
    """

    # List to store the positions
    positions = []

    # Check that the path is valid
    if not os.path.isfile(file_path):
        # Error management
        print("ERROR - A valid TSP file must be passed as an argument")
        sys.exit()

    # If the file is valid, open it and start extracting information
    with open(file_path, "r") as file:

        # Loop through all the lines in the file
        for line in file:
            # Check if the line is not empty
            if line is not None:
                # Check that the line starts with a number
                if line[0].isdigit():

                    # File starts with a number - extract the info and store it
                    # Note - the X and Y coordinates are flipped in the file
                    line = line.strip()
                    line_elements = line.split()
                    positions.append((float(line_elements[1]), float(line_elements[2])))

    # Extract the file name
    file_name = os.path.basename(file_path)

    return positions, file_name


def generate_distances_matrix(node_list):
    """
    Given a node list of size N, creates an NxN matrix containing the distance
    between each pair of nodes

    The matrix is returned as a numpy matrix, and the distance used is the Euclidean distance

    :param node_list: List containing all the nodes of the problem
    :return: Numpy matrix containing all pairs of distances between the nodes
    """

    # Generate an initial matrix filled with 0s
    matrix = np.zeros((len(node_list), len(node_list)))

    # For each pair of nodes, compute the distance
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            matrix[i][j] = matrix[j][i] = euclidean_distance(node_list[i], node_list[j])

    return matrix


def compute_tour_length(node_list, tour):
    """
    Given a tour, computes the total length of said tour

    :param node_list: List containing all the nodes of the problem
    :param tour: List containing the indexes of the final tour
    :return: Distance of the tour
    """

    # Store the current distance
    current_distance = 0

    # Add all the distances
    for i in range(len(tour) - 1):
        current_distance += euclidean_distance(node_list[tour[i]], node_list[tour[i+1]])

    # Add the distance between the last and the first city
    current_distance += euclidean_distance(node_list[tour[-1]], node_list[tour[0]])

    return current_distance


def print_final_solution_graph(node_list, tour, execution, results_path):
    """
    Prints the final tour found by an execution of an algorithm

    :param node_list: List containing all the nodes in a problem
    :param tour: Final tour found by the algorithm
    :param execution: Number of the execution
    :param results_path: Path to the folder containing the results
    """

    # Get the x and y positions
    x_positions = [node_list[city][0] for city in tour]
    y_positions = [node_list[city][1] for city in tour]

    # Add the first city of the tour to the end to close the cycle
    x_positions.append(node_list[tour[0]][0])
    y_positions.append(node_list[tour[0]][1])

    # Create a new figure
    plt.figure()

    # Print the found path and the cities
    plt.plot(x_positions, y_positions)
    plt.scatter(x_positions, y_positions)

    # Save the graph
    plt.savefig(os.path.join(results_path, "graph_{}.png".format(execution)), dpi=600)


# FOLDER PROCESSING #

def prepare_folders(algorithm_name):
    """
    Prepares the folders to store the results

    :param algorithm_name: Name of the algorithm, used to create the folders
    :return: Path to the folder
    """

    # Check if the results folder exists
    if not os.path.isdir("Results"):
        # If it does not exist, create the folder
        os.mkdir("Results")

    # Check if there is already a results folder for said algorithm
    if os.path.isdir(os.path.join("Results", algorithm_name)):
        # If it exists, remove the folder
        shutil.rmtree(os.path.join("Results", algorithm_name))

    # Create the folder
    os.mkdir(os.path.join("Results", algorithm_name))

    return os.path.join("Results", algorithm_name)


# RESULTS FILE PROCESSING #

def create_file(algorithm_name):
    """
    Creates a file in which to store the results

    :param algorithm_name: Name of the algorithm
    :return: Handle to the file created and path of the folder created
    """

    # Create the appropriate folder for the file
    path = prepare_folders(algorithm_name)

    # Create a file to store the results
    file = open(os.path.join(path, "results.txt"), "w")

    # Print an initial message
    message = "TSP Solutions for algorithm: {0}".format(algorithm_name)
    print(message + "\n")
    file.write(message + "\n\n")

    return file, path


def print_execution_title(execution, file):
    """
    Prints a title for the current execution
    """

    message = "= EXECUTION {} =\n".format(execution)
    print(message)
    file.write(message + "\n")


def print_iteration(iteration, iteration_best, global_best, file):
    """
    Writes information about the current iteration to the file and the screen
    
    :param iteration: Current iteration
    :param iteration_best: Best tour found this iteration
    :param global_best: Best tour found globally
    :param file: Handle for the file
    """

    message = "{0} iteration - Iteration best: {1} / Global best: {2}".format(iteration, iteration_best, global_best)

    print(message)
    file.write(message + "\n")


def print_execution_final_information(best_path, iteration_count, time, file):
    """
    Prints the final results of an execution of the algorithm

    :param best_path: Length of the best tour found
    :param iteration_count: Total number of iterations performed
    :param time: Time taken
    :param file: Handle to the file
    """

    # Print a break line
    print("\n")
    file.write("\n\n")

    # Print the information
    title = "-RESULTS-"
    print(title)
    file.write(title + "\n")

    iterations = "Total iterations performed: {}".format(iteration_count)
    path = "Best path length: {:.4f}".format(best_path)
    time = "Total time taken: {:.4f}s\n".format(time)

    print(iterations)
    print(path)
    print(time)

    file.write(iterations + "\n")
    file.write(path + "\n")
    file.write(time + "\n")


def print_global_information(best_tours, times, file):
    """
    Prints the final results of the full algorithm execution

    :param best_tours: List containing all the tours of all executions
    :param times: List containing the time for each execution
    :param file: Handle of the file
    """

    # Print a break line
    print("\n")
    file.write("\n\n")

    # Print the information
    title = "= FINAL RESULTS =\n"
    print(title)
    file.write(title + "\n")

    # Find the best and worst executions
    best_execution = best_tours.index(min(best_tours))
    worst_execution = best_tours.index(max(best_tours))

    best_exe_message = "Best execution: {}".format(best_execution + 1)
    worst_exe_message = "Worst execution: {}\n".format(worst_execution + 1)

    best_tour = "Best tour length: {:.4f}".format(best_tours[best_execution])
    avg_tour = "Average tour length: {:.4f}".format(sum(best_tours) / len(best_tours))
    worst_tour = "Worst tour length: {:.4f}\n".format(best_tours[worst_execution])

    best_time = "Time of the best tour: {:.4f}".format(times[best_execution])
    worst_time = "Time of the worst tour: {:.4f}".format(times[worst_execution])
    avg_time = "Average time per tour: {:.4f}\n".format(sum(times) / len(times))

    print(best_exe_message)
    print(worst_exe_message)
    print(best_tour)
    print(avg_tour)
    print(worst_tour)
    print(best_time)
    print(worst_time)
    print(avg_time)

    file.write(best_exe_message + "\n")
    file.write(worst_exe_message + "\n")
    file.write(best_tour + "\n")
    file.write(avg_tour + "\n")
    file.write(worst_tour + "\n")
    file.write(best_time + "\n")
    file.write(worst_time + "\n")
    file.write(avg_time + "\n")


def print_tour_lengths(best_lengths, iteration_lengths, file):
    """
    Prints a list containing all the lengths of the tours for each iteration.
    This will be used later to create a graph with the evolution

    :param best_lengths: List containing the best global length of each iteration
    :param iteration_lengths: List containing the best length for each iteration
    :param file: Handle of the file
    """

    # Separation
    file.write("\n")

    # Write the best lengths
    file.write(array_to_string(best_lengths) + "\n")

    # Write the iteration lengths
    file.write(array_to_string(iteration_lengths) + "\n")


def array_to_string(array):
    """
    Converts an array to a comma delimited string
    """

    string = ""

    # Add each element
    for elem in array:
        string = "{0}{1},".format(string, elem)

    # Remove the last comma
    string = string[:-1]

    return string
