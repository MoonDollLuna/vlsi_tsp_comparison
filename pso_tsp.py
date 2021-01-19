# COURSE PROJECT - Comparison of collective intelligence techniques performances
# Multiagent Systems - MUIA 2020/21
#
# (DISCRETE) PARTICLE SWARM OPTIMIZATION
#
# Author: Luna Jimenez Fernandez
#
# This file contains the implementation for a Particle Swarm Optimization Algorithm used
# to solve the Travelling Salesman Problem
#
# This uses a specific variation called Discrete PSO, designed to work with discrete problems
# such as TSP.
# The function to minimize is the tour length

###########
# IMPORTS #
###########

import argparse
import time
import sys
import math

import shared_methods

import numpy as np

##################
# DEFAULT VALUES #
##################

# path - Path to the file to process. This argument MUST be specified
path = None

# These values are used by default by the algorithm, and can be modified by arguments

# method - Which method to use for DPSO. Options:
#   * "standard" - The standard DPSO implementation
#   * "improved" - An improved DPSO implementation
method = "improved"

# particle_count - Number of particles in the swarm
particle_count = 100

# inertia - Inertia coefficient, used while updating the velocity
inertia = 0.5

# local_acceleration - Acceleration constant between the current position and the best local position
local_acceleration = 1

# global_acceleration - Acceleration constant between the current position and the best global position
global_acceleration = 1

# iteration_count - Number of iterations
iteration_count = 300

# total_executions - Number of executions of the algorithm
total_executions = 5

# seed = Seed used for random events, used for reproducibility
seed = 0


###########
# CLASSES #
###########

class Particle:
    """
    A particle used in the Discrete Particle Swarm Optimization

    This particle uses the following set of operations, depending on the method:
    - STANDARD
        * Difference - Swaps between both tours (for x - y, swaps needed for y to become x)
        * Multiplication - Proportion of swaps taken
        * Addition - Concatenation of swap
        * Velocity - Applying the appropriate swaps to the solution

    - IMPROVED
        * Difference - Flips between both tours (for x - y, flips needed for y to become x)
        * Multiplication - Proportion of swaps taken
        * Addition and velocity: The following formulas are used:
            d_loc = current_pos + random * local_acceleration * (particle_attractor - current_position)
            d_glob = current_pos + random * global_acceleration * (global_attractor - current_position)
            random_v = random * random_constant * (random_accelerator - current_position)
            new_position = d_glob + 1/2 * (d_loc - d_glob) + random_v
    """

    def __init__(self, method, node_list, distance_matrix, inertia, local_acceleration, global_acceleration):
        """
        Constructor for a particle

        :param method: Method of DPSO to use.
        :param node_list: List of all the cities in the problem
        :param distance_matrix: Matrix containing all the distances between cities
        :param inertia: Inertia coefficient, used while updating the velocity
        :param local_acceleration: Acceleration constant between the current position and the best local position
        :param global_acceleration: Acceleration constant between the current position and the best global position
        """

        # Store the parameters
        self.method = method
        self.node_list = node_list
        self.distance_matrix = distance_matrix
        self.inertia = inertia
        self.local_acceleration = local_acceleration
        self.global_acceleration = global_acceleration

        # Generate an initial random position and store it as the best solution
        self.position = self._initial_position()
        self.particle_attractor = self.position
        self.particle_length = shared_methods.compute_tour_length(node_list, self.particle_attractor)

        # The initial global best is unknown
        self.global_attractor = None
        self.global_length = None

        # Particles have no initial velocity
        self.velocity = []

    # PRIVATE METHODS

    def _initial_position(self):
        """
        Generates an initial random position
        """

        position = np.arange(len(self.node_list))
        np.random.shuffle(position)
        return position.tolist()

    # STANDARD IMPLEMENTATION

    @staticmethod
    def _difference_standard(position_1, position_2):
        """
        Computes the difference between two positions

        The difference is understood as the SEQUENCE OF SWAPS
        needed to transform position_2 into position_1

        These swaps are computed from the first position to the
        last position in order

        Swaps are stored as (i, j), both being indexes

        :param position_1: First position to compare
        :param position_2: Second position to compare
        :return: Difference between position_1 and position_2
        """

        # Store temporal positions
        temp_pos1 = position_1[:]
        temp_pos2 = position_2[:]

        # Store the swaps being made
        swaps = []

        # Compare all positions
        for pos in range(len(temp_pos1)):

            # Check if the element at pos is the same
            if temp_pos1[pos] != temp_pos2[pos]:

                # A swap needs to be performed
                swap_index = temp_pos2.index(temp_pos1[pos])
                swap = (pos, swap_index)
                swaps.append(swap)

                # Update the temporal position with the swaps
                temp_pos2[pos], temp_pos2[swap_index] = temp_pos2[swap_index], temp_pos2[pos]

        return swaps

    @staticmethod
    def _product(scalar, difference):
        """
        Computes the product between a scalar and a difference
        (a list of swaps)

        # The scalar specifies which fraction of the difference is taken

        :param scalar: Fraction of the difference to take
        :param difference: List of swaps to be performed
        :return: List of swaps
        """

        # Compute the amount of swaps to take
        swap_count = round(scalar * len(difference))

        # Store the swaps
        swaps = []
        for swap in range(swap_count):
            swaps.append(difference[swap])

        return swaps

    @staticmethod
    def _addition(difference1, difference2):
        """
        Computes the addition between two differences

        The addition is just the concatenation of both differences
        """

        return difference1 + difference2

    def _apply_velocity(self):
        """
        Applies the velocity to the current position to obtain the new position

        The position is applied by applying all swaps
        """

        # Apply all swaps in order
        for swap in self.velocity:
            pos_1, pos_2 = swap
            self.position[pos_1], self.position[pos_2] = self.position[pos_2], self.position[pos_1]

    # IMPROVED IMPLEMENTATION

    @staticmethod
    def _addition_improved(position, difference):
        """
        Applies a list of edge exchanges to a position

        An edge exchange is a pair of values (i, j) such that:
        pos_i, pos_i+1 ... pos_j-1, pos_j = pos_j, pos_j-1, ..., pos_i+1, pos_i
        """

        # Store the temporal position
        temp_position = position[:]

        # Apply all edge exchanges
        for initial, end in difference:
            temp_position[initial:end+1] = reversed(temp_position[initial:end+1])

        return temp_position

    @staticmethod
    def _difference_improved(position1, position2):
        """
        Obtains the number of necessary edge exchanges for
        position 2 to become position 1

        An edge exchange is a pair of values (i, j) such that:
        pos_i, pos_i+1 ... pos_j-1, pos_j = pos_j, pos_j-1, ..., pos_i+1, pos_i

        :return:
        """

        # List of exchanges, stored as tuples
        exchanges = []

        # Exchange the temporal positions
        temp_pos1 = position1[:]
        temp_pos2 = position2[:]

        # While the lists are still not equal
        while temp_pos1 != temp_pos2:

            # Loop through all positions
            for index in range(len(temp_pos1)):

                # If indexes are different
                if temp_pos1[index] != temp_pos2[index]:
                    # Find where is the expected value in position 2
                    position2_index = temp_pos2.index(temp_pos1[index])

                    # Add the exchange to the list
                    exchanges.append((index, position2_index))

                    # Perform the exchange
                    temp_pos2[index:position2_index + 1] = reversed(temp_pos2[index:position2_index + 1])

        return exchanges

    # PUBLIC METHODS

    def update_position(self):
        """
        Computes the position reached by the particle,
        based on its current position and its attractors

        The formula from PSO is adapted to the TSP problem
        (since it is a discrete problem)
        """

        # Check which method to use
        if self.method == "standard":
            # Standard method

            # VELOCITY #

            # Compute the differences
            # diff_particle = (particle attractor - current position)
            diff_particle = self._difference_standard(self.particle_attractor, self.position)

            # diff_global = (global attractor - current position)
            diff_global = self._difference_standard(self.global_attractor, self.position)

            # Compute the products
            # temp_velocity = inertia * current_velocity
            temp_velocity = self._product(self.inertia, self.velocity)

            # temp_particle = local_acceleration * random * diff_particle
            temp_particle = self._product(self.local_acceleration * np.random.random_sample(), diff_particle)

            # temp_global = global_acceleration * diff_global
            temp_global = self._product(self.global_acceleration * np.random.random_sample(), diff_global)

            # Compute the additions
            # velocity = temp_velocity + temp_particle + temp_global
            self.velocity = self._addition(self._addition(temp_velocity, temp_particle), temp_global)

            # POSITION #

            # Compute the position by applying the velocity
            self._apply_velocity()

            # Check if the position improves the particle attractor
            current_position_length = shared_methods.compute_tour_length(node_list, self.position)
            if current_position_length < self.particle_length:
                # Improvement - store it
                self.particle_attractor = self.position
                self.particle_length = current_position_length

        else:
            # Improved method - ignores inertia

            # Compute the position that would be reached by following the local attractor
            local_difference = self._difference_improved(self.particle_attractor, self.position)
            local_product = self._product(np.random.random_sample() * self.local_acceleration, local_difference)
            local_destination = self._addition_improved(self.position, local_product)

            # Compute the position that would be reached by following the global attractor
            global_difference = self._difference_improved(self.global_attractor, self.position)
            global_product = self._product(np.random.random_sample() * self.global_acceleration, global_difference)
            global_destination = self._addition_improved(self.position, global_product)

            # Generate a random velocity to avoid premature convergence
            random_position = np.arange(len(self.node_list))
            np.random.shuffle(random_position)
            random_position = random_position.tolist()
            random_difference = self._difference_improved(random_position, self.position)
            random_velocity = self._product(np.random.random_sample() * np.random.random_sample(), random_difference)

            # Obtain the centroid from all of these positions
            new_difference = self._difference_improved(local_destination, global_destination)
            new_product = self._product(1/2, new_difference)
            new_position = self._addition_improved(global_destination, new_product)
            new_position = self._addition_improved(new_position, random_velocity)

            # Update the position
            self.position = new_position

            # Check if the position improves the particle attractor
            current_position_length = shared_methods.compute_tour_length(node_list, self.position)
            if current_position_length < self.particle_length:
                # Improvement - store it
                self.particle_attractor = self.position
                self.particle_length = current_position_length

    def set_global_attractor(self, global_attractor, global_length):
        """
        Updates the particle global attractor
        """

        self.global_attractor = global_attractor[:]
        self.global_length = global_length

    def get_particle_attractor(self):
        """
        Returns the current best position of the particle
        """

        return self.particle_attractor, self.particle_length


####################
# AUXILIAR METHODS #
####################

def get_global_best(particle_swarm):
    """
    Finds the global best from the particle swarm

    The length considered is the particle best

    :param particle_swarm: Swarm containing all the particles
    :return: Tour and length of the best particle
    """

    # Current best
    current_best = None
    current_length = math.inf

    # Check all particles
    for particle in particle_swarm:
        # Get the best from the particle
        particle_best, particle_length = particle.get_particle_attractor()

        # If the particle's best improves, keep it
        if particle_length < current_length:
            current_best, current_length = particle_best, particle_length

    return current_best, current_length


def update_global_best(particle_swarm, global_best_path, global_best_length):
    """
    Updates the global best path and length of all particles

    :param particle_swarm: Swarm containing all the particles
    :param global_best_path: Best path found globally
    :param global_best_length: Length of the best path found globally
    """

    # Update each particle one by one
    for particle in particle_swarm:
        particle.set_global_attractor(global_best_path, global_best_length)


######################
# MAIN LOOP AND CODE #
######################

def particle_swarm_optimization(method, node_list, distance_matrix, particle_count,
                                inertia, local_acceleration, global_acceleration,
                                iteration_count, file):
    """
    Performs Particle Swarm Optimization on the TSP, in order to obtain a tour

    A discretized version of PSO is used, with different operators to a typical PSO one

    :param method: Method of DPSO to use
    :param node_list: List containing all the nodes in the problem
    :param distance_matrix: Matrix containing the distances between all pairs of nodes
    :param particle_count: Number of particles in the swarm
    :param inertia: Importance given to the previous velocity in each iteration
    :param local_acceleration: Importance given to the particle attractor in each iteration
    :param global_acceleration: Importance given to the global attractor in each iteration
    :param iteration_count: Number of iterations to perform
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

    # Create the swarm of particles
    particle_swarm = []

    # Initialize all particles with a random initial position
    # The initial velocity will be empty (particles start without movement)
    for i in range(particle_count):
        particle_swarm.append(Particle(method, node_list, distance_matrix, inertia, local_acceleration, global_acceleration))

    # Find the best initial particle and set it as the global
    global_best_path, global_best_length = get_global_best(particle_swarm)

    # Assign all particles with the global best
    update_global_best(particle_swarm, global_best_path, global_best_length)

    # Start the iterations
    for iteration in range(iteration_count):

        # Compute the new position of all particles
        for particle in particle_swarm:
            particle.update_position()

        # Store the best tour of the iteration
        iteration_best_path, iteration_best_length = get_global_best(particle_swarm)

        # If the best tour in the iteration improves, keep it
        if iteration_best_length < global_best_length:
            global_best_path, global_best_length = iteration_best_path, iteration_best_length

        # Assign all particles with the global best
        update_global_best(particle_swarm, global_best_path, global_best_length)

        # Store the best lengths for the iteration
        list_iteration_lengths.append(iteration_best_length)
        list_best_lengths.append(global_best_length)

        # Print relevant information
        shared_methods.print_iteration(iteration+1, iteration_best_length, global_best_length, file)

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

    # method - Method of DPSO to use
    parser.add_argument('-m',
                        '--method',
                        choices=["standard", "improved"],
                        help="Method of DPSO to use. DEFAULT: {}".format(method))

    # particle_count - Number of particles in the swarm
    parser.add_argument('-pc',
                        '--particle_count',
                        type=int,
                        help="Number of particles in the swarm. Must be greater than 0. DEFAULT: {}".format(particle_count))

    # inertia - Inertia coefficient, how much of the previous velocity is used while updating
    parser.add_argument('-i',
                        '--inertia',
                        type=float,
                        help="Inertia coefficient, weight given to the previous velocity. Must be between 0.0 and 1.0. "
                             "DEFAULT:{}".format(inertia))

    # local_acceleration - Acceleration constant between the current position and the best local position
    parser.add_argument('-la',
                        '--local_acceleration',
                        type=float,
                        help="Acceleration constant between the current position and the best local position. Must "
                             "be between 0.0 and 1.0. DEFAULT:{}".format(local_acceleration))

    # global_acceleration - Acceleration constant between the current position and the best global position
    parser.add_argument('-ga',
                        '--global_acceleration',
                        type=float,
                        help="Acceleration constant between the current position and the best global position. Must "
                             "be between 0.0 and 1.0. DEFAULT:{}".format(global_acceleration))

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

    if arguments["method"]:
        method = arguments["method"]

    if arguments["particle_count"]:
        if arguments["particle_count"] <= 0:
            print("ERROR: Particle count must be greater than 0")
            sys.exit()
        else:
            particle_count = arguments["particle_count"]

    if arguments["inertia"]:
        if arguments["inertia"] < 0.0 or arguments["inertia"] > 1.0:
            print("ERROR: Inertia must be between 0.0 and 1.0")
            sys.exit()
        else:
            inertia = arguments["inertia"]

    if arguments["local_acceleration"]:
        if arguments["local_acceleration"] < 0.0 or arguments["local_acceleration"] > 1.0:
            print("ERROR: Local acceleration must be between 0.0 and 1.0")
            sys.exit()
        else:
            local_acceleration = arguments["local_acceleration"]

    if arguments["global_acceleration"]:
        if arguments["global_acceleration"] < 0.0 or arguments["global_acceleration"] > 1.0:
            print("ERROR: Global acceleration must be between 0.0 and 1.0")
            sys.exit()
        else:
            global_acceleration = arguments["global_acceleration"]

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
    file, folder = shared_methods.create_file("pso_{0}".format(file_name))

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
        best_path, best_distance, list_best, list_iteration = particle_swarm_optimization(method,
                                                                                          node_list,
                                                                                          distances_matrix,
                                                                                          particle_count,
                                                                                          inertia,
                                                                                          local_acceleration,
                                                                                          global_acceleration,
                                                                                          iteration_count,
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
