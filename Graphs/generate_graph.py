# COURSE PROJECT - Comparison of collective intelligence techniques performances
# Multiagent Systems - MUIA 2020/21
#
# GRAPH GENERATION
#
# Author: Luna Jimenez Fernandez
#
# This file is used to generate the plot of the evolution of solutions during the algorithm

###########
# IMPORTS #
###########

import sys

import matplotlib.pyplot as plt
import numpy as np

########
# CODE #
########

# Check that there is a file to be read
if len(sys.argv) == 1:
    print("ERROR: A file must be specified as an argument")
    sys.exit()

# Store the elements from the files in arrays
legend_name = []
iteration_data = []
best_data = []

# Read the file
with open(sys.argv[1], "r") as file:

    # Store the amount of NUMBERS read
    numbers = 0

    # Read through all the lines
    for line in file:

        # If the line starts with a character, it's a legend
        if line[0].isalpha():
            # Store the line
            legend_name.append(line.strip())

        # If not, it's a set of values
        elif line[0].isdigit():
            # Read the numbers
            line = line.strip()
            line_elements = [float(number) for number in line.split(',')]

            # If the length of the array is 1, increase it
            if len(line_elements) == 1:
                line_elements = line_elements * len(iteration_data[0])

            # Increase the amount of numbers
            numbers += 1

            # Depending on the value of numbers, store it in the appropriate array
            if numbers % 2 == 0:
                iteration_data.append(line_elements)
            else:
                best_data.append(line_elements)

# Create the iterations
iterations = np.arange(1, len(iteration_data[0]) + 1).tolist()

# Create the graphs

# Iteration best
plt.figure(figsize=(8,6))
plt.tight_layout()

for index in range(len(iteration_data)):

    plt.plot(iterations, iteration_data[index], label=legend_name[index])

plt.xlabel("Iteration")
plt.ylabel("Length of the best tour (iteration)")
plt.legend()
plt.savefig("iteration_best.png", dpi=600, bbox_inches='tight')
plt.savefig("iteration_best.eps", dpi=600, format="eps", bbox_inches='tight')

# Global best
plt.figure(figsize=(8,6))

for index in range(len(best_data)):

    plt.plot(iterations, best_data[index], label=legend_name[index])

plt.xlabel("Iteration")
plt.ylabel("Length of the best tour (global)")
plt.legend()
plt.savefig("global_best.png", dpi=600, bbox_inches='tight')
plt.savefig("global_best.eps", dpi=600, format="eps", bbox_inches='tight')

# Show the graphs
plt.show()