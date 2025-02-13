import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import os

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time
NUM_SIMULATIONS = 10  # Number of independent simulations

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 

# Create folder for saving figures
OUTPUT_FOLDER = "Figures_task1"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

def simulate_wildfire(simulation_id):
    """Simulates wildfire spread over time and returns the fire spread data."""
    forest, burn_time = initialize_forest()
    fire_spread = []  # Track number of burning trees each day

    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break

    # Save simulation plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(fire_spread)), fire_spread, label=f"Simulation {simulation_id}")
    plt.xlabel("Days")
    plt.ylabel("Number of Burning Trees")
    plt.title(f"Wildfire Spread - Simulation {simulation_id}")
    plt.legend()
    
    # Save image to Figures_task1 folder
    filename = os.path.join(OUTPUT_FOLDER, f"multi_simulation_{simulation_id}_wildfire.png")
    plt.savefig(filename)
    plt.close()  # Close plot to free memory

    return fire_spread

def run_parallel_simulations(num_simulations):
    """Runs multiple wildfire simulations in parallel and aggregates the results."""
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(simulate_wildfire, range(num_simulations))
    
    return results

if __name__ == "__main__":
    # Run simulations in parallel
    fire_spread_results = run_parallel_simulations(NUM_SIMULATIONS)

    # Aggregate results
    max_length = max(len(result) for result in fire_spread_results)
    fire_spread_avg = np.zeros(max_length)

    for result in fire_spread_results:
        padded_result = np.pad(result, (0, max_length - len(result)), 'constant', constant_values=0)
        fire_spread_avg += padded_result

    fire_spread_avg /= NUM_SIMULATIONS

    # Plot and save final average wildfire spread graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(fire_spread_avg)), fire_spread_avg, label="Average Burning Trees")
    plt.xlabel("Days")
    plt.ylabel("Number of Burning Trees")
    plt.title(f"Wildfire Spread Over Time (Avg of {NUM_SIMULATIONS} Runs)")
    plt.legend()
    
    # Save the averaged plot
    avg_plot_filename = os.path.join(OUTPUT_FOLDER, "average_wildfire_spread.png")
    plt.savefig(avg_plot_filename)
    plt.show()