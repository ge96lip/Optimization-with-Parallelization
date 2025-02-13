import numpy as np
import matplotlib.pyplot as plt
import random
import os
import dask
import dask.array as da
from dask import delayed

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time
NUM_SIMULATIONS = 10  # Number of independent simulations
AVAILABLE_CORES = 8

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 

# Create folder for saving figures
OUTPUT_FOLDER = "Figures_task2"
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

@delayed
@delayed
def simulate_wildfire(simulation_id):
    """Simulates wildfire spread over time and returns fire spread data."""
    forest, burn_time = initialize_forest()
    fire_spread = []  # Track number of burning trees each day

    for _ in range(DAYS):
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

    return fire_spread 

def run_parallel_simulations(num_simulations):
    """Runs multiple wildfire simulations in parallel using Dask."""
    simulations = [simulate_wildfire(i) for i in range(num_simulations)]
    results = dask.compute(*simulations)  # Executes tasks in parallel
    return results


def plot_results(fire_spread_results, output_folder):
    """Plot individual simulations and the average wildfire spread in the main thread."""

    # Plot each individual simulation
    for i, fire_spread in enumerate(fire_spread_results):
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(fire_spread)), fire_spread, label=f"Simulation {i}")
        plt.xlabel("Days")
        plt.ylabel("Number of Burning Trees")
        plt.title(f"Wildfire Spread - Simulation {i}")
        plt.legend()

        filename = os.path.join(output_folder, f"multi_simulation_{i}_wildfire.png")
        plt.savefig(filename)
        plt.close()  # Free memory

    # Compute the average wildfire spread
    max_days = max(len(fs) for fs in fire_spread_results)
    padded_results = [fs + [0] * (max_days - len(fs)) for fs in fire_spread_results]  # Pad with zeros
    fire_spread_avg = np.mean(padded_results, axis=0)

    # Plot and save the average wildfire spread graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(fire_spread_avg)), fire_spread_avg, label="Average Burning Trees")
    plt.xlabel("Days")
    plt.ylabel("Number of Burning Trees")
    plt.title(f"Wildfire Spread Over Time (Avg of {NUM_SIMULATIONS} Runs)")
    plt.legend()

    avg_plot_filename = os.path.join(output_folder, "average_wildfire_spread.png")
    plt.savefig(avg_plot_filename)
    plt.show()
    
if __name__ == "__main__":
    from dask.distributed import Client

    # Start Dask Client
    client = Client(n_workers=AVAILABLE_CORES)  # Client(n_workers=4, threads_per_worker=4)
    print(client.dashboard_link)  # Open this link in a browser to monitor performance

    # Run simulations in parallel using Dask
    fire_spread_results = run_parallel_simulations(NUM_SIMULATIONS)

    # Plot all results in the main thread (avoiding crashes)
    plot_results(fire_spread_results, OUTPUT_FOLDER)