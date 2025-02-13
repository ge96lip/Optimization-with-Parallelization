import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import multiprocessing
import dask
import dask.array as da
from dask import delayed
from dask.distributed import Client

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

# Experiment Settings
WORKER_COUNTS = [1, 2, 4, 8, 16]  # Test different Dask worker counts
CHUNK_SIZES = [5, 20, 100]  # Test different chunk sizes

# Create folder for saving figures
OUTPUT_FOLDER = "Figures_task3"
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

### Serial Execution
def run_serial():
    """Runs wildfire simulations sequentially."""
    results = []
    for i in range(NUM_SIMULATIONS):
        results.append(simulate_wildfire(i))
    return results

### Multiprocessing Execution
def run_multiprocessing():
    """Runs wildfire simulations using multiprocessing."""
    with multiprocessing.Pool(processes=AVAILABLE_CORES) as pool:
        results = pool.map(simulate_wildfire, range(NUM_SIMULATIONS))
    return results

### Dask Execution
@delayed
def simulate_wildfire_dask(simulation_id):
    """Delayed version of wildfire simulation for Dask execution."""
    return simulate_wildfire(simulation_id)

def run_dask(num_workers):
    """Runs wildfire simulations using Dask with a specific number of workers."""
    client = Client(n_workers=num_workers)

    start_time = time.time()
    simulations = [simulate_wildfire_dask(i) for i in range(NUM_SIMULATIONS)]
    results = dask.compute(*simulations)  # Executes tasks in parallel
    execution_time = time.time() - start_time

    client.close()
    return results, execution_time

### Experiment: Test Different Worker Counts
def test_worker_counts():
    """Test Dask execution with different worker counts."""
    execution_times = []

    for workers in WORKER_COUNTS:
        print(f"\nRunning Dask with {workers} workers...")
        _, exec_time = run_dask(workers)
        execution_times.append(exec_time)
        print(f"Execution Time with {workers} workers: {exec_time:.2f} sec")

    return WORKER_COUNTS, execution_times

### Experiment: Test Different Chunk Sizes
def test_chunk_sizes(fire_spread_results):
    """Measure execution time for different chunk sizes."""
    
    # Find the longest simulation (max days)
    max_days = max(len(sim) for sim in fire_spread_results)

    # Pad all arrays to have the same shape
    padded_results = [sim + [0] * (max_days - len(sim)) for sim in fire_spread_results]

    chunk_times = []

    for chunk in CHUNK_SIZES:
        dask_results = [da.from_array(result, chunks=chunk) for result in padded_results]
        
        start = time.time()
        fire_spread_avg = da.stack(dask_results, axis=0).mean(axis=0).compute()
        end = time.time()

        chunk_times.append(end - start)
        print(f"Chunk Size {chunk}: Execution Time = {end - start:.2f} sec")

    return CHUNK_SIZES, chunk_times

### Plot Performance Scaling
def plot_performance_scaling(worker_counts, execution_times, chunk_sizes, chunk_times):
    """Plot Worker Count vs. Execution Time and Chunk Size vs. Execution Time."""
    
    # Worker Count vs. Execution Time
    plt.figure(figsize=(8, 5))
    plt.plot(worker_counts, execution_times, marker="o", linestyle="-", label="Dask Workers")
    plt.xlabel("Number of Workers")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Worker Count vs. Execution Time")
    plt.legend()
    plt.grid()
    plt.savefig("Figures_task2/worker_vs_time.png")
    plt.show()

    # Chunk Size vs. Execution Time
    plt.figure(figsize=(8, 5))
    plt.plot(chunk_sizes, chunk_times, marker="s", linestyle="--", color="r", label="Chunk Size")
    plt.xlabel("Chunk Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Chunk Size vs. Execution Time")
    plt.legend()
    plt.grid()
    plt.savefig("Figures_task2/chunk_vs_time.png")
    plt.show()

### Measure Execution Times
if __name__ == "__main__":
    # Serial Execution
    print("\nRunning Serial Execution...")
    start_time = time.time()
    serial_results = run_serial()
    serial_time = time.time() - start_time
    print(f"Serial Execution Time: {serial_time:.2f} seconds")

    # Multiprocessing Execution
    print("\nRunning Multiprocessing Execution...")
    start_time = time.time()
    multiprocessing_results = run_multiprocessing()
    multiprocessing_time = time.time() - start_time
    print(f"Multiprocessing Execution Time: {multiprocessing_time:.2f} seconds")

    # Dask Worker Experiment
    worker_counts, execution_times = test_worker_counts()

    # Run Dask with Optimal Worker Count
    print("\nRunning Dask with Optimal Workers...")
    dask_results, dask_time = run_dask(AVAILABLE_CORES)
    print(f"Dask Execution Time: {dask_time:.2f} seconds")

    # Test Chunk Sizes
    chunk_sizes, chunk_times = test_chunk_sizes(dask_results)

    # Plot Performance Comparisons
    plot_performance_scaling(worker_counts, execution_times, chunk_sizes, chunk_times)

    # Print Comparison
    print("\n===== Performance Comparison =====")
    print(f"Serial Execution Time: {serial_time:.2f} sec")
    print(f"Multiprocessing Execution Time: {multiprocessing_time:.2f} sec")
    print(f"Dask Execution Time: {dask_time:.2f} sec")