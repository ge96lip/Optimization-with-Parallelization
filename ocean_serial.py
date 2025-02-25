import numpy as np
import matplotlib.pyplot as plt
import os
import time
# Grid size 

TIME_STEPS = 100

def laplacian(field):
    """Computes the discrete Laplacian of a 2D field using finite differences."""
    lap = (
        np.roll(field, shift=1, axis=0) +
        np.roll(field, shift=-1, axis=0) +
        np.roll(field, shift=1, axis=1) +
        np.roll(field, shift=-1, axis=1) -
        4 * field
    )
    return lap

def update_ocean(u, v, temperature, wind, alpha=0.1, beta=0.02):
    """Updates ocean velocity and temperature fields using a simplified flow model."""    
    u_new = u + alpha * laplacian(u) + beta * wind
    v_new = v + alpha * laplacian(v) + beta * wind
    temperature_new = temperature + 0.01 * laplacian(temperature)  # Small diffusion
    return u_new, v_new, temperature_new


def main(size):
  # Create folder for saving figures

  current_directory = os.getcwd()
  OUTPUT_FOLDER = os.path.join(current_directory, r'Figures_BonusTask')
  
  os.makedirs(OUTPUT_FOLDER, exist_ok=True)
  # start = time.time()
  grid_size = size
  # Initialize temperature field (random values between 5C and 30C)
  temperature = np.random.uniform(5, 30, size=(grid_size, grid_size))

  # Initialize velocity fields (u: x-direction, v: y-direction)
  u_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))
  v_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))

  # Initialize wind influence (adds turbulence)
  wind = np.random.uniform(-0.5, 0.5, size=(grid_size, grid_size))   
  # Run the simulation
  for t in range(TIME_STEPS):
      u_velocity, v_velocity, temperature = update_ocean(u_velocity, v_velocity, temperature, wind)
      # if t % 10 == 0 or t == TIME_STEPS - 1:
          # print(f"Time Step {t}: Ocean currents updated.")
  # print("Serial with size: ",size,": ", time.time()-start)
  plt.ioff()
  # Plot the velocity field
  plt.figure(figsize=(6, 5))
  plt.quiver(u_velocity[::10, ::10], v_velocity[::10, ::10])
  plt.title("Ocean Current Directions")
  plt.xlabel("X Position")
  plt.ylabel("Y Position")
  plt.plot()
  filename = os.path.join(OUTPUT_FOLDER, f"serial_vels_{size}_ocean.png")
  plt.savefig(filename)
  plt.close()  # Free memory

  # Plot temperature distribution
  plt.figure(figsize=(6, 5))
  plt.imshow(temperature, cmap='coolwarm', origin='lower')
  plt.colorbar(label="Temperature (°C)")
  plt.title("Ocean Temperature Distribution")
  plt.plot()
  filename = os.path.join(OUTPUT_FOLDER, f"serial_temps_{size}_ocean.png")
  plt.savefig(filename)
  plt.close()  # Free memory
  

  print("Simulation complete.")