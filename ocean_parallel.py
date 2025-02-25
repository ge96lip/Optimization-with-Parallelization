import numpy as np
import matplotlib.pyplot as plt
import dask
import dask.array as da
import os
import time
import pyvtk  # Importing VTK module for structured grid output

# Grid size 

TIME_STEPS = 100
GRID_SIZE = 200



def laplacian(field):
    """Computes the discrete Laplacian of a 2D field using finite differences."""
    lap = da.map_overlap( 
        lambda x: da.roll(x, shift=1, axis=0) +
        da.roll(x, shift=-1, axis=0) +
        da.roll(x, shift=1, axis=1) +
        da.roll(x, shift=-1, axis=1) - 4 * x,
        field,
        depth=1,
        boundary="periodic"
    )
    comp = lap.compute()
    #print("Comp: ", comp)
    return comp

def update_ocean(u, v, temperature, wind, alpha=0.1, beta=0.02):
    """Updates ocean velocity and temperature fields using a simplified flow model."""    
    # start = time.time()
    # u_func = da.map_blocks(lambda a,b : a+alpha*laplacian(a)+beta*b, u, wind, dtype=float)
    #print("Calculate U: ", time.time() - start)
    u_new = u + alpha*laplacian(u)+beta*wind
    #print("Finish U: ", time.time() - start)
    
    # v_func = da.map_blocks(lambda a,b : a+alpha*laplacian(a)+beta*b, v, wind, dtype=float)
    #print("Calculate V: ", time.time() - start)
    v_new = v + alpha*laplacian(v)+beta*wind
    #print("Finish V: ", time.time() - start)
    # temp_func = da.map_blocks(lambda a : a+0.01*laplacian(a), temperature, dtype=float)
    #print("Calculate Temp: ", time.time() - start)
    temperature_new = temperature + alpha*laplacian(temperature)+beta*wind  # Small diffusion
    #print("Finish temp: ", time.time() - start)
    a = u_new, v_new, temperature_new
    return a


def main(size):
    GRID_SIZE = size 
    grid_size = size
    chunk_size = 100
    current_directory = os.getcwd()
    outputCount = 1
    OUTPUT_FOLDER = os.path.join(current_directory, r'Figures_BonusTask')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Initialize temperature field (random values between 5C and 30C)
    
    # start = time.time()
    #print("Start timer")
    temperature = np.random.uniform(5, 30, size=(grid_size, grid_size))
    d_temp = da.from_array(temperature, chunks=(chunk_size, chunk_size))

    # Initialize velocity fields (u: x-direction, v: y-direction)
    u_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))
    v_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))
    d_u_vel = da.from_array(u_velocity, chunks=(chunk_size, chunk_size))
    d_v_vel = da.from_array(v_velocity, chunks=(chunk_size, chunk_size))

    # Initialize wind influence (adds turbulence)
    wind = np.random.uniform(-0.5, 0.5, size=(grid_size, grid_size))
    d_wind = da.from_array(wind, chunks=(chunk_size, chunk_size))
    # Run the simulation
    #print("initialized")
    for t in range(TIME_STEPS):
        #mapped = da.map_blocks(update_ocean, d_u_vel, d_v_vel, d_temp, d_wind, dtype=float, enforce_ndim=True)
        #ret = mapped.compute()
        #print("RET: ", len(ret), " : ", ret)
        d_u_vel, d_v_vel, temperature = update_ocean(d_u_vel, d_v_vel, d_temp, d_wind)
        # d_u_vel, d_v_vel, d_temp = mapped.compute()
        if t % 10 == 0 or t == TIME_STEPS - 1:
            vtk_filename = f"bonus_frame_{outputCount:03d}.vtk"
            save_to_vtk(vtk_filename, d_u_vel, d_v_vel, temperature)
            outputCount += 1
            # print(f"Time Step {t}: Ocean currents updated.")
    # print("Parallel with size: ",size,": ", time.time()-start)
    
    plt.ioff()
    # Plot the velocity field=======================================]=
    plt.figure(figsize=(6, 5))
    plt.quiver(d_u_vel[::10, ::10], d_v_vel[::10, ::10])
    plt.title("Ocean Current Directions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.plot()
    filename = os.path.join(OUTPUT_FOLDER, f"parallel_vels_{size}_ocean.png")
    plt.savefig(filename)
    plt.close()  # Free memory

    # Plot temperature distribution
    plt.figure(figsize=(6, 5))
    plt.imshow(d_temp, cmap='coolwarm', origin='lower')
    plt.colorbar(label="Temperature (°C)")
    plt.title("Ocean Temperature Distribution")
    plt.plot()
    filename = os.path.join(OUTPUT_FOLDER, f"parallel_temps_{size}_ocean.png")
    plt.savefig(filename)
    plt.close()  # Free memory
    print("Simulation complete.")




def save_to_vtk(filename, vu, vv, temp):
    """
    Save the evolving simulation data as a VTK file.
    This function extracts the primitive variables and writes them to a VTK structured grid file.
    """
    # Convert conserved variables to primitive variables
    # vu, vv, temp, P = getPrimitive(VelU, VelV, Temperature)

    # Grid size
    nx = np.arange(0,GRID_SIZE,1)
    ny = np.arange(0,GRID_SIZE,1)
    nz = np.array([0])

    X,Y,Z = np.meshgrid(nx,ny,nz, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


    # Flatten data for VTK
    vu_flat = vu.compute().T.flatten()
    vv_flat = vv.compute().T.flatten()
    temp_flat = temp.compute().T.flatten()

    # Create VTK structure
    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredGrid(dimensions=(GRID_SIZE, GRID_SIZE, 1),points=points),  # 2D structured grid
        pyvtk.PointData(
            pyvtk.Scalars(temp_flat, name="temperature"),  # Density as a scalar field
            pyvtk.Vectors(np.column_stack((vu_flat, vv_flat, np.zeros_like(vu_flat))), name="velocity")  # Velocity as vectors
        )
    )
    vtk_data.tofile(filename)
    print(f"Saved VTK file: {filename}")