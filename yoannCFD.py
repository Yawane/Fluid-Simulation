import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import jit


def apply_boundary_conditions(horizontal_field, vertical_field):
    """
    Apply no-slip boundary conditions to velocity fields.

    Parameters:
        horizontal_field (np.ndarray): 2D array representing the horizontal velocity component (u).
        vertical_field (np.ndarray): 2D array representing the vertical velocity component (v).

    Modifies:
        horizontal_field: Updates the boundary rows to match the adjacent interior rows.
        vertical_field: Updates the boundary columns to match the adjacent interior columns.
    """
    start_time = time()
    # boundary condition (no-stick)
    horizontal_field[0, :] = horizontal_field[1, :]
    horizontal_field[-1, :] = horizontal_field[-2, :]

    # boundary conditions (no-stick)
    vertical_field[:, 0] = vertical_field[:, 1]
    vertical_field[:, -1] = vertical_field[:, -2]
    end_time = time()
    print(f"\t(apply_boundary_conditions)->\t{end_time - start_time:.4f} seconds")


def interpolate(field, position, grid_spacing, max_index):
    """
       Perform linear interpolation for a 1D field.

       Parameters:
           field (np.ndarray): 1D array representing the field values.
           position (float): Position at which to interpolate.
           grid_spacing (float): Spacing between grid points.
           max_index (int): Maximum valid index for interpolation.

       Returns:
           float: Interpolated field value.
       """
    index = int(position // grid_spacing)
    weight = (position % grid_spacing) / grid_spacing

    if index >= max_index:  # Avoid out-of-bound interpolation
        return field[index]
    return (1 - weight) * field[index] + weight * field[index + 1]


def advect_component(field, grid_spacing, time_step, axis_size):
    """
    Perform advection for a single velocity component.

    Parameters:
        field (np.ndarray): 2D array representing the velocity component (u or v).
        grid_spacing (float): Spatial grid spacing (dx or dy).
        time_step (float): Time step for the simulation.
        axis_size (int): Size of the field along the advection axis.

    Returns:
        np.ndarray: Updated velocity component after advection.
    """
    start_time = time()
    new_field = np.zeros_like(field)
    m, n = field.shape
    is_horizontal = False if m < n else True

    for i in range(1 if is_horizontal else 0, m - 1 if is_horizontal else m):
        for j in range(0 if is_horizontal else 1, n if is_horizontal else n - 1):
            if field[i, j] == 0:
                continue

            # Compute the position of the particle back in time
            grid_pos = (j if is_horizontal else i) * grid_spacing
            advected_pos = max(grid_pos - field[i, j].item() * time_step, 0)
            advected_pos = min(advected_pos, (axis_size - 1) * grid_spacing)

            # Interpolation
            if is_horizontal:
                new_field[i, j] = interpolate(field[i, :], advected_pos, grid_spacing, n - 1)
            else:
                new_field[i, j] = interpolate(field[:, j], advected_pos, grid_spacing, m - 1)

    end_time = time()
    print(f"\t(advect_component)->\t{end_time - start_time:.4f} seconds")
    return new_field


def advection(horizontal_field, vertical_field, grid_spacing, delta_y, time_step):
    """
    Perform the advection step for horizontal and vertical velocity fields.

    Parameters:
        horizontal_field (np.ndarray): 2D array representing the horizontal velocity component (u).
        vertical_field (np.ndarray): 2D array representing the vertical velocity component (v).
        grid_spacing (float): Grid spacing in the x direction.
        delta_y (float): Grid spacing in the y direction.
        time_step (float): Time step for the simulation.

    Returns:
        tuple: Updated (horizontal_field, vertical_field) after advection.
    """
    start_time = time()
    nu = advect_component(horizontal_field, grid_spacing, time_step, horizontal_field.shape[1])
    nv = advect_component(vertical_field, delta_y, time_step, vertical_field.shape[0])

    # Apply boundary conditions
    apply_boundary_conditions(nu, nv)
    end_time = time()
    print(f"\t(advection)->\t{end_time - start_time:.4f} seconds")
    return nu, nv


def build_matrix_A(grid_size):
    """
    Build the sparse matrix A for the pressure gradient calculation.
    """
    start_time = time()
    n = grid_size ** 2
    A = np.zeros((n, n), dtype='int32')

    # Main diagonal (-4)
    np.fill_diagonal(A, -4)

    # Off-diagonals for neighbors
    np.fill_diagonal(A[1:], 1)  # Right neighbor
    np.fill_diagonal(A[:, 1:], 1)  # Left neighbor
    np.fill_diagonal(A[grid_size:], 1)  # Bottom neighbor
    np.fill_diagonal(A[:, grid_size:], 1)  # Top neighbor

    # Zero out periodic connections in rows
    for i in range(grid_size, n, grid_size):
        A[i, i - 1] = 0

    end_time = time()
    print(f"\t(build_matrix_A)->\t{end_time - start_time:.4f} seconds")
    return A


def build_vector_b(horizontal_field, vertical_field, grid_size):
    """
    Build the vector b from the velocity fields.
    """
    start_time = time()
    b = np.zeros(grid_size ** 2, dtype='float64')
    for j in range(grid_size):
        for i in range(grid_size):
            b[i + j * grid_size] = (
                    horizontal_field[i + 1, j] - horizontal_field[i + 1, j + 1] +
                    vertical_field[i, j + 1] - vertical_field[i + 1, j + 1]
            )
    end_time = time()
    print(f"\t(build_vector_b)->\t{end_time - start_time:.4f} seconds")
    return b


def pressure_gradient(horizontal_field, vertical_field, grid_size, grid_spacing, time_step, rho):
    """
    Compute the pressure gradient correction term for the velocity fields.

    Parameters:
        horizontal_field (np.ndarray): 2D array representing the horizontal velocity component (u).
        vertical_field (np.ndarray): 2D array representing the vertical velocity component (v).
        grid_size (int): Number of grid points along one axis.
        grid_spacing (float): Spatial grid spacing.
        time_step (float): Time step for the simulation.
        rho (float): Fluid density.

    Returns:
        np.ndarray: 2D array representing the pressure gradient correction.
    """
    start_time = time()
    A = build_matrix_A(grid_size)
    b = build_vector_b(horizontal_field, vertical_field, grid_size)

    # Solve for delta_p
    delta_p = np.zeros((grid_size + 2, grid_size + 2), dtype='float64') # with boundaries
    delta_p[1:-1, 1:-1] = np.linalg.solve((time_step / (rho * grid_spacing)) * A, b).reshape((grid_size, grid_size), order='F') # without boundaries

    end_time = time()
    print(f"\t(pressure_gradient)->\t{end_time - start_time:.4f} seconds")
    return delta_p


def compute_divergence(vu, vv, grid_size, grid_spacing):
    """
    Computes the divergence of the velocity field.

    Parameters:
        vu (np.ndarray): The horizontal velocity component (u).
        vv (np.ndarray): The vertical velocity component (v).
        grid_size (int): Number of cells along one dimension of the computational grid.
        grid_spacing (float): Grid spacing.

    Returns:
        np.ndarray: The divergence field.
    """
    start_time = time()
    divergence = np.zeros((grid_size, grid_size), dtype='float64')
    for j in range(grid_size):
        for i in range(grid_size):
            divergence[i, j] = (
                (vu[i + 1, j + 1] - vu[i + 1, j]) +
                (vv[i + 1, j + 1] - vv[i, j + 1])
            ) / grid_spacing
    end_time = time()
    print(f"\t(compute_divergence)->\t{end_time - start_time:.4f} seconds")
    return divergence


def apply_pressure_correction(horizontal_field, vertical_field, pressure, grid_size, grid_spacing, time_step, rho, boundary_conditions):
    """
    Updates the velocity field based on the pressure gradient.

    Parameters:
        horizontal_field (np.ndarray): The horizontal velocity component (u).
        vertical_field (np.ndarray): The vertical velocity component (v).
        pressure (np.ndarray): The pressure field.
        grid_size (int): Number of cells along one dimension of the computational grid.
        grid_spacing (float): Grid spacing.
        time_step (float): Time step.
        rho (float): Fluid density.
        boundary_conditions (callable): Function to enforce boundary conditions on vu and vv.
    """
    start_time = time()
    for i in range(1, grid_size + 1):
        for j in range(grid_size + 1):
            horizontal_field[i, j] += time_step / (rho * grid_spacing) * (pressure[i, j + 1] - pressure[i, j])

    for i in range(grid_size + 1):
        for j in range(1, grid_size + 1):
            vertical_field[i, j] += time_step / (rho * grid_spacing) * (pressure[i + 1, j] - pressure[i, j])

    boundary_conditions(horizontal_field, vertical_field)
    end_time = time()
    print(f"\t(apply_pressure_correction)->\t{end_time - start_time:.4f} seconds")


def compute_horizontal_velocity(horizontal_field, grid_size):
    """
    Computes the cell-centered horizontal velocity from the staggered grid.

    Parameters:
        horizontal_field (np.ndarray): The horizontal velocity component (u).
        grid_size (int): Number of cells along one dimension of the computational grid.

    Returns:
        np.ndarray: The cell-centered horizontal velocity.
    """
    start_time = time()
    horizontal_velocity = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            horizontal_velocity[i, j] = (horizontal_field[i + 1, j + 1] + horizontal_field[i + 1, j]) / 2

    end_time = time()
    print(f"\t(compute_horizontal_velocity)->\t{end_time - start_time:.4f} seconds")
    return horizontal_velocity


def compute_vertical_velocity(vertical_field, grid_size):
    """
    Computes the cell-centered vertical velocity from the staggered grid.

    Parameters:
        vertical_field (np.ndarray): The vertical velocity component (v).
        grid_size (int): Number of cells along one dimension of the computational grid.

    Returns:
        np.ndarray: The cell-centered vertical velocity.
    """
    start_time = time()
    vertical_velocity = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            vertical_velocity[i, j] = (vertical_field[i + 1, j + 1] + vertical_field[i, j + 1]) / 2
    end_time = time()
    print(f"\t(compute_vertical_velocity)->\t{end_time - start_time:.4f} seconds")
    return vertical_velocity


def visualize_pressure_correction(horizontal_field, vertical_field, pressure, grid_size, grid_spacing, time_step, rho, boundary_conditions):
    """
    Visualizes the effects of pressure correction on the velocity and divergence fields.

    Parameters:
        horizontal_field (np.ndarray): The horizontal velocity component (u).
        vertical_field (np.ndarray): The vertical velocity component (v).
        pressure (np.ndarray): The pressure field.
        grid_size (int): Number of cells along one dimension of the computational grid.
        grid_spacing (float): Grid spacing.
        time_step (float): Time step.
        rho (float): Fluid density.
        boundary_conditions (callable): Function to enforce boundary conditions on vu and vv.
    """
    u, v = horizontal_field, vertical_field
    start_time = time()
    # Compute initial divergence and velocity magnitude
    divergence_before = compute_divergence(horizontal_field, vertical_field, grid_size, grid_spacing)
    vel_u_initial = compute_horizontal_velocity(horizontal_field, grid_size)
    vel_v_initial = compute_vertical_velocity(vertical_field, grid_size)
    velocity_magnitude_initial = np.sqrt(vel_u_initial ** 2 + vel_v_initial ** 2)

    # Set up grid for plotting
    x = np.arange(grid_size)
    X, Y = np.meshgrid(x, x)

    # Plot initial divergence and pressure gradient
    plt.figure(figsize=(16, 10))

    plt.subplot(231)
    plt.pcolormesh(X, Y, divergence_before, shading='auto')
    plt.colorbar()
    plt.title("Divergence BEFORE pressure update")

    plt.subplot(232)
    plt.pcolormesh(X, Y, pressure[1:-1, 1:-1], shading='auto')
    plt.colorbar()
    plt.title("Pressure Field")

    # Apply pressure correction
    apply_pressure_correction(u, v, pressure, grid_size, grid_spacing, time_step, rho, boundary_conditions)

    # Compute updated divergence and velocity magnitude
    divergence_after = compute_divergence(u, v, grid_size, grid_spacing)
    vel_u_final = compute_horizontal_velocity(u, grid_size)
    vel_v_final = compute_vertical_velocity(v, grid_size)
    velocity_magnitude_final = np.sqrt(vel_u_final ** 2 + vel_v_final ** 2)

    # Plot updated divergence and velocity fields
    plt.subplot(233)
    plt.pcolormesh(X, Y, divergence_after, shading='auto')
    plt.colorbar()
    plt.title("Divergence AFTER pressure update")

    plt.subplot(234)
    plt.pcolormesh(X, Y, velocity_magnitude_initial, shading='auto')
    plt.colorbar()
    plt.title(f"Initial Velocity Field (Avg={np.mean(velocity_magnitude_initial):.4f})")

    plt.subplot(235)
    plt.pcolormesh(X, Y, velocity_magnitude_final, shading='auto')
    plt.colorbar()
    plt.title(f"Final Velocity Field (Avg={np.mean(velocity_magnitude_final):.4f})")

    plt.tight_layout()
    plt.show()
    end_time = time()
    print(f"\t(visualize_pressure_correction)->\t{end_time - start_time:.4f} seconds")