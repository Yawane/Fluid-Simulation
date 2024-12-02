import numpy as np
import matplotlib.pyplot as plt
from time import time



def apply_boundary_conditions(horizontal_field: np.ndarray, vertical_field: np.ndarray) -> None:
    """
    Apply no-slip boundary conditions to velocity fields.

    :param horizontal_field: 2D array representing the horizontal velocity component (u).
    :param vertical_field: 2D array representing the vertical velocity component (v).

    Modifies:
        horizontal_field: Updates the boundary rows to match the adjacent interior rows.
        vertical_field: Updates the boundary columns to match the adjacent interior columns.
    """
    # boundary condition (no-stick)
    horizontal_field[0, :] = horizontal_field[1, :]
    horizontal_field[-1, :] = horizontal_field[-2, :]

    # boundary conditions (no-stick)
    vertical_field[:, 0] = vertical_field[:, 1]
    vertical_field[:, -1] = vertical_field[:, -2]


def interpolate(field: np.ndarray, position: float, grid_spacing: float, max_index: int) -> np.ndarray:
    """
       Perform linear interpolation for a 1D field.

       :param field: 1D array representing the field values.
       :param position: Position at which to interpolate.
       :param grid_spacing: Spacing between grid points.
       :param max_index: Maximum valid index for interpolation.

       :return: Interpolated field value.
       """
    index = int(position // grid_spacing)
    weight = (position % grid_spacing) / grid_spacing

    if index >= max_index:  # Avoid out-of-bound interpolation
        return field[index]
    return (1 - weight) * field[index] + weight * field[index + 1]


def advect_component(field: np.ndarray, grid_spacing: float, time_step: float, axis_size: int) -> np.ndarray:
    """
    Perform advection for a single velocity component.

    :param field: 2D array representing the velocity component (u or v).
    :param grid_spacing: Spacing between grid points.
    :param time_step: Time step for the simulation.
    :param axis_size: Size of the field along the advection axis.

    :return: Updated velocity component after advection.
    """
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

    return new_field


def test_advection(horizontal_field: np.ndarray, vertical_field: np.ndarray, grid_spacing: float, delta_y: float, time_step: float) -> tuple:
    """
    Perform the advection step for horizontal and vertical velocity fields.

    Parameters:
        :param horizontal_field: 2D array representing the horizontal velocity component (u).
        :param vertical_field: 2D array representing the vertical velocity component (v).
        :param grid_spacing: Spacing between grid points.
        :param delta_y: Spacing between grid points in y-axis.
        :param time_step: Time step for the simulation.

        :return: Updated (horizontal_field, vertical_field) after advection.
    """
    # TODO: Particles moves horizontally and vertically only. The advection should integrate using the mean velocity value.
    start_time = time()
    nu = advect_component(horizontal_field, grid_spacing, time_step, horizontal_field.shape[1])
    nv = advect_component(vertical_field, delta_y, time_step, vertical_field.shape[0])

    # Apply boundary conditions
    apply_boundary_conditions(nu, nv)
    end_time = time()
    print(f"\t(test_advection)->\t{end_time - start_time:.4f} seconds")
    return nu, nv


def get_hypotenuse(distance1: float, distance2: float) -> float:
    return np.sqrt(distance1**2 + distance2**2)


def advection(horizontal_field: np.ndarray, vertical_field: np.ndarray, grid_spacing_x: float, grid_spacing_y: float, time_step: float) -> tuple:
    #TODO: advect u and v at the same time. u and v should both move in 2D space.
    start_time = time()
    new_horizontal_field = np.zeros_like(horizontal_field)
    new_vertical_field = np.zeros_like(vertical_field)

    # Advect horizontal_field
    for i in range(1, horizontal_field.shape[0] - 1):
        for j in range(horizontal_field.shape[1]):
            u = horizontal_field[i, j].item()
            v = .25 * np.sum(vertical_field[i-1:i+1, j:j+2])
            xG = max(0., min(grid_spacing_x*(horizontal_field.shape[1] - 1), grid_spacing_x*j - u*time_step))
            yG = max(0., min(grid_spacing_y*(vertical_field.shape[0] - 1), grid_spacing_y*(i - .5) - v*time_step))
            # print(f"({xG:.2f}, {yG:.2f})")

            jx = int(np.floor(xG % grid_spacing_x))
            iy = int(np.floor(yG % grid_spacing_y))
            # TODO: Find how to get if i index with the y-axis

            d0 = get_hypotenuse(xG - jx, yG - iy)
            d1 = get_hypotenuse(xG - jx + grid_spacing_x, yG - iy)
            d2 = get_hypotenuse(xG - jx, yG - iy + grid_spacing_y)
            d3 = get_hypotenuse(xG - jx + grid_spacing_x, yG - iy + grid_spacing_y)
            weight = d0 + d1 + d2 + d3
            # print(d0, d1, d2, d3, "\t", weight)

            new_horizontal_field[i, j] = (
                                             d0 * horizontal_field[jx, iy] +
                                             d1 * horizontal_field[jx, iy + 1] +
                                             d2 * horizontal_field[jx + 1, iy] +
                                             d3 * horizontal_field[jx + 1, iy + 1]
                                     ) / weight

    # Advect vertical field
    for i in range(vertical_field.shape[0]):
        for j in range(1, vertical_field.shape[1]-1):
            v = vertical_field[i, j].item()
            u = .25 * np.sum(horizontal_field[i:i+2, j-1:j+1])
            xG = max(0., min(grid_spacing_x * (horizontal_field.shape[1] - 1), grid_spacing_x * (j - .5) - u * time_step))
            yG = max(0., min(grid_spacing_y * (vertical_field.shape[0] - 1), grid_spacing_y * i - v * time_step))

            jx = int(np.floor(xG % grid_spacing_x))
            iy = int(np.floor(yG % grid_spacing_y))

            d0 = get_hypotenuse(xG - jx, yG - iy)
            d1 = get_hypotenuse(xG - jx + grid_spacing_x, yG - iy)
            d2 = get_hypotenuse(xG - jx, yG - iy + grid_spacing_y)
            d3 = get_hypotenuse(xG - jx + grid_spacing_x, yG - iy + grid_spacing_y)
            weight = d0 + d1 + d2 + d3

            new_vertical_field[i, j] = (
                d0 * vertical_field[jx, iy] +
                d1 * vertical_field[jx, iy + 1] +
                d2 * vertical_field[jx + 1, iy] +
                d3 * vertical_field[jx + 1, iy + 1]
            ) / weight

    end_time = time()
    print(f"\t(advection)->\t{end_time - start_time:.4f} seconds")
    apply_boundary_conditions(new_horizontal_field, new_vertical_field)
    return new_horizontal_field, new_vertical_field

if __name__ == "__main__":
    size = 2**2
    dx = dy = .1
    dt = .5
    u = np.random.randint(-3, 3, size=(size+2, size+1)).astype("float64")
    v = np.random.randint(-3, 3, size=(size+1, size+2)).astype("float64")
    apply_boundary_conditions(u, v)
    print(u)
    print()
    print(v)
    print("-"*20)

    nu, nv = advection(u, v, dx, dy, dt)

    print(np.round(nu, 4))
    print()
    print(np.round(nv, 4))


def build_matrix_A(grid_size: int) -> np.ndarray:
    """
    Build the sparse matrix A for the pressure gradient calculation.

    :param grid_size: Number of grid points along one axis.

    :return: matrix A
    """
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
    return A


def build_vector_b(horizontal_field: np.ndarray, vertical_field: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Build the vector b from the velocity fields.

    :param horizontal_field: 2D array representing the horizontal velocity component (u).
    :param vertical_field: 2D array representing the vertical velocity component (v).
    :param grid_size: Number of grid points along one axis.

    :return: vector b
    """
    b = np.zeros(grid_size ** 2, dtype='float64')
    for j in range(grid_size):
        for i in range(grid_size):
            b[i + j * grid_size] = (
                    horizontal_field[i + 1, j] - horizontal_field[i + 1, j + 1] +
                    vertical_field[i, j + 1] - vertical_field[i + 1, j + 1]
            )
    return b


def pressure_gradient(horizontal_field: np.ndarray, vertical_field: np.ndarray, grid_size: int, grid_spacing: float, time_step: float, rho: float) -> np.ndarray:
    """
    Compute the pressure gradient correction term for the velocity fields.

    :param horizontal_field: 2D array representing the horizontal velocity component (u).
    :param vertical_field: 2D array representing the vertical velocity component (v).
    :param grid_size: Number of grid points along one axis.
    :param grid_spacing: Spacing between grid points.
    :param time_step: Time step for the simulation.
    :param rho: Fluid density.

    :return: 2D array representing the pressure gradient correction.
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


def compute_divergence(horizontal_field: np.ndarray, vertical_field: np.ndarray, grid_size: int, grid_spacing: float) -> np.ndarray:
    """
    Computes the divergence of the velocity field.

    :param horizontal_field: 2D array representing the horizontal velocity component (u).
    :param vertical_field: 2D array representing the vertical velocity component (v).
    :param grid_size: umber of grid points along one axis.
    :param grid_spacing: Spacing between grid points.

    :return: The divergence field.
    """
    start_time = time()
    divergence = np.zeros((grid_size, grid_size), dtype='float64')
    for j in range(grid_size):
        for i in range(grid_size):
            divergence[i, j] = (
                                       (horizontal_field[i + 1, j + 1] - horizontal_field[i + 1, j]) +
                                       (vertical_field[i + 1, j + 1] - vertical_field[i, j + 1])
            ) / grid_spacing
    end_time = time()
    print(f"\t(compute_divergence)->\t{end_time - start_time:.4f} seconds")
    return divergence


def apply_pressure_correction(horizontal_field: np.ndarray, vertical_field: np.ndarray, grid_size: int, grid_spacing: float, time_step: float,
                              rho: float,
                              boundary_conditions: callable) -> None:
    """
    Updates the velocity field based on the pressure gradient.

    :param horizontal_field: 2D array representing the horizontal velocity component (u).
    :param vertical_field: 2D array representing the vertical velocity component (v).
    :param grid_size: Number of cells along one dimension of the computational grid.
    :param grid_spacing: Grid spacing.
    :param time_step: Time step.
    :param rho: Fluid density.
    :param boundary_conditions: Function to enforce boundary conditions on vu and vv.

    :return: None
    """
    pressure = pressure_gradient(horizontal_field, vertical_field, grid_size, grid_spacing, time_step, rho)
    for i in range(1, grid_size + 1):
        for j in range(grid_size + 1):
            horizontal_field[i, j] += time_step / (rho * grid_spacing) * (pressure[i, j + 1] - pressure[i, j])

    for i in range(grid_size + 1):
        for j in range(1, grid_size + 1):
            vertical_field[i, j] += time_step / (rho * grid_spacing) * (pressure[i + 1, j] - pressure[i, j])

    boundary_conditions(horizontal_field, vertical_field)


def compute_horizontal_velocity(horizontal_field: np.ndarray) -> np.ndarray:
    """
    Computes the cell-centered horizontal velocity from the staggered grid.

    :param horizontal_field: The horizontal velocity component (u).

    :return: The cell-centered horizontal velocity.
    """
    grid_size = horizontal_field.shape[0] - 2
    horizontal_velocity = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            horizontal_velocity[i, j] = (horizontal_field[i + 1, j + 1] + horizontal_field[i + 1, j]) / 2

    return horizontal_velocity


def compute_vertical_velocity(vertical_field: np.ndarray) -> np.ndarray:
    """
    Computes the cell-centered vertical velocity from the staggered grid.

    :param vertical_field: The vertical velocity component (v).

    :return: The cell-centered vertical velocity.
    """
    grid_size = vertical_field.shape[1] - 2
    vertical_velocity = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            vertical_velocity[i, j] = (vertical_field[i + 1, j + 1] + vertical_field[i, j + 1]) / 2
    return vertical_velocity


def compute_velocity(horizontal_velocity: np.ndarray, vertical_velocity: np.ndarray) -> np.ndarray:
    """
    Computes the magnitude of the velocity field.

    :param horizontal_velocity: The horizontal velocity component (u).
    :param vertical_velocity: The vertical velocity component (v).

    :return: The magnitude of the velocity field.
    """
    return np.sqrt(horizontal_velocity ** 2 + vertical_velocity ** 2)


def apply_external_forces(vertical_field: np.ndarray, vertical_force: np.ndarray, horizontal_field: np.ndarray, horizontal_force: np.ndarray, dt: float) -> None:
    vertical_field += vertical_force * dt
    horizontal_field += horizontal_force * dt


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
    vel_u_initial = compute_horizontal_velocity(horizontal_field)
    vel_v_initial = compute_vertical_velocity(vertical_field)
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
    apply_pressure_correction(u, v, grid_size, grid_spacing, time_step, rho, boundary_conditions)

    # Compute updated divergence and velocity magnitude
    divergence_after = compute_divergence(u, v, grid_size, grid_spacing)
    vel_u_final = compute_horizontal_velocity(u)
    vel_v_final = compute_vertical_velocity(v)
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