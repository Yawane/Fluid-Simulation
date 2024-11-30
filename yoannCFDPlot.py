import numpy as np
import matplotlib.pyplot as plt



def plot_field(field: np.ndarray, grid_spacing: float) -> None:
    """
    Display the velocity field using matplotlib.pcolormesh

    :param field: 2D array of real velocity field
    :param grid_spacing: Spacing between grid points.

    :return: None
    """
    m, n = field.shape
    x = np.arange(n) * grid_spacing
    y = np.arange(m) * grid_spacing
    X, Y = np.meshgrid(x, y)

    plt.pcolormesh(X, Y, field)
    plt.colorbar()
    plt.title("Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.axis("equal")
    plt.show()












def plot_velocity_streamlines(horizontal_field: np.ndarray, vertical_field: np.ndarray, grid_spacing: float):
    """
       Displays streamlines based on the velocity field.

       Parameters:
           :param horizontal_field: Horizontal velocity component.
           :param vertical_field: Vertical velocity component.
           :param grid_spacing: Spacing of the simulation grid.
       """

    x = np.arange(horizontal_field.shape[1]) * grid_spacing
    y = np.arange(horizontal_field.shape[0]) * grid_spacing
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, horizontal_field, vertical_field, color='blue', density=1.5)
    plt.title("Streamlines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()


def plot_vorticity(horizontal_field: np.ndarray, vertical_field, grid_spacing: float):
    """
    Displays the vorticity field.

    Parameters:
        horizontal_field (ndarray): Horizontal velocity component.
        vertical_field (ndarray): Vertical velocity component.
        grid_spacing (float): Spacing of the simulation grid.
    """
    dv_dx = np.gradient(vertical_field, axis=1) / grid_spacing
    du_dy = np.gradient(horizontal_field, axis=0) / grid_spacing
    vorticity = dv_dx - du_dy

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(vorticity, cmap='RdBu', shading='auto')
    plt.colorbar(label="Vorticity")
    plt.title("Vorticity field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


def plot_pressure_gradient(pressure_field, grid_spacing: float):
    """
    Visualizes the pressure gradient using arrows.

    Parameters:
        pressure_field (ndarray): Pressure field.
        grid_spacing (float): Spacing of the simulation grid.
    """

    dp_dx = np.gradient(pressure_field, axis=1) / grid_spacing
    dp_dy = np.gradient(pressure_field, axis=0) / grid_spacing

    x = np.arange(pressure_field.shape[1]) * grid_spacing
    y = np.arange(pressure_field.shape[0]) * grid_spacing
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, -dp_dx, -dp_dy, scale=1, scale_units='xy', angles='xy', color='red')
    plt.title("Pressure gradient")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()


def plot_kinetic_energy(horizontal_field, vertical_field):
    """
    Displays the kinetic energy density (E = 0.5 * (u^2 + v^2)).

    Parameters:
        horizontal_field (ndarray): Horizontal velocity component.
        vertical_field (ndarray): Vertical velocity component.
    """

    kinetic_energy = 0.5 * (horizontal_field ** 2 + vertical_field ** 2)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(kinetic_energy, cmap='viridis', shading='auto')
    plt.colorbar(label="Kinetic energy")
    plt.title("Kinetic energy density")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


def plot_residual_divergence(horizontal_field, vertical_field, grid_spacing: float):
    """
    Displays the divergence residual of the velocity field after pressure correction.

    Parameters:
        horizontal_field (ndarray): Horizontal velocity component.
        vertical_field (ndarray): Vertical velocity component.
        grid_spacing (float): Spacing of the simulation grid.
    """
    du_dx = np.gradient(horizontal_field, axis=1) / grid_spacing
    dv_dy = np.gradient(vertical_field, axis=0) / grid_spacing
    divergence = du_dx + dv_dy

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(divergence, cmap='RdBu', shading='auto')
    plt.colorbar(label="Divergence")
    plt.title("Residual divergence")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()
