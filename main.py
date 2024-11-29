import numpy as np
import matplotlib.pyplot as plt


def boundary(nu, nv):
    # boundary condition (no-stick)
    nu[0, :] = nu[1, :]
    nu[-1, :] = nu[-2, :]

    # boundary conditions (no-stick)
    nv[:, 0] = nv[:, 1]
    nv[:, -1] = nv[:, -2]


def interpolate(field, pos, delta, max_index):
    """
    Interpolate the value of the field at a given position.
    """
    index = int(pos // delta)
    weight = (pos % delta) / delta

    if index >= max_index:  # Avoid out-of-bound interpolation
        return field[index]
    return (1 - weight) * field[index] + weight * field[index + 1]

def advect_component(field, delta_x, delta_t, axis_size):
    """
    Perform the advection for a single velocity component (field = u or v).
    """
    new_field = np.zeros_like(field)
    m, n = field.shape
    is_horizontal = True if m < n else False

    for i in range(0 if is_horizontal else 1, m if is_horizontal else m - 1):
        for j in range(1 if is_horizontal else 0, n - 1 if is_horizontal else n):
            if field[i, j] == 0:
                continue

            # Compute the position of the particle back in time
            grid_pos = (i if is_horizontal else j) * delta_x
            advected_pos = max(grid_pos - field[i, j] * delta_t, 0)
            advected_pos = min(advected_pos, (axis_size - 1) * delta_x)

            # Interpolation
            if not is_horizontal:
                new_field[i, j] = interpolate(field[i, :], advected_pos, delta_x, n - 1)
            else:
                new_field[i, j] = interpolate(field[:, j], advected_pos, delta_x, m - 1)

    return new_field

def advection(vu, vv, delta_x, delta_y, delta_t):
    """
    Perform the advection step for velocity components vu and vv.
    """
    nu = advect_component(vu, delta_x, delta_t, vu.shape[1])
    nv = advect_component(vv, delta_y, delta_t, vv.shape[0])

    # Apply boundary conditions
    boundary(nu, nv)
    return nu, nv


def build_matrix_A(size):
    """
    Build the sparse matrix A for the pressure gradient calculation.
    """
    n = size ** 2
    A = np.zeros((n, n))

    # Main diagonal (-4)
    np.fill_diagonal(A, -4)

    # Off-diagonals for neighbors
    np.fill_diagonal(A[1:], 1)  # Right neighbor
    np.fill_diagonal(A[:, 1:], 1)  # Left neighbor
    np.fill_diagonal(A[size:], 1)  # Bottom neighbor
    np.fill_diagonal(A[:, size:], 1)  # Top neighbor

    # Zero out periodic connections in rows
    for i in range(size, n, size):
        A[i, i - 1] = 0
        A[i - 1, i] = 0

    return A


def build_vector_b(vu, vv, size):
    """
    Build the vector b from the velocity fields.
    """
    b = np.zeros(size ** 2)
    for j in range(size):
        for i in range(size):
            b[i + j * size] = (
                vu[i + 1, j] - vu[i + 1, j + 1] +
                vv[i, j + 1] - vv[i + 1, j + 1]
            )
    return b


def pressure_gradient(vu, vv, size, delta_x, delta_t, rho):
    """
    Compute the pressure gradient correction term.
    """
    A = build_matrix_A(size)
    b = build_vector_b(vu, vv, size)

    # Solve for delta_p
    delta_p = np.zeros((size + 2, size + 2)) # with boundaries
    delta_p[1:-1, 1:-1] = np.linalg.solve((delta_t / (rho * delta_x)) * A, b).reshape((size, size), order='F') # without boundaries

    return delta_p


def check_divergence(vu, vv):
    d = np.zeros((size, size))
    for j in range(size):
        for i in range(size):
            d[i, j] = (vu[i + 1, j + 1] - vu[i + 1, j] + vv[i + 1, j + 1] - vv[i, j + 1]) / dx

    return d


def update_field(vu, vv):
    for i in range(1, size + 1):
        for j in range(size + 1):
            vu[i, j] -= dt / (rho * dx) * (p[i, j + 1] - p[i, j])

    for i in range(size + 1):
        for j in range(1, size + 1):
            vv[i, j] -= dt / (rho * dx) * (p[i + 1, j] - p[i, j])

    boundary(vu, vv)


def velocity_u(vu):
    vel_u = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            vel_u[i, j] = (vu[i + 1, j + 1] - vu[i + 1, j]) / 2

    return vel_u


def velocity_v(vv):
    vel_v = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            vel_v[i, j] = (vv[i + 1, j + 1] - vv[i, j + 1]) / 2

    return vel_v


def show_pressure_gradient(vu, vv):
    vel_u = velocity_u(vu)
    vel_v = velocity_v(vv)
    vel = np.sqrt(vel_u ** 2 + vel_v ** 2)

    x = np.arange(size)
    X, Y = np.meshgrid(x, x)

    plt.figure(figsize=(16, 10))
    plt.subplot(231)
    plt.pcolormesh(X, Y, check_divergence(u, v))
    plt.colorbar()
    plt.title("Divergence BEFORE pressure update")

    plt.subplot(232)
    plt.pcolormesh(X, Y, p[1:-1, 1:-1])
    plt.colorbar()
    plt.title("Pressure gradient")

    update_field(vu, vv)

    plt.subplot(233)
    plt.pcolormesh(X, Y, check_divergence(u, v))
    plt.colorbar()
    plt.title("Divergence AFTER pressure update")

    plt.subplot(234)
    plt.pcolormesh(X, Y, vel)
    plt.colorbar()
    plt.title(f"Initial velocity field. Avg={np.mean(vel):.3f}")

    vel_u = velocity_u(vu)
    vel_v = velocity_v(vv)
    vel = np.sqrt(vel_u ** 2 + vel_v ** 2)

    plt.subplot(235)
    plt.pcolormesh(X, Y, vel)
    plt.colorbar()
    plt.title(f"Final velocity field. Avg={np.mean(vel):.3f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    size = 2 ** 5
    dx = dy = .1
    rho = 1000

    u = np.random.randint(-3, 3, (size + 2, size + 1)).astype("float64")
    v = np.random.randint(-3, 3, (size + 1, size + 2)).astype("float64")

    dt = dx / (2 * max(np.max(u), np.max(v)))
    print("dt:", dt)

    new_u, new_v = advection(u, v, dx, dx, dt)

    p = pressure_gradient(u, v, size, dx, dt, rho)

    show_pressure_gradient(u, v)
