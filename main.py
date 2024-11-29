import numpy as np
import matplotlib.pyplot as plt


def boundary(nu, nv):
    # boundery condition (no-stick)
    nu[0, :] = nu[1, :]
    nu[-1, :] = nu[-2, :]

    # boundery conditions (no-stick)
    nv[:, 0] = nv[:, 1]
    nv[:, -1] = nv[:, -2]


def advection(vu, vv):

    nu = np.zeros_like(vu)
    nv = np.zeros_like(vv)

    # u-component
    m, n = vu.shape
    for i in range(1, m - 1):
        for j in range(n):
            # fluid advection
            if vu[i, j] == 0:
                nu[i, j] = 0
                continue
            xg = j * dx
            xp = max(xg - vu[i, j] * dt, 0)
            xp = min(xp, (n - 1) * dx)
            nj = int((xp - xp % dx) / dx)
            if xp % dx == 0:
                nu[i, j] = vu[i, nj]
                continue
            p = xp / dx - nj
            nu[i, j] = (1 - p) * vu[i, nj] + p * vu[i, nj + 1]
    # boundery condition (no-stick)
    # nu[0, :] = nu[1, :]
    # nu[-1, :] = nu[-2, :]

    # v-component
    m, n = vv.shape
    for i in range(m):
        for j in range(1, n - 1):
            # fluid advection
            if vv[i, j] == 0:
                nv[i, j] = 0
                continue
            yg = i * dy
            yp = max(yg - vv[i, j] * dt, 0)
            yp = min(yp, (m - 1) * dy)
            ni = int((yp - yp % dy) / dy)
            if yp % dy == 0:
                nv[i, j] = vv[ni, j]
                continue
            p = yp / dy - ni
            nv[i, j] = (1 - p) * vv[ni, j] + p * vv[ni + 1, j]
    # boundery conditions (no-stick)
    # nv[:, 0] = nv[:, 1]
    # nv[:, -1] = nv[:, -2]

    boundary(nu, nv)
    return nu, nv


def pressure_gradient(vu, vv):
    # Get A matrix
    A = np.zeros((size ** 2, size ** 2))
    A -= -np.diag(np.ones(size ** 2)) * 4 + np.diag(np.ones(size ** 2 - 1), 1) + np.diag(np.ones(size ** 2 - 1),
                                                                                         -1) + np.diag(
        np.ones(size ** 2 - size), -size) + np.diag(np.ones(size ** 2 - size), size)
    for i in range(size, size ** 2, size):
        A[i, i - 1] = 0

    # Get b vector
    b = np.zeros(size ** 2)
    for j in range(size):
        for i in range(size):
            b[i + j * size] = vu[i + 1, j] - vu[i + 1, j + 1] + vv[i, j + 1] - vv[i + 1, j + 1]

    p = np.zeros((size + 2, size + 2))
    p[1:-1, 1:-1] = np.linalg.solve(dt / (rho * dx) * A, b).reshape((size, size), order='F')

    return p


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

    new_u, new_v = advection(u, v)

    p = pressure_gradient(u, v)

    show_pressure_gradient(u, v)
