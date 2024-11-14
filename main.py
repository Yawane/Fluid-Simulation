import numpy as np


def advection(u, v):
    nu = np.zeros_like(u)
    nv = np.zeros_like(v)

    # u-component
    m, n = u.shape
    for i in range(1, m - 1):
        for j in range(n):
            # fluid advection
            if u[i, j] == 0:
                nu[i, j] = 0
                continue
            xg = j * dx
            xp = max(xg - u[i, j] * dt, 0)
            xp = min(xp, (n - 1) * dx)
            nj = int((xp - xp % dx) / dx)
            if xp % dx == 0:
                nu[i, j] = u[i, nj]
                continue
            p = xp / dx - nj
            nu[i, j] = (1 - p) * u[i, nj] + p * u[i, nj + 1]
    # boundary condition (no-stick)
    nu[0, :] = nu[1, :]
    nu[-1, :] = nu[-2, :]

    # v-component
    m, n = v.shape
    for i in range(m):
        for j in range(1, n - 1):
            # fluid advection
            if v[i, j] == 0:
                nv[i, j] = 0
                continue
            yg = i * dy
            yp = max(yg - v[i, j] * dt, 0)
            yp = min(yp, (m - 1) * dy)
            ni = int((yp - yp % dy) / dy)
            if yp % dy == 0:
                nv[i, j] = v[ni, j]
                continue
            p = yp / dy - ni
            nv[i, j] = (1 - p) * v[ni, j] + p * v[ni + 1, j]
    # boundary conditions (no-stick)
    nv[:, 0] = nv[:, 1]
    nv[:, -1] = nv[:, -2]

    return nu, nv


def show():
    print(u)
    print()
    print(nu)

    print()
    print('----')
    print()

    print(v)
    print()
    print(nv)

epsilon = 1e-5

sx = sy = 2 ** 2
dx = dy = 1.5
dt = 1.2

u = np.random.randint(-3, 3, (sx + 2, sy + 1)).astype("float32")
v = np.random.randint(-3, 3, (sx + 1, sy + 2)).astype("float32")

nu, nv = advection(u, v)


if __name__ == '__main__':
    show()
