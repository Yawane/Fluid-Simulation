from yoannCFD import *
from yoannCFDPlot import *
import matplotlib.animation as animation



if __name__ == "__main__":
    size = 2**5
    dx = dy = .1
    rho = 1000
    current_time = 0.0
    dt = .01

    u = np.zeros((size + 2, size + 1)).astype('float64')
    v = np.zeros((size + 1, size + 2)).astype('float64')

    walls = np.zeros((size, size)).astype('int16')
    top_left = (size//2 - 5, size//4 - 5)  # Row 2, Column 3
    bottom_right = (size//2 + 5, size//4 + 5)  # Row 6, Column 8

    # Fill the rectangle with 1s
    walls[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1] = 1

    plt.figure(figsize=(19, 10))
    x = np.arange(size) * dx
    y = np.arange(size, 0, -1) * dy
    X, Y = np.meshgrid(x, y)

    n = 150
    for i in range(n):
        # vertical field
        vel_u = compute_horizontal_velocity(u)
        vel_v = compute_vertical_velocity(v)
        vel = np.sqrt(vel_u**2 + vel_v**2)

        # plt.subplot(121)
        plt.pcolormesh(X, Y, vel, cmap='jet', vmin=0, vmax=3.5)
        # plt.pcolormesh(X, Y, vel_u, cmap='jet')
        plt.colorbar()
        plt.title(f"Time: {current_time:.3f}")
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')

        # divergence
        # plt.subplot(122)
        # plt.pcolormesh(X, Y, compute_divergence(u, v, size, dx), cmap='jet')
        # plt.streamplot(X, Y, vel_u, vel_v, color='blue', density=1.5)
        # plt.colorbar()

        # next step
        # dt = min(dx / np.max(vel), dx**2 / (4*1.5e-5)) if current_time > .01 else .01
        u, v = advection(u, v, dx, dx, dt)
        # adding external force
        horizontal_force, vertical_force = np.zeros_like(u), np.zeros_like(v)
        # u[1 * u.shape[0]//2, 0] = 1
        # u[3 * u.shape[0]//8 + 2, -1] = -1
        u[1, 0:2] = 1
        v[0:2, 1] = 1


        apply_external_forces(v, vertical_force, u, horizontal_force, dt)
        apply_pressure_correction(u, v, size, dx, dt, rho, apply_boundary_conditions)
        current_time += dt

        plt.show()