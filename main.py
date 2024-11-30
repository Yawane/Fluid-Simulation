from yoannCFD import *
from yoannCFDPlot import *
import matplotlib.animation as animation



if __name__ == "__main__":
    size = 2**5
    dx = .1
    rho = 1000
    current_time = 0.0
    dt = .01

    u = np.zeros((size + 2, size + 1)).astype('float64')
    v = np.zeros((size + 1, size + 2)).astype('float64')

    plt.figure(figsize=(19, 10))
    x = np.arange(size) * dx
    X, Y = np.meshgrid(x, x)

    n = 100
    for i in range(n):
        # vertical field
        vel_u = compute_horizontal_velocity(u)
        vel_v = compute_vertical_velocity(v)

        # plt.subplot(121)
        plt.pcolormesh(X, Y, vel_u, cmap='jet', vmin=0)
        # plt.pcolormesh(X, Y, compute_horizontal_velocity(u), cmap='jet', vmin=-2, vmax=3)
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
        u, v = advection(u, v, dx, dx, dt)
        # adding external force
        horizontal_force, vertical_force = np.zeros_like(u), np.zeros_like(v) + 9.81
        u[u.shape[0]//2, 0] = 1

        apply_external_forces(v, vertical_force, u, horizontal_force, dt)
        p = pressure_gradient(u, v, size, dx, dt, rho)
        apply_pressure_correction(u, v, p, size, dx, dt, rho, apply_boundary_conditions)
        current_time += dt

        plt.show()