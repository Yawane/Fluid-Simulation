from yoannCFD import *
from yoannCFDPlot import *


if __name__ == "__main__":
    size = 2**5
    dx = dy = 1
    rho = 1000

    u = np.random.randint(-3, 3, (size + 2, size + 1)).astype("float64")
    v = np.random.randint(-3, 3, (size + 1, size + 2)).astype("float64")

    u_velocity = compute_horizontal_velocity(u)
    v_velocity = compute_vertical_velocity(v)
    velocity = compute_velocity(u_velocity, v_velocity)

    plot_velocity_field(velocity, dx)

    # dt = dx / (2 * max(np.max(u), np.max(v)))
    #
    # print("Get pressure gradient with size", size)
    # start_time = time.time()
    # p = pressure_gradient(u, v, size, dx, dt, rho)
    # end_time = time.time()
    # pressure_time = end_time - start_time

    # print(f"\nComputation time:\t{end_time - start_time:.3f} seconds")