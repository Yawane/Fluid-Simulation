from yoannCFD import *
import time


def test(size, pressure_list=[], size_list=[]):
    dx = dy = .1
    rho = 1000

    u = np.random.randint(-3, 3, (size + 2, size + 1)).astype("float64")
    v = np.random.randint(-3, 3, (size + 1, size + 2)).astype("float64")

    dt = dx / (2 * max(np.max(u), np.max(v)))
    # print("dt:", dt)

    # new_u, new_v = advection(u, v, dx, dx, dt)

    print("Get pressure gradient with size", size)
    start_time = time.time()
    p = pressure_gradient(u, v, size, dx, dt, rho)
    end_time = time.time()
    pressure_time = end_time - start_time

    print(f"\nComputation time:\t{end_time - start_time:.3f} seconds")
    pressure_list.append(end_time - start_time)
    size_list.append(size)

    p = test(size)
    start_time = time.time()
    visualize_pressure_correction(u, v, p, size, dx, dt, rho, apply_boundary_conditions)
    end_time = time.time()

    print(f"Rendering time:\t\t{end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    # pressure_list = []
    # other_list = []
    # size_list = []
    #
    # size = 2 ** 3
    # coeff = 1.2
    # while size < 2**7:
    #     test(size, pressure_list, size_list)
    #     # print(pressure_list)
    #     size = int(size * coeff) if int(size*coeff) > size else size + 1
    #     if size > 50:
    #         coeff = 1.1
    #     if size > 90:
    #         coeff = 1.05
    #
    #
    # plt.plot(size_list, pressure_list, "o")
    # plt.xlabel("Size of the field (square field)")
    # plt.ylabel("Computing time (seconds)")
    # plt.title("Computing time for pressure gradient calculation")
    # plt.grid()
    # plt.show()
    # print(pressure_list)
    # print(size_list)
    size = 2**6
    test(size)
