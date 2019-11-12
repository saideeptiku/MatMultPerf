"""
Performance Analysis of 1D vector multiplication  with numpy
"""
import numpy as np
from time import time
from tqdm import tqdm
from sys import argv

def do_vec_mult(vec_size: int, depth: int, repeat: int):
    """
    vect mult
    """

    # vector with random values
    vec = np.random.rand(depth, vec_size)

    # store results here
    times = []

    # repeat the experiment 
    for _ in range(repeat):
        # do vector multiplications
        j = 0
        st = time()
        for i in range(1, depth):
            # start time

            # compute
            np.dot(vec[j, :], vec[i, :].T)
            
            # update vector pointer
            j += 1

        # end time
        et = time()

        # time passed/elapsed
        tp = et - st

        times.append(tp)

    return times
    


def do_mat_mult(mat_size: int, depth: int, repeat: int):
    """
    multiply sqaure matrices
    """

    # vector with random values
    mat = np.random.rand(depth, mat_size, mat_size)

    # store results here
    times = []

    # repeat the experiment 
    for _ in range(repeat):
        # do vector multiplications
        j = 0
        st = time()
        for i in range(1, depth):
            # start time

            # compute
            np.dot(mat[j, :, :], mat[i, :, :])
            
            # update vector pointer
            j += 1

        # end time
        et = time()

        # time passed/elapsed
        tp = et - st

        times.append(tp)

    return times
    


def run_perf(vector_range, depth_range, repeat, pbar_disable=False):
    """
    runs perf over range described above
    """
    # ignore these times
    warm_up = 100

    # number of results
    num_results = len(list(vector_range)) * len(list(depth_range))

    # results
    results = np.zeros((num_results, 3))

    # progress bar
    pbar = tqdm(total=num_results, disable=pbar_disable)

    row = 0
    for dt in depth_range:
        for vs in vector_range:

            times = do_mat_mult(vs, dt, repeat + warm_up)

            # ignore warm up
            avg_time = sum(times[warm_up:])/len(times[warm_up:])

            # update result store
            results[row, :] = np.array([dt, vs, avg_time])

            # update pointers
            row += 1

            # update pbar
            pbar.update()

    pbar.close()
    return results


if __name__ == "__main__":

    vect_range = range(1, 300 + 1, 10)
    depth_range = range(2, 10 + 1, 1)

    print(argv)

    results = run_perf(vect_range, depth_range, 1000)

    try:
        save_file = argv[1]
    except IndexError:
        save_file = "vec_perf.csv"

    np.savetxt(save_file, results, delimiter=',')