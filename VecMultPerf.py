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
        for i in range(1, depth):
            # start time
            st = time()

            # compute
            np.dot(vec[j, :], vec[i, :].T)
            
            # end time
            et = time()

            # update vector pointer
            j += 1

            # time passed/elapsed
            tp = et - st

            times.append(tp)

    return times
    


def run_perf(min_vec, max_vec, min_depth, max_depth, repeat, pbar_disable=False):
    """
    runs perf over range described above
    """
    # ignore these times
    warm_up = 50

    # number of results
    num_results = (max_vec - min_vec + 1) * (max_depth - min_depth + 1)

    # results
    results = np.zeros((num_results, 3))

    # progress bar
    pbar = tqdm(total=num_results, disable=pbar_disable)

    row = 0
    for dt in range(min_depth, max_depth + 1):
        for vs in range(min_vec, max_vec + 1):

            times = do_vec_mult(vs, dt, repeat + warm_up)

            # ignore warm up
            avg_time = sum(times[warm_up:])/len(times[warm_up:])

            # update result store
            results[row, :] = np.array([dt, vs, avg_time])

            # update pointers
            row += 1

            # update pbar
            pbar.update()

    return results


if __name__ == "__main__":
    results = run_perf(2, 500, 2, 10, 100)

    try:
        save_file = argv[1]
    except IndexError:
        save_file = "vec_perf.csv"

    np.savetxt(save_file, results, delimiter=',')