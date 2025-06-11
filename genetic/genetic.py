from .utils import get_matrix
import time
from .utils import get_matrix, populations
import numpy as np

def genetic(args, graph):
    W = get_matrix(args, graph)
    POPS = populations(args, W.shape[0], W, args.k)
    
    t0 = time.time()
    for generation in range(args.MAXGENS):
        if (time.time() - t0) / (generation+1) > 15:
            if args.save:
                with open("./res/genetic_time.txt", "a") as ff:
                    ff.write("numpy.inf, ")
            return np.inf
        POPS.selection()

        POPS.crossover()

        POPS.mutation()

        POPS.evaluate()

        POPS.keep_MAXGEN_entity(args)

        POPS.report(generation)
    solution = POPS.pops[0].solution
    runtime = time.time() - t0
    print("Time: " + str(time.time() - t0) + "s")
    if args.save:
        with open("./res/genetic_time.txt", "a") as ff:
            ff.write(str(runtime) + ", ")
    return solution
