import numpy
import time
import copy
from .utils import get_matrix


def bqp(args, graph):
    W = get_matrix(args, graph)
    
    X0 = numpy.ones((args.n, args.k)) / args.k
    
    iter = 0
    
    t0 = time.time()
    while True:
        flag = False
        optimal0 = numpy.ones((args.n, args.k)) * numpy.inf
        optimal1 = numpy.ones((args.n, args.k)) * numpy.inf
        iter = iter + 1
        W1 = (W + W.T) @ X0

        for i in range(args.n):
            for k in range(args.k):
                if X0[i][k] != 0 and X0[i][k] != 1:
                    flag = True
                    X0_try0 = copy.deepcopy(X0)

                    tmp = X0_try0[i][k]
                    X0_try0[i][k] = 0
                    X0_try0[i][X0_try0[i]!=0] = 1 / (1 / tmp - 1)

                    X0_try1 = copy.deepcopy(X0)
                    X0_try1[i][k] = 1
                    X0_try1[i][X0_try1[i] != 1] = 0

                    z_ik_try0 = X0_try0[i, :]
                    z_ik_try1 = X0_try1[i, :]

                    Delta_0 = W1[i].dot(z_ik_try0 - X0[i])
                    Delta_1 = W1[i].dot(z_ik_try1 - X0[i])

                    optimal0[i][k] = Delta_0
                    optimal1[i][k] = Delta_1
                if time.time() - t0 > 1800:
                    if args.save:
                        with open("./res/bqp_time.txt", "a") as ff:
                            ff.write("numpy.inf, ")
                    return numpy.inf
        if not flag:
            break
        min_index0_i, min_index0_j = numpy.unravel_index(numpy.argmin(optimal0), optimal0.shape)
        min_index1_i, min_index1_j = numpy.unravel_index(numpy.argmin(optimal1), optimal1.shape)
        if optimal0[min_index0_i][min_index0_j] < optimal1[min_index1_i][min_index1_j]:
            tmp = X0[min_index0_i][min_index0_j]
            X0[min_index0_i][min_index0_j] = 0
            X0[min_index0_i][X0[min_index0_i] != 0] = 1 / (1 / tmp - 1)
        else:
            X0[min_index1_i][min_index1_j] = 1
            X0[min_index1_i][X0[min_index1_i] != 1] = 0
    result = numpy.argmax(X0, axis=1)
    runtime = time.time() - t0
    if args.save:
        with open("./res/bqp_time.txt", "a") as ff:
            ff.write(str(runtime) + ", ")
    return result