import numpy
import random

def get_matrix(args, nx_G):
    n_nodes = args.n
    W_mat = numpy.zeros((n_nodes, n_nodes))
    for (u, v, val) in nx_G.edges(data = True):
        W_mat[u][v] = val['weight'] / 2
        W_mat[v][u] = val['weight'] / 2
    return W_mat


class populations:
    def __init__(self, args, n, W, k):
        self.pops = []
        for i in range(args.POPSIZE):
            ent = Entity(n, W, k)
            ent.generate_random()
            self.pops.append(ent)
        self.popsize = args.POPSIZE
        self.pselect = args.PSELECT
        self.pxover = args.PXOVER
        self.pmutation = args.PMUTATION
        self.W = W
        self.n = n
        self.k = k

    def sort_pops(self):
        self.pops = sorted(self.pops)

    def selection(self):
        self.sort_pops()
        selected_count = int(self.popsize * self.pselect)
        self.pops = self.pops[:selected_count]
        self.popsize = selected_count

    def crossover(self):
        self.sort_pops()
        now_popsize = self.popsize
        selected_count = int(self.popsize * self.pxover)
        for idx1 in range(now_popsize - selected_count, now_popsize):
            for idx2 in range(idx1 + 1, now_popsize):
                parent1 = self.pops[idx1].solution
                parent2 = self.pops[idx2].solution
                crossover_point = random.randint(0, len(parent1) - 1)
                child = Entity(self.n, self.W)
                mask1 = numpy.array([1] * crossover_point + [0] * (len(parent1) - crossover_point)).reshape((-1, 1))
                mask2 = numpy.array([0] * crossover_point + [1] * (len(parent1) - crossover_point)).reshape((-1, 1))
                child.update_solution(mask1 * parent1 + mask2 * parent2)
                self.pops.append(child)
                self.popsize = self.popsize + 1

    def mutation(self):
        self.sort_pops()
        selected_count = int(self.popsize * self.pmutation)
        for idx in range(self.popsize - selected_count, self.popsize):
            target = self.pops[idx].solution
            mutation_point = random.randint(0, len(target) - 3)
            start = (mutation_point // 3) * 3
            new_pci = numpy.random.randint(0, self.k, size=3)
            target[start:start + 3] = numpy.array(new_pci).reshape((-1, 1))
            self.pops[idx].update_solution(target)

    def evaluate(self):
        for i in range(self.popsize):
            self.pops[i].fitness = self.pops[i].getFit()

    def report(self, generation):
        self.sort_pops()
        print("No. " + str(generation) + " Generation: Best value" + str(self.pops[0].fitness))

    def keep_MAXGEN_entity(self, args):
        self.sort_pops()
        if self.popsize > args.POPSIZE:
            self.popsize = args.POPSIZE
            self.pops = self.pops[:self.popsize]


class Entity:
    def __init__(self, num_nodes, W, k=3):
        self.k = k
        self.num_nodes = num_nodes
        self.solution = numpy.zeros((num_nodes, 1))
        self.W = W
        self.fitness = 0

    def update_solution(self, solution):
        self.solution = solution
        self.fitness = self.getFit()

    def generate_random(self):
        for i in range(self.num_nodes):
            self.solution[i] = numpy.floor(numpy.random.rand() * self.k)
        self.fitness = self.getFit()

    def getFit(self):
        mod3s = self.solution
        M3 = (self.W * (mod3s - mod3s.T == 0)).sum()
        return M3

    def __lt__(self, other):
        return self.getFit() < other.getFit()


