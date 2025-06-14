import numpy as np
import networkx as nx
import os
import glob
from tqdm import tqdm
import scipy


def load_dimacs_graph(path):
    f = open(path, 'r')
    g = nx.Graph()
    for line in f:
        s = line.split()
        if s[0] == 'p':
            g.add_nodes_from(range(int(s[2])))
        elif s[0] == 'e':
            if len(s) == 4:
                g.add_edge(int(s[1]) - 1, int(s[2]) - 1, weight=int(s[3]))
            else:
                g.add_edge(int(s[1])-1, int(s[2])-1)

    f.close()
    return g


def write_dimacs_graph(graph, path):
    f = open(path, 'w')
    
    f.write(f'p edge {graph.number_of_nodes()} {graph.number_of_edges()}\n')

    for u, v in graph.edges():
        f.write(f'e {int(u)+1} {int(v)+1}\n')

    f.close()


def load_graphs(path):
    """
    Loads the graphs from all '.adj' files in NetworkX adjacency list format
    :param path: The pattern under which to look for .adj files
    :return: A list of NetworkX graphs
    """
    paths = glob.glob(os.path.join(path, '*.dimacs'), recursive=True)
    graphs = [load_dimacs_graph(p) for p in tqdm(paths)]
    names = [os.path.basename(p) for p in paths]
    return names, graphs


def load_dimacs_cnf(path, weighted=False):
    """
    Loads a cnf formula from a file in dimacs cnf format
    :param path: the path to a .cnf file in dimacs format
    :return: The formula as a list of lists of signed integers. 
             I.E. ((X1 or X2) and (not X2 or X3)) is [[1, 2], [-2, 3]]
    """
    file = open(path, 'r')
    f = []
    if weighted:
        weights = []
    for line in file:
        s = line.split()
        if not len(s) == 0 and not line[0] == 'c' and not s[0] == 'p':
            if s[-1] == '0' and len(s) > 1:
                if weighted:
                    weight = int(s[0])
                    weights.append(weight)
                    clause = [int(l) for l in s[1:-1]]
                else:
                    clause = [int(l) for l in s[:-1]]
                f.append(clause)
    file.close()
    if weighted:
        return f, weights
    else:
        return f


def write_dimacs_cnf(f, path):
    """
    Stores a cnf formula in the dimacs cnf format
    :param f: The formula as a list of lists of signed integers.
    :param path: The path to a file in which f is will be stored
    """
    file = open(path, 'w')
    
    num_v = np.max([np.max(np.abs(clause)) for clause in f])
    num_c = len(f)
    file.write(f'p cnf {num_v} {num_c}\n')

    for clause in f:
        line = ''
        for l in clause:
            line += f'{l} '
        line += '0\n'
        file.write(line)
    file.close()
    return f


def load_formulas(path, weighted=False):
    """ Loads cnf formulas from all .cnf files found under the pattern 'path' """
    paths = glob.glob(os.path.join(path, f'**/*.{"wcnf" if weighted else "cnf"}'), recursive=True)
    formulas = [load_dimacs_cnf(p, weighted) for p in tqdm(paths)]
    names = [os.path.basename(p) for p in paths]
    return names, formulas


def load_mtx(path):
    with open(path, 'r') as f:
        g = nx.Graph()
        weights = []
        first_line = True
        for line in f:
            if not line[0] == '%':
                s = line.split()
                if first_line:
                    g.add_nodes_from(range(int(s[0])))
                    first_line = False
                else:
                    g.add_edge(int(s[0]) - 1, int(s[1]) - 1)
                    if len(s) > 2:
                        weights.append(int(s[2]))

    if len(weights) < g.number_of_edges():
        weights = None
    else:
        weights = np.int64(weights)
    return g, weights


def load_mc(path):
    with open(path, 'r') as f:
        g = nx.Graph()
        weights = []

        for line in f:
            s = line.split()
            if s[0] == 'p':
                g.add_nodes_from(range(int(s[2])))
            elif s[0] == 'e':
                g.add_edge(int(s[1]) - 1, int(s[2]) - 1)
                weights.append(int(s[3]))
    weights = np.int64(weights)
    return g, weights


def load_txt(path):
    with open(path, "r") as f:
        g = nx.Graph()
        weights = []
        line_ct = 0
        for line in f:
            s = line.split()
            if line_ct == 0:
                num_nodes = int(s[0])
                for i in range(num_nodes):
                    g.add_node(i)
            else:
                g.add_edge(int(s[0])-1, int(s[1])-1, weight=int(s[2]))
                weights.append(int(s[2]))
            line_ct = line_ct + 1
    return g, np.array(weights)

def load_col(path):
    with open(path, "r") as f:
        g = nx.Graph()
        weights = []
        for line in f:
            if line.split()[0] == "c":
                pass
            elif line.split()[0] == "p":
                _, _, num_nodes, __ = line.split()
                for i in range(int(num_nodes)):
                    g.add_node(i)
            elif line.split()[0] == "e":
                _, l, r = line.split()
                if int(l)<int(r):
                    g.add_edge(int(l)-1, int(r)-1, weight=1)
                    weights.append(1)
    return g, np.array(weights)
