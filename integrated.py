import networkx as nx
import time
import sys
import random as r
import math
import os
from os import path
import networkx as nx
from networkx.algorithms.approximation import vertex_cover
import time
import sys
import random
from os import listdir
from os.path import isfile, join
import argparse
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

# THE WRAPPER CLASS THAT INCLUDES DATA PROCESSING AND ALL ALGORIHTMS
class RunExperiments:
    def __init__(self, cut_off, random_seed=None):
        self.start_time = time.time()
        self.cut_off = cut_off
        self.random_seed = random_seed

    # Read file and produce corrsponding newtorkx graph
    def read_graph(self, filename):
        G = nx.Graph()
        with open(filename, "r") as vertices:
            V, E, Temp = vertices.readline().split()
            for i, line in enumerate(vertices):
                vertex_data = list(map(lambda x: int(x), line.split()))
                for v in vertex_data:
                    G.add_edge(i + 1, v)
        return G, int(V), int(E)

    # Helper function that checks whehter the current partial vertex cover can be refined
    # (cover the same edges with fewer vertices)
    def minus_1(self, G, initial_ans, cut_off):
        start_time = time.time()
        while time.time() - start_time < cut_off and time.time() - self.start_time < self.cut_off:
            c = Counter()
            rn = r.sample(range(len(initial_ans)), 5)
            set2 = set(initial_ans)
            for every in rn:
                set2.remove(initial_ans[every])
            for each in rn:
                delete_point = initial_ans[each]
                set1 = set(G.neighbors(delete_point))
                set3 = set1 - set2
                c.update(set3)
            if len(c) < 5:
                for each in list(c):
                    set2.add(each)
            return set2
        return 0

    # Heurestic solution found by iteration from max to min degree nodes and removing node if still vertex cover after
    def initial_solution(self, G, cutoff):
        start_time = time.time()
        temp_G = list(G)
        VC = sorted(list(G.degree(temp_G)))
        VC_sol = temp_G
        i = 0
        uncov_edges = []
        optvc_len = len(VC_sol)
        while i < len(VC) and (time.time() - start_time) < cutoff:
            flag = True
            for x in G.neighbors(VC[i][0]):
                if x not in temp_G:
                    flag = False
            if flag:
                temp_G.remove(VC[i][0])
            i = i + 1
        return VC_sol

    # Local serach approach 1
    def LS1(self, G, solution_quality):
        if self.random_seed:
            random.seed(self.random_seed)
        self.start_time = time.time()
        # The cutoff time for finding the initial solutino is 30 seconds
        initial_ans = self.initial_solution(G, 30)
        trace = []
        temp = 1
        output_ans = initial_ans[:]
        while temp and time.time() - self.start_time < self.cut_off:
            temp = self.minus_1(G, output_ans, 20)
            if temp:
                duration = time.time() - self.start_time
                output_ans = list(temp)
                trace.append([str(round(time.time() - self.start_time, 2)), str(len(output_ans))])
                if len(output_ans) <= solution_quality:
                    break
        print("Local Search Algorithm 1: time_used: {} | best ans: {}".format(round(time.time() - self.start_time, 2), len(output_ans)))

        return output_ans, time.time() - self.start_time, trace

    # The evaluation function for Local Search 2: checks the number of remaining uncovered edges with the current partial vertex cover
    def cost(self, G, C):
        A = deepcopy(G)
        A.remove_nodes_from(C)
        return A.number_of_edges()

    # The helper function that checks validity of the vertex cover
    def is_vertex_cover(self, G, C):
        if len(G.edges()) == len(G.edges(C)):
            return True
        else:
            return False

    # The helper function that randonly swap one node from the covering set to its complement
    def one_exchange(self, all, C):
        C_complement = [x for x in all if x not in C]
        u = random.choice(C)
        v = random.choice(C_complement)
        C_copy = C[:]
        C_copy.remove(u)
        C_copy.append(v)
        return C_copy

    # The helper function that performs one_exchange twice
    def two_exchange(self, all, C):
        new_C = self.one_exchange(all, C)
        new_C = self.one_exchange(all, new_C)
        return new_C

    # The helper function that gets the initial solution for Local Search 2
    def heuristic(self, C):
        h_set = []
        for edge in C.edges():
            if not edge[0] in h_set and not edge[1] in h_set:
                node_choice = random.choice([edge[0], edge[1]])
                h_set.append(node_choice)
        return h_set

    # Local Search Approach 2
    def LS2(self, G):
        if self.random_seed:
            random.seed(self.random_seed)

        self.start_time = time.time()
        new_cost = -1
        previous_cost = 100
        all_nodes = G.nodes()
        local_set = self.heuristic(G)
        trace = []

        while time.time() - self.start_time < self.cut_off:
            if new_cost < previous_cost:
                if self.is_vertex_cover(G, local_set):
                    flag = False
                    for element in local_set:
                        temp = local_set[:]
                        temp.remove(element)

                        if self.is_vertex_cover(G, temp):
                            flag = True
                            new_cost = 0
                            local_set = temp[:]
                            break

                    if flag == False:
                        temp = local_set[:]
                        eliminated_element = random.choice(local_set)
                        local_set.remove(eliminated_element)
                        previous_cost = 0
                    end = time.time()
                    print("Local Search Algorithm 2: time_used: {} | current best answer: {}".format(round(time.time() - self.start_time, 2), len(temp)))
                    trace.append([str(round(time.time() - self.start_time, 2)), str(len(temp))])
                    continue
            # Apply local search
            previous_cost = self.cost(G, local_set)
            while True:
                loop_end = time.time()
                loop_time = loop_end - end
                if loop_time < 10:
                    new_C = self.one_exchange(all_nodes, local_set)
                    # print('one exchanged')
                else:
                    new_C = self.two_exchange(all_nodes, local_set)
                    end = time.time()

                new_cost = self.cost(G, new_C)
                if new_cost <= previous_cost:
                    local_set = new_C
                    break
        return temp, time.time() - self.start_time, trace

    def get_max_edge(self, G):
        edgelist = list(G.edges)
        edge_deg = []
        for i, j in edgelist:
            edge_deg.append(G.degree(i) + G.degree(j))
        ind = np.argmax(edge_deg)
        endnodes = edgelist[ind]
        return endnodes

    # Approximation Algorithm
    def approx(self, G):
        self.start_time = time.time()
        mvc = []
        while not nx.is_empty(G):
            endnode = self.get_max_edge(G)
            mvc.extend(endnode)
            G.remove_nodes_from(endnode)
        print("Approximation Algorithm: time used: {} | best_ans: {}".format(round(time.time() - self.start_time, 2), len(mvc)))
        return mvc, time.time() - self.start_time, [[str(round(time.time() - self.start_time, 2)), str(len(mvc))]]

    # Branch and Bound Algorithm
    def branch_and_bound(self, Graph):
        sys.setrecursionlimit(10 ** 6)
        self.start_time = time.time()
        trace = []

        n = Graph.number_of_nodes()
        upper_bound = float("inf")
        best_sol = []

        graph = Graph.to_undirected()

        def lower_bound(graph):
            if nx.is_empty(graph):
                return 0
            max_degree = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][1]

            return max_degree * 1.0 / graph.size()

            # branch and bound main part

        def recurse(cur_node, sol):
            nonlocal upper_bound
            nonlocal graph
            nonlocal best_sol
            cur_time = time.time()
            if cur_time - self.start_time > self.cut_off:
                return
            graph.remove_node(cur_node)
            lb = lower_bound(graph)
            # if all edges are covered, update the upper bound
            if nx.is_empty(graph):
                if len(sol) < upper_bound:
                    upper_bound = len(sol)
                    best_sol = sol
                    trace.append([str(round(cur_time - self.start_time, 2)), str(len(best_sol))])
                    print("BnB: current best upper_bound: {} | time_used: {}".format(upper_bound, round(cur_time - self.start_time, 2)))
                return
            # if current partial solution is impossible, prune this branch
            elif lb + len(sol) >= upper_bound:
                return
            # expand
            candidates = sorted(graph.degree, key=lambda x: x[1], reverse=True)
            for c in candidates:
                edges = list(graph.edges(c[0]))
                recurse(c[0], sol + [c[0]])
                graph.add_node(c[0])
                graph.add_edges_from(edges)
            # start the recursion
            start_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)
            for node in start_nodes:
                edges = list(graph.edges(node[0]))
                recurse(node[0], [node[0]])
                graph.add_node(node[0])
                graph.add_edges_from(edges)
                return best_sol, upper_bound, trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input Parser for BnB, Approximation Algorithm, and Local Searches (Group 25)")
    parser.add_argument("-inst", action="store", type=str, required=True, help="Input Graph Instance Name")
    parser.add_argument("-alg", action="store", type=str, required=True, help="Algorithm Name (BnB | Approx | LS1 | LS2)")
    parser.add_argument("-time", action="store", type=str, required=True, help="Cut-off Running Time")
    parser.add_argument("-seed", action="store", type=str, required=False, help="Random Seed")
    args = parser.parse_args()

    algorithm = args.alg
    filename = args.inst
    cut_off_time = int(args.time)
    seed = int(args.seed) if args.seed else None
    if algorithm == "BnB" or algorithm == "Approx":
        seed = None

    graph_performance = dict(
        zip(
            [
                "DATA/" + f
                for f in [
                    "as-22july06.graph",
                    "delaunay_n10.graph",
                    "email.graph",
                    "football.graph",
                    "hep-th.graph",
                    "jazz.graph",
                    "karate.graph",
                    "netscience.graph",
                    "power.graph",
                    "star.graph",
                    "star2.graph",
                ]
            ],
            [3303, 703, 594, 94, 3926, 158, 14, 899, 2203, 6902, 4542],
        )
    )

    all_algorithms = set(["BnB", "Approx", "LS1", "LS2"])

    if filename not in graph_performance:
        print("Get filename:", filename)
        print("Expected filename:\n", " | ".join(graph_performance.keys()))
        raise ValueError("Invalid Filename")
    if algorithm not in all_algorithms:
        print("Get algorithm name:", filename)
        print("Expected algorithm name:\n", " | ".join(list(all_algorithms)))
        raise ValueError("Invalid Algorithm Name")

    exp = RunExperiments(cut_off=cut_off_time, random_seed=seed)
    graph, _, _ = exp.read_graph(filename)
    if algorithm == "BnB":
        sol, time, trace = exp.branch_and_bound(graph)
    elif algorithm == "Approx":
        sol, time, trace = exp.approx(graph)
    elif algorithm == "LS1":

        sol, time, trace = exp.LS1(graph, graph_performance[filename])
    else:
        sol, time, trace = exp.LS2(graph)

    filename = filename[5:-6]
    sol = list(map(str, sol))
    output_file_name = filename + "_" + algorithm + "_" + str(cut_off_time) + ("_" + str(seed) if seed else "") + ".sol"
    output_trace_file_name = filename + "_" + algorithm + "_" + str(cut_off_time) + ("_" + str(seed) if seed else "") + ".trace"

    with open(output_file_name, "w") as f:
        f.write(str(len(sol)) + "\n")
        f.write(",".join(sol))

    with open(output_trace_file_name, "w") as f:
        f.write(str(round(time, 2)) + "\n")
        for i, t in enumerate(trace):
            f.write(", ".join(t) + ("\n" if i != len(trace) - 1 else ""))
