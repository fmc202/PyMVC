import numpy as np
import random
import time
import networkx as nx
import sys


class RunExperiments:
    def parse_edges(self, filename):
        f = open(filename, "r")
        line = f.readline().split()
        verticles = int(line[0])
        edges = int(line[1])
        Graph = dict()
        for i in range(1, verticles + 1):
            line = f.readline()
            if line != "":
                line = list(map(int, line.split()))
                Graph[i] = line
            else:
                Graph[i] = []
        return Graph

    def branch_and_bound(self, Graph):
        sys.setrecursionlimit(10 ** 6)
        time_limit = 10 * 60
        start_time = time.time()

        n = len(Graph)
        graph = nx.Graph()
        upper_bound = float("inf")
        best_sol = []

        # contruct the graph
        for i in range(1, n + 1):
            if Graph[i]:
                graph.add_node(i)
            else:
                continue
            for neighbor in Graph[i]:
                if not (graph.has_edge(i, neighbor) or graph.has_edge(neighbor, i)):
                    graph.add_edge(i, neighbor)

        graph = graph.to_undirected()

        def lower_bound(graph):
            if nx.is_empty(graph):
                return 0
            max_degree = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][1]
            return max_degree * 1.0 / graph.size()

        # branch and bound main part
        def recurse(cur_node, sol):
            nonlocal upper_bound
            nonlocal graph
            cur_time = time.time()
            if cur_time - start_time > time_limit:
                return
            graph.remove_node(cur_node)
            lb = lower_bound(graph)
            # if all edges are covered, update the upper bound
            if nx.is_empty(graph):
                if len(sol) < upper_bound:
                    upper_bound = len(sol)
                    best_sol = sol
                    print("current best upper_bound: {} | time_used: {}".format(upper_bound, cur_time - start_time))
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

        return upper_bound, best_sol

    def Heuristics(self, Graph):
        return


runexp = RunExperiments()
path = "C:/Users/Ryan/OneDrive/cse6140/Finalproject/DATA"
# ['as-22july06.graph','as-22july06.graph','dummy1.graph','dummy2.graph','email.graph','football.graph','hep-th.graph','jazz.graph','karate.graph
