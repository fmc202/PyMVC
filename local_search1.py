import networkx as nx
import numpy as np
from itertools import repeat


def get_max_edge(G):
    edgelist = list(G.edges)
    edge_deg = []
    for i, j in edgelist:
        edge_deg.append(G.degree(i) + G.degree(j))
    ind = np.argmax(edge_deg)
    endnodes = edgelist[ind]
    return endnodes


class RunExperiments:
    def generate_graph(self, filename):
        G = nx.Graph()
        with open(filename) as f:
            line1 = f.readline()
            V, E, _ = map(int, line1.split())
            for i in range(1, V + 1):
                string = f.readline()
                if not string:
                    G.add_node(i)
                else:
                    G.add_edges_from(zip(map(int, string.split()), repeat(i)))
        return G

    def approx(self, G):
        mvc = []
        while not nx.is_empty(G):
            endnode = get_max_edge(G)
            mvc.extend(endnode)
            G.remove_nodes_from(endnode)
        return mvc

    # Local Search part,cutoff=6
    # initial_ans is the approx result
    def minus(self, G, initial_ans, cutoff):
        start_time = time.time()
        while time.time() - start_time < cutoff:
            c = Counter()
            rn = r.sample(range(len(initial_ans)), 5)
            for each in rn:
                delete_point = initial_ans[each]
                set1 = set(G.neighbors(delete_point))
                set2 = set(initial_ans)
                set3 = set1 - set2
                c.update(set3)
            if len(c) < 5:
                for every in rn:
                    set2.remove(initial_ans[every])
                for each in list(c):
                    set2.add(each)
                return set2
        return 0

    def LS(self, G, initial_ans, cutoff):
        temp = 1
        while temp:
            temp = self.minus(G, initial_ans, cutoff)
            if temp:
                initial_ans = list(temp)
        return initial_ans


if __name__ == "__main__":
    filename = "./DATA/DATA/dummy1.graph"
    run1 = RunExperiments()
    G = run1.generate_graph(filename)
    mvc_approx = run1.approx(G)
    print(mvc_approx)
