import networkx as nx
from collections import defaultdict

def plot_network(graph, classes_disp):
    G = nx.Graph()
    V = set()
    E = []
    for k, v in graph.items():
        V.add(k)
        for i in v:
            V.add(i)
            E.append((k, i))

    labels = dict()
    for i in V: labels[i] = \
        classes_disp[i]

    duplicate_dict = defaultdict(lambda: 1)
    for v in V:
        count_before_add = G.number_of_nodes()
        G.add_node(labels[v])
        count_after_add = G.number_of_nodes()
        if count_before_add == count_after_add:
            # duplicate annotation name detected
            duplicate_dict[labels[v]] += 1
            labels[v] = labels[v] + " " + str(duplicate_dict[labels[v]])

    for (u,v) in E:
        G.add_edge(labels[u], labels[v])

    nx.write_graphml(G, 'Chow-Liu.graphml')
