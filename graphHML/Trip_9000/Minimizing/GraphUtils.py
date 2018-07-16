import networkx as nx
import matplotlib.pyplot as plt
from Utils import CoordinateUtil
from random import choice
import random

def draw_graph(G):
    edge_labels = dict([((u, v,), d['weight'])
                        for u, v, d in G.edges(data=True)])
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           node_color='r', node_size=100)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows=False)
    print(G.number_of_nodes())
    plt.show()


def getRoadCoordinateTuples(road_graph):
    nodes = road_graph.nodes(data=True)
    node_list =[]
    coordinate_tuples = []
    for node in nodes:
        cor_tuple = CoordinateUtil.getTupleWithStringCoordinate(node[1].get('coordinate'))
        coordinate_tuples.append(cor_tuple)
        node_list.append(node[0])

    # print(coordinate_tuples)
    # print(node_list)

    return node_list, coordinate_tuples


def getBaseCoordinateTuples(coordinate_list):
    coordinate_tuples = []
    for c in coordinate_list:
        cor_tuple = CoordinateUtil.getTupleWithStringCoordinate(c)
        coordinate_tuples.append(cor_tuple)
    print(coordinate_tuples)
    return coordinate_tuples


def addEdge(G, u, v, w):
    # print
    # print(G.nodes(data=True))
    if (w == None):
        w = 0
    if (G.has_edge(u, v)):
        # print(G.get_edge_data(u, v))
        trips = G.get_edge_data(u, v).get('trips')
        # weight = G.get_edge_data(u, v).get('weight')
        if(trips == None):
            trips = 0
        # print(G.get_edge_data(u, v).get('trips'))
        G.remove_edge(u, v)
        # print('weight', weight+w)

        G.add_edge(u, v, trips=trips + w)

        # print(G.get_edge_data(u, v))
        # t = G.get_edge_data(u, v)[0].get('trips');
        # print(t)
    else:
        G.add_edge(u, v, trips=w)

    return G

def get_random_vertex(graph):
    random_vertex = choice(list(graph.nodes()))
    while graph.degree(random_vertex) == 0:
        random_vertex = choice(list(graph.nodes()))

    return random_vertex


def get_highest_degree_vertex(graph):
    node_list = sorted(graph.degree(weight='trips'), key=lambda x: x[1], reverse=True)
    print(node_list)
    return node_list[0][0]


def get_first_degree_vertex(graph):
    node_list = sorted(graph.degree(weight='trips'), key=lambda x: x[1])
    # print(node_list)
    # print(node_list)
    for node in node_list:
        if node[1] > -1:
            # print('dsadasdasd', node)
            return node[0]
    print('aaaaaaaaaaaaaa')

def get_trip_path(graph, start_node, dest_node):
    if(nx.nx.has_path(graph,start_node,dest_node)):
        path = nx.shortest_path(graph,source=start_node,target=dest_node)
        return path

def add_path_to_graph(graph, path):
    for i in range(len(path) - 1):
        current_item, next_item = path[i], path[i + 1]
        # print(current_item, next_item)
        graph = addEdge(graph, current_item, next_item, 1)
        # break
    return graph

def markPartitionNumber(graph, partitions):
    # print(graph.nodes(data=True))
    for i in range(0, len(partitions)):
        for n in partitions[i]:
            graph.add_node(n, partition=i)
            # print(i)
    print(graph.number_of_nodes())
    # print(graph.nodes(data=True))

    return graph



def getSampleGraph():
    sample_graph = nx.Graph()
    sample_graph.add_node(1, coordinate='aaaaaaa')
    sample_graph.add_node(2, coordinate='bbbbbb')
    sample_graph.add_node(3, coordinate='cccc')
    sample_graph.add_node(4, coordinate='dddddd')
    sample_graph.add_node(5, coordinate='eeeeee')
    sample_graph.add_node(6, coordinate='ffffff')
    sample_graph.add_node(7, coordinate='gggggg')
    # sample_graph.add_node(8, coordinate='hhhhhh')
    # sample_graph.add_node(9, coordinate='iiiiiiii')
    # sample_graph.add_node(10, coordinate='fffffff')
    sample_graph.add_edge(1, 2, weight=1)
    sample_graph.add_edge(2, 3, weight=2)
    sample_graph.add_edge(3, 4, weight=1)
    sample_graph.add_edge(4, 5, weight=1)
    sample_graph.add_edge(5, 6, weight=1)
    sample_graph.add_edge(6, 7, weight=3)
    sample_graph.add_edge(4, 6, weight=1)
    sample_graph.add_edge(4, 7, weight=1)
    sample_graph.add_edge(2, 6, weight=1)

    draw_graph(sample_graph)
    return sample_graph


def get_random_graph(n ,e):
    g = nx.gnm_random_graph(n, e)
    for (u, v, w) in g.edges(data=True):
        w['weight'] = random.randint(0, 1)
    draw_graph(g)
    return g

def get_highest_weighted_edge_nodes(graph):
    edges = graph.edges(data=True)
    sorted_edges = sorted(edges, key=lambda x: x[2].get('trips'), reverse=True)
    highest_e = sorted_edges[0]
    return highest_e[0], highest_e[1]