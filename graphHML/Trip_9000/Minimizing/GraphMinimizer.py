
from random import choice
from  Utils import GraphUtils

node_limit = 9000
merged_nodes = []
removed_nodes = []
current_nodes = 0
#randomMatch

def select_neighbor(vertex, graph):
    neighbours = graph.neighbors(vertex)
    max_weight = -1
    highest_n = None
    for n in neighbours:
        edge_data = graph.get_edge_data(vertex, n)
        # print(edge_data)
        if len(edge_data)>1:
            print(edge_data)
            edge_data = edge_data[1]
            graph.remove_edge(vertex, n)
            graph.remove_edge(n, vertex)
            graph.add_edge(vertex, n, trips=edge_data.get('trips'))
            print('aaaaaaaaaaaaaaaaaaa', graph.get_edge_data(vertex, n))
        weight = graph.get_edge_data(vertex, n).get('trips')
        if weight == None:
            weight = 0
        if (weight ==0 and n != vertex):
            highest_n = n
            max_weight = weight
            break
    # print(highest_n, max_weight)
    return highest_n


def connect_neighbors_to_new_node(new_node, vertex, graph, selected_vertex):

    neighbours = graph.neighbors(vertex)
    # print(neighbours)
    print(selected_vertex, graph.number_of_nodes())
    removed_nodes.append(vertex)

    for n in list(neighbours):
        if not n == selected_vertex:
            # graph.add_edge(new_node, n1, weight=graph.get_edge_data(selected_neighbour, n1).get("weight", 0))
            edge_data = graph.get_edge_data(vertex, n)
            # print('qqqqqqqq', edge_data)
            if len(edge_data)>1:
                weight = edge_data[1].get('trips')
            else:
                weight = edge_data[0].get('trips')
            # print(weight)
            Utils.addEdge(graph, new_node, n, weight)
            # print('wwwwwwwww', graph.get_edge_data(new_node, n))


def merge_nodes(vertex1, vertex2, graph):
    new_node = str(vertex1) + '+' + str(vertex2)
    graph.add_node(new_node)

    connect_neighbors_to_new_node(new_node, vertex1, graph, vertex2)
    graph.remove_node(vertex1)

    connect_neighbors_to_new_node(new_node, vertex2, graph, vertex1)
    graph.remove_node(vertex2)
    merged_nodes.append({"merged": [vertex1, vertex2], "new_node": new_node})
    # print(new_node)
    # print(matched_nodes)
    # print(merged_nodes)
    return graph


def is_in_merge_limit(node):
    merged_list = node.split('+')
    if len(merged_list) >15:
        return False
    return True

def remove_multiedges(graph):
    for u, v, edge_data in graph.edges(data=True):
        print(edge_data)
        if len(edge_data) > 1:
            print(edge_data)
            edge_data = edge_data[1]
            graph.remove_edge(u, v)
            graph.remove_edge(v, u)
            graph.add_edge(u, v, trips=edge_data.get('trips'))
            print('aaaaaaaaaaaaaaaaaaa', graph.get_edge_data(u, v))


def random_match(graph):
    new_graph = graph.copy()
    current_nodes = new_graph.number_of_nodes()
    print(new_graph.edges(data=True))
    for n1, n2, d in new_graph.edges(data=True):
        d.pop('0', None)


    count = 0
    print('Minimizing the trip graph...')
    # random_vertex1, random_vertex2 = GraphUtils.get_highest_weighted_edge_nodes(graph)

    while current_nodes > node_limit:
        current_nodes = new_graph.number_of_nodes()
        count += 1
        nodes = new_graph.nodes()
        random_vertex = choice(list(nodes))
        while random_vertex in removed_nodes:
            random_vertex = choice(list(nodes))

        selected_neighbour = select_neighbor(random_vertex, new_graph)
        if (is_in_merge_limit(random_vertex) and is_in_merge_limit(selected_neighbour)):
            # print(random_vertex, selected_neighbour)
            new_graph = merge_nodes(random_vertex, selected_neighbour, new_graph)

        if count > graph.number_of_nodes()*2:
            break

    # print(new_graph.edges(data=True))
    remove_multiedges(new_graph)


    return new_graph