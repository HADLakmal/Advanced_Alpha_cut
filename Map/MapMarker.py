from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

m= Basemap(projection='mill',
           llcrnrlat=6.580800891387973,
           llcrnrlon=79.76853095136596,
           urcrnrlat=7.20249391391774,
           urcrnrlon=80.17365180097534,
           resolution='h')


def mark_map(coordinate, partition_count):
    # print(coordinate1)
    # print(coordinate2)
    color = {
        1: 'blue',
        2: 'black',
        3: 'red',
        4: 'orange',
        5: 'yellow',
        6: 'brown',
        7: 'pink',
        8: 'green',

    }
    xpt, ypt = m(float(coordinate[0]), float(coordinate[1]))
    m.plot(xpt, ypt, 'co', markersize = 3, color= color[partition_count])


def get_node_coordinates(node_list, node):
    node = node.split('+')[0]
    for n in node_list:
        if n[0] == node:
            # print(n[1].get('coordinate'))
            return n[1].get('coordinate')


def initiate_partitions(partitioned_graphs, orig_graph):
    print("marking coordinates")
    partition_count = 1

    m.drawcoastlines()
    m.drawcountries()
    for partition in partitioned_graphs:
        node_list = orig_graph.nodes(data=True)
        nodeCount = 0
        if partition_count == 9:
            break
        for node in partition:
            coordinate = get_node_coordinates(node_list, str(node))

            if not coordinate is None:
                # print(coordinate)
                mark_map(coordinate.split(','), partition_count)
        partition_count += 1



    # Graph.draw_graph(G)
    # m.plot(xs, ys, color='red', linewidth= 1)
    # plt.legend(loc=4)
    # color_map()
    plt.title('Western')

    plt.show()