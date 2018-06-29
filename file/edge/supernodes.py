from sklearn.cluster import KMeans
import numpy as np
import operator
from itertools import combinations

fo = open("edges3.txt", "r")
line = fo.read()
data = line.split("\n")
Ag = {}
a = [[0]*len(data)]*len(data)
coordinatesList=[]
vehicleDensities = {}
for ALines in data:
    ALineElements = ALines.strip().split(" ")
    vehicleDensities[int(ALineElements[0])] = int(ALineElements[3])
    Anodes = ALineElements[1:3]
    adjacentList = []
    row = [0]*len(data)
    for BLines in data:
        Bnodes = BLines.strip().split(" ")[1:3]
        for node1 in Anodes:
            if node1 in Bnodes:
                adjacentList.append(BLines.strip().split(" ")[0])
    Ag[int(ALines.strip().split(" ")[0])] = adjacentList

print("Adjacency matrix created...!")
# print(Ag)

sorted_vehicle_densities = sorted(vehicleDensities.items(), key=operator.itemgetter(1))
sorted_vehicle_density_list = []
sorted_road_segments = []
for eachtupple in sorted_vehicle_densities:
    sorted_vehicle_density_list.append(eachtupple[1])
    sorted_road_segments.append(eachtupple[0])

class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)
def connected_components(nodes):
    # List of connected components found. The order is random.
    result = []

    # Make a copy of the set, so we can modify it.
    nodes = set(nodes)

    # Iterate while we still have nodes to process.
    while nodes:

        # Get a random node and remove it from the global set.
        n = nodes.pop()

        # This set will contain the next group of nodes connected to each other.
        group = {n}

        # Build a queue with this node in it.
        queue = [n]

        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:
            # Consume the next item from the queue.
            n = queue.pop(0)

            # Fetch the neighbors.
            neighbors = n.links

            # Remove the neighbors we already visited.
            neighbors.difference_update(group)

            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)

            # Add them to the group of connected nodes.
            group.update(neighbors)

            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    # Return the list of groups.
    return result

def kmeans(vehicle_density_list, road_segments,k):
    # vehicle_density_list = list(vehicle_density.values())
    # road_segments = list(vehicle_density.keys())
    x = np.array(vehicle_density_list)

    ndarray = [] # centroid values list
    # gathering centroid values; i = (nr/k)*j
    for j in range(1,k+1):
        i = int((len(vehicle_density_list)/k))*j
        # vehicle_density_values = list(vehicle_density.values())
        ndarray.append([vehicle_density_list[i]])
    ndarr = np.array(ndarray, np.float16)

    km = KMeans(n_clusters=k, init=ndarr)
    km.fit(x.reshape(-1, 1))

    dict_clusters = {}
    dict_road_clusters = {}
    kmlabels = list(km.labels_)

    for index in range(len(kmlabels)):
        if kmlabels[index] in dict_clusters:
            dict_clusters[kmlabels[index]].append(vehicle_density_list[index])
            dict_road_clusters[kmlabels[index]].append(road_segments[index])
        else:

            dict_clusters[kmlabels[index]] = [vehicle_density_list[index]]
            dict_road_clusters[kmlabels[index]] = [road_segments[index]]
    # print(dict_clusters)
    return dict_road_clusters

def total_connected_componnts(suggestedPartitionsDict, Ag):
    total = 0
    for partitionIndex in suggestedPartitionsDict:
        total+= connected_component_count(suggestedPartitionsDict[partitionIndex], Ag)
    return total

def connected_component_count(partition, Ag):
    grahpNodeNames = []
    for roadID in partition:
        grahpNodeNames.append('n' + str(roadID))
    node_dict = {}
    nodes = {Data(x) for x in grahpNodeNames}
    for eachNode in nodes:
        node_dict[int(eachNode.name[1:])] = eachNode

    if __name__ == "__main__":
        for roadID1 in partition:
            for roadID2 in partition:
                if int(roadID1) > int(roadID2):
                    if str(roadID2) in Ag[int(roadID1)]:
                        node_dict[int(roadID1)].add_link(node_dict[int(roadID2)])

        # Find all the connected components.
        number = 0
        for components in connected_components(nodes):
            names = sorted(node.name for node in components)
            names = ", ".join(names)
            # print("Group #%i: %s" % (number, names))
            number += 1
        return number

#============================================================================

threshold = 25
min = 999999

for k in range(threshold, threshold+100):
    try:
        cluster_k = kmeans(sorted_vehicle_density_list, sorted_road_segments, k)
        new_count = total_connected_componnts(cluster_k, Ag)
        if (total_connected_componnts(cluster_k, Ag)) < min:
            min = new_count
            optimal_k = k
            optimal_config_vector = cluster_k
    except (IndexError): #to be handled later
        pass

print(optimal_config_vector)

#cluster means of supernodes
number_of_clusters = len(optimal_config_vector)
for cluster_index in range(number_of_clusters):
    cluster_total = 0
    for each in optimal_config_vector[cluster_index]:
        cluster_total += vehicleDensities[int(each)]
    cluster_mean = cluster_total/len(optimal_config_vector[cluster_index])
    # print("cluster mean of supernode", cluster_index, " = ", cluster_mean)

#link weights
def rSubset(arr, r):
	return list(combinations(arr, r))

arr = list(optimal_config_vector.keys())
r = 2
link_weights = {}
for eachlink in rSubset(arr, r):
    link_weights[eachlink] = 0

for cluster_index in range(number_of_clusters):
    selected_node_set = optimal_config_vector[cluster_index]
    # print(selected_node_set)
    for cluster_index2 in range(number_of_clusters):
        to_be_compared = optimal_config_vector[cluster_index2]
        if cluster_index!= cluster_index2:
            for node1 in selected_node_set:
                # node1 = int(eachnode)-1
                for node2 in to_be_compared:
                    # node2 = int(eachothernode)-1
                    if str(node2) in Ag[int(node1)]:
                        # print(node1, node2)
                        l = sorted([int(cluster_index), int(cluster_index2)])
                        # print(tuple(l))
                        if tuple(l) in link_weights:
                            link_weights[tuple(l)] +=1
                        else:
                            link_weights[tuple(l)] = 1
                    # link_weights[(int(cluster_index), int(cluster_index2))]+=1

# print("link weighs: ", link_weights)
for key in link_weights:
    print(key[0], key[1], link_weights[key])
