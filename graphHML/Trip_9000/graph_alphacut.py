import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from handler import Handler
import collections
import queue
import operator
import csv


def partition(partitionSize, G):
    # print(nx.adjacency_matrix(G,nodelist = ['c','e']).todense())
    partitionNodeArray = []
    for a in G.nodes():
        partitionNodeArray.append(str(a).split("+"))
    H = nx.read_graphml("../Osmnx_large_trips_graph_75000.graphml")
    print("Done partition array")
    adjMat, pickMat = Handler.adjecencyMatrixes(partitionNodeArray,H,'trips')
    with open('matrix_alpha.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in adjMat]

    with open('pickMatrix_alpha.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in pickMat]

    # with open('matrix_alpha.csv', 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     adjecencyMatrix = [[float(e) for e in r] for r in reader]

    A = np.array(pickMat)

    # Alpah cut
    M = Handler.alphaCut(A)
    # eigen calculatio
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # tempEigenValues = np.absolute(eigenvalues)
    tempEigenValues = eigenvalues
    idx = tempEigenValues.argsort()[:partitionSize][::]
    eigenValues = tempEigenValues[idx]
    eigenVectors = eigenvectors[:, idx]

    z = eigenVectors
    # normalize the matrix
    for i in range(0, eigenVectors.shape[0]):
        total = 0
        for j in range(0, eigenVectors.shape[1]):
            total += abs(eigenVectors.item((i, j))) ** 2
        if (total > 0):
            z[i] = z[i] / (total ** (1 / 2))
    # z = np.matrix.transpose(z)
    # find k means paritions
    kmeans = KMeans(n_clusters=partitionSize, random_state=0).fit(z)

    lables = kmeans.labels_
    array = Handler.indexArray(G.nodes(), partitionSize, lables)
    return array

G = nx.read_graphml("../minimized_trips_9000.graphml")
# G = nx.read_edgelist('file/edgeTestPickme/edgeList.txt', nodetype=int, data=(('weight', float),))
print(len(G.nodes()))

# define K
partitionSize = 100
array = partition(partitionSize, G)
# array = list(G.nodes())
# print(len(array))
# G = G.subgraph(array[:1000])
# array = []
# while True:
#     array = partition(partitionSize, G)
#     max = 0
#     for k in array:
#         if(len(k)>max):
#             max = len(k)
#     if(max<=G.nodes()*0.1):
#         break
#     else:
#         partitionSize +=partitionSize+ (1-(((len(G.nodes()*0.1))/max)))*100

# while minValue*2<maxValue:
#     if minValue==0:
#         array = partition(partitionSize, G, G.nodes())
#         dictry = dict()
#         for c in range(0, len(array)):
#             print(len(array[c]))
#             dictry[c] = len(array[c])
#         maxValue = max(dictry.items(), key=operator.itemgetter(1))[1]
#         minValue = min(dictry.items(), key=operator.itemgetter(1))[1]
#         maxArray = max(dictry.items(), key=operator.itemgetter(1))[0]
#     else:
#         for k in partition(partitionSize, G, array[maxArray]):
#             array.append(k)
#         del array[maxArray]
#     dictry = dict()
#     for c in range(0, len(array)):
#         print(len(array[c]))
#         dictry[c] = len(array[c])
#     print(".....")
#     maxValue = max(dictry.items(), key=operator.itemgetter(1))[1]
#     maxArray = max(dictry.items(), key=operator.itemgetter(1))[0]



np.savetxt('alpha_1_2.txt', array, fmt='%r')
# New partition array
partitionArray = []
# get each laplacian matrix
for k in array:
    # print(nx.adjacency_matrix(G,nodelist = k).todense())


    # sort = tempEigenvalues
    if (len(k) > 1):
        print(k)
        H = G.subgraph(k)
        r = nx.connected_components(H)
        for i in r:
            partitionArray.append(i)
    else:
        partitionArray.append(k)
np.savetxt('alpha_2_whole_map.txt', partitionArray, fmt='%r')


"""

pos=nx.spring_layout(G)
# edges
nx.draw_networkx_edges(G,pos,
                        width=1)
# labels
nx.draw_networkx_labels(G,pos,font_size=2,font_family='sans-serif')

nx.draw_networkx_nodes(G,pos,node_size=1)
plt.show()


#show partitioning nodes

for i in partition:


    nx.draw_networkx_nodes(G,pos,nodelist=i,node_size=1)
    nx.draw_networkx_labels(G,pos,font_size=2,font_family='sans-serif')
    nx.draw_networkx_edges(G,pos,nodelist = i,
                        width=1)


    plt.show()

"""

