import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from handler import Handler
import collections
import queue
import json


#H = nx.read_graphml("graph.graphml.xml")
G = nx.read_edgelist('file/edgeTestPickme/edgeList.txt',nodetype=int, data=(('weight',float),))

# data = dict()
# file = str()
# with open("supernodes.txt","r") as raw_data:
#     file = raw_data.read()
# count = -1
# print(file,len(file))
# while(count<len(file)-1):
#     count += 1
#     if ':' in file[count]:
#         start = 0
#         key = file[count - 1:count]
#         value = ''
#         while (count < len(file)-1):
#             count += 1
#             if '[' in file[count]:
#                 start = count
#             if ']' in file[count]:
#
#                 value = file[start+1:count].split(", ")
#                 data[key] = value
#                 break
#
#
# print(data)
        #     key, value = item.split(':', 1)
        #     data[key] = value
        # else:
        #     pass  # deal with bad lines of text here
# array = list(H.nodes())
# G = H.subgraph(array[:1000])

#
# G = nx.Graph()
#
# G.add_edge('a','b',weight=1)
# G.add_edge('e','f',weight=1)
# G.add_edge('e','d',weight=4)
#
# G.add_edge('f','g',weight=1)
# G.add_edge('g','h',weight=1)
# G.add_edge('e','h',weight=1)
# G.add_edge('x','y',weight=1)
# G.add_edge('y','z',weight=1)
# G.add_edge('x','z',weight=1)


#print(nx.adjacency_matrix(G,nodelist = ['c','e']).todense())
def partitioning(G,k):
    A = nx.adjacency_matrix(G)

    # Alpah cut
    M = Handler.alphaCut(A, 1)

    # eigen calculation
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # define K
    partitionSize = k
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
    np.savetxt('test_1.txt', array, fmt='%r')
    print(lables)
    # New partition array
    partitionArray = []
    # get each laplacian matrix
    for k in array:

        # sort = tempEigenvalues
        if (len(k) > 1):
            print(k)
            H = G.subgraph(k)
            r = nx.connected_components(H)
            for i in r:
                partitionArray.append(i)
        else:
            partitionArray.append(k)

    matrix, edgecut1 = Handler.conectivityMatrix(partitionArray, G)
    print(edgecut1)
    edgecut2 = 0
    q = queue.Queue()
    partitionQueue = queue.Queue();
    partitionQueue.put(partitionArray);
    q.put(matrix)
    alpha = Handler.alphaCut(matrix, 0)
    partitionCount = 1
    part = []
    part.append(partitionArray)
    while (partitionCount != partitionSize):
        if (q.empty() is False):
            matrix = q.get()
            if (matrix.shape[0] > 1):
                alpha = Handler.alphaCut(matrix, 0)

                eigenvalues, eigenvectors = np.linalg.eig(alpha)

                tempEigenValues = np.absolute(eigenvalues)
                idx = tempEigenValues.argsort()[:2][::]
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
                # norm = np.linalg.norm(z,axis=1,ord=2)
                # print(norm)
                # z = z.astype(np.float)/norm[:,None]
                # print(z)
                # find k means paritions
                kmeans = KMeans(n_clusters=2, random_state=0).fit(z)
                w = 0
                p1, p2 = [], []
                partition = partitionQueue.get()
                for p in kmeans.labels_:
                    if (p == 0):
                        p1.append(partition[w])
                    else:
                        p2.append(partition[w])
                    w += 1
                put1, tempedge1 = Handler.conectivityMatrix(p1, G)
                put2, tempedge2 = Handler.conectivityMatrix(p2, G)
                edgecut2 += tempedge1 + tempedge2
                part.pop(0)
                if(len(p1)>=len(p2)):
                    partitionQueue.put(p2)
                    partitionQueue.put(p1)

                    q.put(put2)
                    q.put(put1)

                    part.append(p2)
                    part.append(p1)
                else:
                    partitionQueue.put(p1)
                    partitionQueue.put(p2)

                    q.put(put1)
                    q.put(put2)

                    part.append(p1)
                    part.append(p2)

        partitionCount += 1

    partition = []
    for p in part:
        partTemp = []
        for par in p:
            for part in par:
                partTemp.append(part)
        partition.append(partTemp)
    return partition
partition = partitioning(G,2)
print(partition)

for z in partition:
    edgeWeight = 0
    h = G.subgraph(z)
    for (u, v, d) in h.edges(data=True):
        edgeWeight += abs(d['weight'])
    print(edgeWeight)
np.savetxt('test_3.txt', partition,fmt='%r')

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

