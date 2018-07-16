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
import sys

def partition(partitionSize,G,k):
    # print(nx.adjacency_matrix(G,nodelist = ['c','e']).todense())
    A = nx.adjacency_matrix(G,nodelist=k)

    # Alpah cut
    M = Handler.alphaCut(A, 1)
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
    return  array

#H = nx.read_graphml("graph.graphml.xml")
G = nx.read_edgelist('colombo.txt',nodetype=int, data=(('weight',float),))

nodecount = G.nodes()
# define K
partitionSize = 8

array = partition(partitionSize, G, G.nodes())

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


print(array)
np.savetxt('test_1.txt', array,fmt='%r')
# New partition array
partitionArray = []
# get each laplacian matrix
for k in array:
    # print(nx.adjacency_matrix(G,nodelist = k).todense())
    A = nx.laplacian_matrix(G, nodelist=k)
    tempEigenvalues, tempEigenvectors = np.linalg.eig(A.toarray())

    # sort = tempEigenvalues
    if (len(k) > 1):
        counter = collections.Counter(tempEigenvalues)
        p = counter[0]
        if (0 in tempEigenvectors) & (p > 1):
            index = []
            for t in range(0, len(tempEigenvalues)):
                if (tempEigenvalues[t] == 0):
                    index.append(t)
            kmeans = KMeans(n_clusters=p, random_state=0).fit(tempEigenvectors[:, index])
            lables = kmeans.labels_
            arrays = Handler.indexArray(k, p, lables)
            for i in arrays:
                partitionArray.append(i)
        else:
            partitionArray.append(k)
    else:
        partitionArray.append(k)
np.savetxt('test_colombo.txt', partitionArray,fmt='%r')
matrix,edgecut1 = Handler.conectivityMatrix(partitionArray,G)
edgecut2 = 0
put,edgeConectivity = Handler.conectivityMatrix(partitionArray,G)
alpha = Handler.alphaCut(matrix,0)
partitionCount = 1
part =[]
part.append(partitionArray)
while(len(part)!=partitionSize):
    if(len(part)>0):
        max , id = 0,0
        for i in range(0,len(part)):
            if len(part[i])>max:
                max = len(part[i])
                id = i
        matrix, edgeCount = Handler.conectivityMatrix(part[id], G)
        partition = part[id]
        print(partition,len(part))
        del part[id]
        if(matrix.shape[0]>1):
            alpha = Handler.alphaCut(matrix,0)

            eigenvalues, eigenvectors = np.linalg.eig(alpha)

            tempEigenValues = np.absolute(eigenvalues)
            idx = tempEigenValues.argsort()[:2][::]
            eigenValues = tempEigenValues[idx]
            eigenVectors = eigenvectors[:,idx]



            z = eigenVectors

            #normalize the matrix
            for i in range(0,eigenVectors.shape[0]):
                total = 0
                for j in range(0,eigenVectors.shape[1]):
                    total += abs(eigenVectors.item((i,j)))**2
                if(total>0):
                    z[i]=+z[i]/(total**(1/2))
            # norm = np.linalg.norm(z,axis=1,ord=2)
            # print(norm)
            # z = z.astype(np.float)/norm[:,None]
            # print(z)
            #find k means paritions
            kmeans = KMeans(n_clusters=2, random_state=0).fit(z)
            w = 0
            p1,p2 = [],[]
            for p in kmeans.labels_:
                if(p==0):
                    p1.append(partition[w])
                else:
                    p2.append(partition[w])
                w+=1
            # if(int(edgeConectivity*1.2/partitionSize)<len(p1)|int(edgeConectivity*1.2/partitionSize)<len(p2)):
            #     if (int(edgeConectivity * 1.2 / partitionSize) < len(p1) & int(edgeConectivity * 1.2 / partitionSize) < len(p2)):
            #         partitionSize.put(p1)
            #         partitionQueue.put(p2)
            #         swap = True
            #     elif (int(edgeConectivity * 1.2 / partitionSize) < len(p1)):
            #         if(swap):
            #             part = partitionQueue.get()
            #         partitionQueue.put(p2)
            #     else:
            #         partitionQueue.put(p1)
            part.append(p1)
            part.append(p2)

    partitionCount+=1

partition = []
for p in part:
    partTemp = []
    for par in p:
        for part in par:
            partTemp.append(part)
    partition.append(partTemp)
print(partition)
np.savetxt('test_4.txt', partition,fmt='%r')

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

