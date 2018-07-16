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


partitionSize = 7
# G = nx.read_graphml("graphHML/Osmnx_large_trips_graph_75000.graphml")
# data = 'trips'
G = nx.read_edgelist('file/edgeTestPickme/edgeList.txt',nodetype=int, data=(('weight',float),))
data = 'weight'
print(len(G.nodes()))
weight_size = G.size(weight='weight')
print(weight_size)
edgeWeight= 0
for (u,v,d) in G.edges(data=True):
    edgeWeight += abs(d[data])
print(edgeWeight)

with open("test_5.txt") as f:
    datas = f.read()
# print(data.split(']')[0])
partitionArray = []
for k in range(0, 238):
    partition_data = datas.split('\n')[k].replace('{','').replace('}', '').split(', ')

    tempartition = []
    for i in partition_data:
        tempartition.append(int(i))

    partitionArray.append(tempartition)

adjecencyMatrix,edgecut = Handler.conectivityMatrix(partitionArray,G,data)
# sort the partition array
sorter = []
for e in partitionArray:
    sorter.append(len(e))
sorter = np.asarray(sorter)
idSorter = sorter.argsort()[::-1][:]
tempPartitionArray = np.asarray(partitionArray)
tempPartitionArray = tempPartitionArray[idSorter]
partitionArray = []
for a in tempPartitionArray:
    partitionArray.append(a)
tempPartitionArray = partitionArray.copy()

dict = dict()
for i in range(0,partitionSize):
    dict[i] = 0
upperbound = int(edgeWeight/(partitionSize+1))
print(edgecut,upperbound)
condition = True
while(condition):
    #define partition array
    partitionArray = tempPartitionArray.copy()
    #zero the dic values
    for i in range(0, partitionSize):
        dict[i] = 0
    # choose the partition to merge begin
    part = []
    indexer = []
    for i in range(0, partitionSize):
        p = []
        if i == 0:
            part.append(p)
            id = Handler.partitionRandom(partitionArray, indexer)
            part[i].append(partitionArray[id])
            del partitionArray[id]
        else:
            indexer = []
            for z in range(0, len(partitionArray)):
                id = Handler.partitionRandom(partitionArray, indexer)
                indexer.append(id)
                if (Handler.edgeConectivity(G, part, partitionArray[id],data) == 0):
                    part.append(p)
                    part[i].append(partitionArray[id])
                    del partitionArray[id]
                    break
    tempPartitionSize = 0
    while (len(partitionArray) != 0):
        cont = 0
        for k in partitionArray:
            max, id = 0, -1
            #dictionary upper bound
            dicUpper = 0
            for v in range(0, partitionSize):
                dicUpper += dict[v]

            dicUpper = dicUpper / partitionSize
            for r in range(0, len(part)):
                edgeconectivity = Handler.interPartitionConectivity(G, part[r], k,data)
                if (edgeconectivity > max) & (dicUpper > dict[r]):
                    dicUpper = dict[r]
                    id = r
                    max = edgeconectivity
            if (id == -1):
                for r in range(0, len(part)):
                    edgeconectivity = Handler.interPartitionConectivity(G, part[r], k,data)
                    if (edgeconectivity > max):
                        id = r
                        max = edgeconectivity
            if (id != -1):
                part[id].append(k)
                partTemp = []
                for par in part[id]:
                    for p in par:
                        partTemp.append(p)
                H = G.subgraph(partTemp)
                edgeWeight = 0
                for (u, v, d) in H.edges(data=True):
                    edgeWeight += abs(d[data])
                dict[id] = edgeWeight
                del partitionArray[cont]
            cont += 1

        if (len(partitionArray) == tempPartitionSize):
            break
        tempPartitionSize = len(partitionArray)

        # edgecut is in define threshold
    print(len(partitionArray))
    if(len(partitionArray)==32):
        print(partitionArray)
    dicUpper = 0
    for v in range(0, partitionSize):
        dicUpper += dict[v]
        print(dict[v])

    dicUpper = dicUpper/partitionSize
    conditionCount = 0
    for r in range(0, partitionSize):
        if (dicUpper - dicUpper * 0.3 <= dict[r] <= dicUpper + dicUpper * 0.3):
            conditionCount += 1
    if (conditionCount == partitionSize):
        edgeWeight = 0
        for z in partitionArray:
            h = G.subgraph(z)
            for (u, v, d) in h.edges(data=True):
                edgeWeight += abs(d[data])
            print(edgeWeight)
        condition = False
        break
"""
part = []
degreeMatrix = Handler.degreeMatrix(G,partitionArray)
adjecencyMatrix,edgecut = Handler.conectivityMatrix(partitionArray,G)
print(adjecencyMatrix)
laplacian = np.subtract(degreeMatrix,adjecencyMatrix)
eigenvalues, eigenvectors = np.linalg.eig(laplacian)

tempEigenValues = eigenvalues

idx = tempEigenValues.argsort()[:1][::]
eigenValues = tempEigenValues[idx]
eigenVectors = eigenvectors[:, idx]
kmeans = KMeans(n_clusters=3, random_state=0).fit(eigenVectors)
part = []
w=0
p1, p2,p3 = [], [],[]
for p in kmeans.labels_:
    if (p == 0):
        p1.append(partitionArray[w])
    elif(p==1):
        p2.append(partitionArray[w])
    else:
        p3.append(partitionArray[w])
    w += 1
part.append(p1)
part.append(p2)
part.append(p3)

"""
'''
matrix,edgecut1 = Handler.conectivityMatrix(partitionArray,G)
edgecut2 = 0
put,edgeConectivity = Handler.conectivityMatrix(partitionArray,G)
alpha = Handler.alphaCut(matrix,0)
partitionCount = 1
part =[]
np.set_printoptions(threshold=np.nan)
indexer = []
p1 = []
p2 = []
p3 = []
#part.append(partitionArray)
#partitioning in to two array
max , id = 0,0
for i in range(0,len(partitionArray)):
    if len(partitionArray[i])>max:
        max = len(partitionArray[i])
        id = i
partition = partitionArray[id]
p1.append(partition)
del partitionArray[id]

max, id = 0, 0
for i in range(0, len(partitionArray)):
    if len(partitionArray[i]) > max:
        max = len(partitionArray[i])
        id = i
partition = partitionArray[id]
p2.append(partition)
del partitionArray[id]

max, id = 0, 0
for i in range(0, len(partitionArray)):
    if len(partitionArray[i]) > max:
        max = len(partitionArray[i])
        id = i
partition = partitionArray[id]
p3.append(partition)
del partitionArray[id]
cond = False
while(len(partitionArray)!=0):
    counter = 0
    for r in range(len(partitionArray)-1,-1,-1):
        if (Handler.edgeConectivity(G,p1,partitionArray[r])+Handler.edgeConectivity(G,p2,partitionArray[r])+Handler.edgeConectivity(G,p3,partitionArray[r]))==0:
            print("continue",len(partitionArray),counter)
            if(r==0)&(counter==len(partitionArray)-1):
                print(partitionArray)
                cond = True
                break
            counter+=1
            continue
        if (Handler.edgeConectivity(G,p1,partitionArray[r])>=Handler.edgeConectivity(G,p2,partitionArray[r]))&(Handler.edgeConectivity(G,p1,partitionArray[r])>=Handler.edgeConectivity(G,p3,partitionArray[r])):
            p1.append(partitionArray[r])
            del partitionArray[r]
        elif (Handler.edgeConectivity(G,p2,partitionArray[r])>=Handler.edgeConectivity(G,p1,partitionArray[r]))&(Handler.edgeConectivity(G,p2,partitionArray[r])>=Handler.edgeConectivity(G,p3,partitionArray[r])):
            p2.append(partitionArray[r])
            del partitionArray[r]
        elif (Handler.edgeConectivity(G,p3,partitionArray[r])>=Handler.edgeConectivity(G,p1,partitionArray[r]))&(Handler.edgeConectivity(G,p3,partitionArray[r])>=Handler.edgeConectivity(G,p2,partitionArray[r])):
            p2.append(partitionArray[r])
            del partitionArray[r]
        counter-=1
        print(len(partitionArray))
    if(cond):
        break
part.append(p1)
part.append(p2)
part.append(p3)
'''
partition = []
for p in part:
    partTemp = []
    for par in p:
        for part in par:
            partTemp.append(part)
    partition.append(partTemp)
for z in partition:
    h = G.subgraph(z)
    edgeWeight = 0
    for (u, v, d) in h.edges(data=True):
        edgeWeight += abs(d['weight'])
    print(edgeWeight)
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

