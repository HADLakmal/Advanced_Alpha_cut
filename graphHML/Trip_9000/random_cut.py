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
import csv


partitionSize = 2
G = nx.read_graphml("../Osmnx_large_trips_graph_75000.graphml")
data = 'trips'
# G = nx.read_edgelist('file/edgeTestPickme/edgeList.txt',nodetype=int, data=(('weight',float),))
# data = 'weight'

print(len(G.nodes()))


with open("test_2.txt") as f:
    datas = f.read()
# print(data.split(']')[0])
partitionArray = []
for k in range(0, 601):
    partition_data = datas.split('\n')[k].replace('{','').replace('}', '').split(', ')
    tempartition = []
    for i in partition_data:
        for z in i.split('+'):
            tempartition.append(int(z.replace('\'','')))
    partitionArray.append(tempartition)
np.savetxt('test.txt', partitionArray, fmt='%r')
print("Done")
# for i in data:
#     print(i.split(" "))


# adjecency of each partition
# adjecencyMatrix = Handler.adjecencyMatrix(partitionArray, G, data)
#
#
#
# # write it
# with open('matrix.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     [writer.writerow(r) for r in adjecencyMatrix]


with open('matrix.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    adjecencyMatrix = [[float(e) for e in r] for r in reader]

# counting = 0
# for x in range(0,len(partitionArray)):
#     values = 0
#     for y in range(0,len(partitionArray)):
#         if(adjecencyMatrix[x][y]>0):
#             values+=1
#     if(len(partitionArray[x])<2):
#         values =2
#     if(values<2):
#         counting= x
#         break
# print(counting,"dasdasd")
tempPartitionArray = partitionArray.copy()

dict = dict()
for i in range(0, partitionSize):
    dict[i] = 0

condition = True
while(condition):
    # initialize again parameters
    partitionArray = tempPartitionArray.copy()
    for i in range(0, partitionSize):
        dict[i] = 0
    part = []
    indexer = []
    # create partition buckets
    for i in range(0, partitionSize):
        p = []
        for z in range(0, len(partitionArray)):
            id = Handler.partitionRandom(adjecencyMatrix, indexer,partitionSize)
            print(id,".....")
            indexer.append(id)
            if (Handler.edgeConectivity(adjecencyMatrix, part,id) == 0):
                part.append(p)
                dict[i] = adjecencyMatrix[id][id]
                part[i].append(id)
                break
            else:
                indexer.pop()
    tempPartitionSize = 0
    while (len(partitionArray) != len(indexer)):
        cont = 0
        for k in range(0,len(partitionArray)):
            if(k  in indexer):
                continue
            max, id = 0, -1
            # dictionary upper bound
            dicUpper = 0
            for v in range(0, partitionSize):
                dicUpper += dict[v]
            dicUpper = dicUpper/partitionSize
            for r in range(0, len(part)):
                edgeconectivity = Handler.interPartitionConectivity(adjecencyMatrix, part[r], k)
                if (edgeconectivity > max) & (dicUpper - dicUpper * 0.1 > dict[r]):
                    id = r
                    max = edgeconectivity
            if (id == -1):
                for r in range(0, len(part)):
                    edgeconectivity = Handler.interPartitionConectivity(adjecencyMatrix, part[r], k)
                    if (edgeconectivity > max):
                        id = r
                        max = edgeconectivity
            if (id != -1):

                part[id].append(k)
                indexer.append(k)
                dict[id] += max+adjecencyMatrix[k][k]
            cont += 1
        if (len(indexer) == tempPartitionSize):
            break
        tempPartitionSize = len(indexer)

    dicUpper = 0
    for v in range(0, partitionSize):
        print("inter-conectivity", v, "=", dict[v])
        dicUpper += dict[v]
    dicUpper = dicUpper / partitionSize
    conditionCount = 0
    for r in range(0, partitionSize):
        if (dicUpper - dicUpper * 0.02 <= dict[r] <= dicUpper + dicUpper * 0.02):
            conditionCount += 1
    print(len(indexer))
    if (conditionCount == partitionSize)&(len(indexer)==len(partitionArray)):
        break



counting = 0
for x in range(0,len(partitionArray)):
    values = 0
    for y in range(0,len(partitionArray)):
        counting+=adjecencyMatrix[x][y]
print("Edge conectivity",counting)

partition = []
for p in part:
    partTemp = []
    for par in p:
        for atemp in partitionArray[par]:
            partTemp.append(atemp)
    partition.append(partTemp)
np.savetxt('test_4.txt', partition, fmt='%r')

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

