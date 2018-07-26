from handler import Handler
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


# Test one


# with open("test_2.txt") as f:
#     datas = f.read()
# # print(data.split(']')[0])
# partitionArray = []
# for k in range(0, 601):
#     partition_data = datas.split('\n')[k].replace('{','').replace('}', '').split(', ')
#     tempartition = []
#     for i in partition_data:
#         for z in i.split('+'):
#             tempartition.append(int(z.replace('\'','')))
#     partitionArray.append(tempartition)
#
# G = nx.read_graphml("../Osmnx_large_trips_graph_75000.graphml")
# data = 'trips'
#
# for k in range(0,len(partitionArray[1])):
#     part = partitionArray[1]
#     neighbours = G.neighbors(str(part[k]))
#     print(list(neighbours),G[str(part[k])])
#     for r in range(0,len(partitionArray[1])):
#         if(r>k):
#
#             if(G.get_edge_data(str(part[k]), str(part[r])) is not None):
#
#                 print(part[k],part[r],G.get_edge_data(str(part[k]), str(part[r])))


# Test two

with open('matrix_alpha.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    adjecencyMatrix = [[float(e) for e in r] for r in reader]

adjecencyMatrix = np.array(adjecencyMatrix)
val = 0
print(adjecencyMatrix.shape[0])
for x in range(0,adjecencyMatrix.shape[0]):
    for y in range(0,adjecencyMatrix.shape[0]):
        if(adjecencyMatrix[x][y]>0):
            val+=1
            break
print(val)
