
import numpy as np
from handler import  Handler
import networkx as nx

G = nx.read_graphml("../../Osmnx_large_trips_graph_75000.graphml")
data = 'trips'

with open("alpha_1_6.txt") as f:
    datas = f.read()
# print(data.split(']')[0])
partitionArray = []
nodeCounter = 0
for k in range(0, 6):
    partition_data = datas.split('\n')[k].replace('{','').replace('}', '').replace(']', '').replace('[', '').split(', ')
    tempartition = []
    for i in partition_data:
        for z in i.split('+'):
            nodeCounter+=1
            tempartition.append(int(z.replace('\'','')))
    partitionArray.append(tempartition)
np.savetxt('test.txt', partitionArray, fmt='%r')

adjecencyMatrix , pickMatrix = Handler.adjecencyMatrix(partitionArray, G, data)

sum = 0
for x in range(len(pickMatrix)):
    sum += pickMatrix[x][x]
print("6")
print(pickMatrix)
print(sum,(9980283-sum)*100/9980283)