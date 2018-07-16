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

G = nx.read_graphml("../Osmnx_large_trips_graph_75000.graphml")
data = 'trips'

for k in partitionArray[1]:
    for r in partitionArray[1]:
        print(G.get_edge_data(str(k), str(r)).get('trips'))

