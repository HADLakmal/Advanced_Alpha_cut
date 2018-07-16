import networkx as nx
import numpy as np
from random import randint
from scipy import linalg
from scipy import sparse

class Handler:

    #get exact node list from graph partition
    def indexArray(nodeList,k,lables):
        array = []
        for r in range(0,k):
            array.append([])
            count = 0
            for i in lables:
                count+=1
                graph_count = 0
                for a in nodeList:
                    graph_count+=1
                    if(r==i and count==graph_count):
                        array[r].append(a)
                        break
        return array
    def alphaCut(A,adjecency):
        #define zero matrix
        degree = np.zeros((A.shape[0],A.shape[0]),dtype=np.float)
        #define ones matrix
        oneMatrix = np.ones((A.shape[0],1),dtype=np.float)
        #get sum of row
        rowsum = A.sum(axis=0)
        #create degree matrix
        for j in range(0, A.shape[0]):
            if(adjecency==1):
                degree[j,j] = rowsum[0,j]
            else:
                degree[j,j] = rowsum[j]
        #Get alpha cut matrix
        maltiply = np.matmul(oneMatrix.transpose(), degree)

        numerator = np.matmul(maltiply.transpose(),maltiply)
        deno = np.matmul(maltiply,oneMatrix)
        print(deno)
        denominator = np.linalg.inv(deno)
        value = denominator*numerator
        # value = numerator*denominator
        return value-A
    def conectivityMatrix(partitionArray,G):
        matrix = np.zeros((len(partitionArray),len(partitionArray)),dtype=np.float)
        i = 0
        edgecut = 0
        for r in partitionArray:
            for k in range(0,len(partitionArray)):
                if(k>i):
                    value = 0.0
                    count = 0.0
                    for c in r:
                        for d in partitionArray[k]:

                            if(G.get_edge_data(d,c) is not None):
                                value+=float(G.get_edge_data(d,c)['weight'])
                                count+=1
                                # if (len(G.get_edge_data(d, c)) > 1):
                                #     value += abs(G.get_edge_data(d, c)[1][data])
                    if(count!=0):
                        edgecut+=value
                        matrix[i][k] = value/count
                        matrix[k][i] = value/count

            i+=1;
        return matrix,edgecut
    def edgeConectivity(G,p1,p2):
        value = 0
        for q in p1:
            for r in q:
                for c in r:
                    for d in p2:
                        # value+=float(G.get_edge_data(d,c)['weight'])
                        if (G.get_edge_data(d, c) is not None):
                            value += float(G.get_edge_data(d, c)['weight'])
        return value

    def degreeMatrix(G,p):
        matrix = np.zeros((len(p), len(p)), dtype=np.float)
        i=0
        for r in p:
            degreeCount = 0
            for c in r:
                degreeCount+=G.degree(c)
            matrix[i][i] = degreeCount
            i+=1
        return matrix

    def partitionSize(partitionArray,indexer):
        max, id = 100000, 0
        for i in range(0, len(partitionArray)):
            if (len(partitionArray[i]) < max)&(i not in indexer):
                max = len(partitionArray[i])
                id = i
        return id

    def interPartitionConectivity(G,p1,p2):

        value = 0
        for r in p1:
            for c in r:
                for d in p2:
                    # value+=float(G.get_edge_data(d,c)['weight'])
                    if (G.get_edge_data(d, c) is not None):
                        value += float(G.get_edge_data(d, c)['weight'])
        return value

    def partitionRandom(partitionArray,indexer):
        value =0
        while(True):
            value = randint(0,len(partitionArray)-1)
            if value not in indexer:
                break

        return value
