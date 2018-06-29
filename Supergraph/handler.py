import networkx as nx
import numpy as np


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
        degree = np.zeros((A.shape[0],A.shape[0]))
        #define ones matrix
        oneMatrix = np.ones((A.shape[0],A.shape[0]))
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
        denominator = np.matmul(maltiply,oneMatrix)
        value = np.subtract(numerator,denominator)
        return value-A
    def conectivityMatrix(partitionArray,G):
        matrix = np.zeros((len(partitionArray),len(partitionArray)))
        i = 0
        for r in partitionArray:
            for k in range(0,len(partitionArray)):
                if(k>i):
                    value = 0.0
                    count = 0.0
                    for c in r:
                        for d in partitionArray[k]:
                            #value+=float(G.get_edge_data(d,c)['weight'])
                            if(G.get_edge_data(d,c) is not None):
                                count+=1.0
                                value+=G.get_edge_data(d,c)['weight']
                    if(count!=0):
                        matrix[i][k] = value/count
                        matrix[k][i] = value/count

            i+=1;
        return matrix; 

