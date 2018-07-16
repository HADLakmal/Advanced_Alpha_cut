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
        del maltiply
        del oneMatrix
        del degree
        del rowsum
        denominator = np.linalg.inv(deno)
        value = denominator*numerator
        # value = numerator*denominator
        return value-A
    def conectivityMatrix(partitionArray,G,data):
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
                            #value+=float(G.get_edge_data(d,c)['weight'])
                            if (G.get_edge_data(str(d), str(c)) is not None):
                                # value += abs(G.get_edge_data(d, c)[data])
                                if (len(G.get_edge_data(str(d), str(c))) > 1):
                                    count+=1
                                    value += abs(G.get_edge_data(str(d), str(c))[1][data])
                    if(count!=0):
                        edgecut+=value
                        matrix[i][k] = value/count
                        matrix[k][i] = value/count

            i+=1;
        return matrix,edgecut
    def edgeConectivity(matrix,part,index):
        value = 0
        for q in part:
            for r in q:
                value = matrix[r][index]

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

    def interPartitionConectivity(matrix,p1,index):

        value = 0
        for r in p1:
            value += matrix[r][index]
        return value

    def partitionRandom(adjecencyMatrix,indexer,partitionSize):
        value =0
        while(True):
            max = 0
            for z in range(0,int(len(adjecencyMatrix)/partitionSize)):
                value = randint(0,len(adjecencyMatrix)-1)
                if (value not in indexer)&(adjecencyMatrix[max][max]<adjecencyMatrix[value][value]):
                    max = value
            if max not in indexer:
                value = max
                break

        return value

    def adjecencyMatrix(partitionArray,G,data):
        matrix = np.zeros((len(partitionArray),len(partitionArray)),dtype=np.float)
        i = 0
        edgecut = 0
        for r in partitionArray:
            for k in range(0,len(partitionArray)):
                if(k>i):
                    value = 0.0
                    for c in r:
                        for d in partitionArray[k]:
                            #value+=float(G.get_edge_data(d,c)['weight'])
                            if (G.get_edge_data(str(d), str(c)) is not None):
                                # value += abs(G.get_edge_data(d, c)[data])
                                value+=1
                                if (len(G.get_edge_data(str(d), str(c))) > 1):
                                    value += abs(G.get_edge_data(str(d), str(c))[1][data])
                        matrix[i][k] = value
                        matrix[k][i] = value
                elif(i==k):
                    value = 0
                    for x in range(0,len(r)):
                        for y in range(x,len(r)):
                            if (G.get_edge_data(str(r[x]), str(r[y])) is not None):
                                value+=1
                                if (len(G.get_edge_data(str(r[x]), str(r[y]))) > 1):
                                    value += abs(G.get_edge_data(str(r[x]), str(r[y]))[1][data])
                    matrix[i][k] = value


            i+=1;
        return matrix

    def partCon(G,p1,p2,data):

        value = 0
        for r in p1:
            for d in p2:
                # value+=float(G.get_edge_data(d,c)['weight'])

                if (G.get_edge_data(str(r), str(d)) is not None):
                    # value += abs(G.get_edge_data(d, c)[data])
                    if (len(G.get_edge_data(str(r), str(d))) > 1):
                        value += abs(G.get_edge_data(str(r), str(d))[1][data])
        return value
