import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from handler import Handler
import collections
import queue



#G = nx.read_graphml("CombinedMinimizedGraph.graphml")


G=nx.Graph()

G.add_edge('a','b',weight=0.8)
G.add_edge('b','c',weight=0.9)
G.add_edge('a','c',weight=0.6)

G.add_edge('a','d',weight=0.1)
G.add_edge('c','f',weight=0.2)
G.add_edge('f','d',weight=0.7)
G.add_edge('e','f',weight=0.8)
G.add_edge('e','d',weight=0.6)

G.add_edge('f','g',weight=1)
G.add_edge('g','h',weight=0.1)
G.add_edge('e','h',weight=1)



#print(nx.adjacency_matrix(G,nodelist = ['c','e']).todense())
A = nx.adjacency_matrix(G)

#Alpah cut
M = Handler.alphaCut(A,1)

#eigen calculation

eigenvalues, eigenvectors = np.linalg.eig(M)

#define K
partitionSize=3
tempEigenValues = np.absolute(eigenvalues)
idx = tempEigenValues.argsort()[:partitionSize][::]
eigenValues = tempEigenValues[idx]
eigenVectors = eigenvectors[:,idx]


z = eigenVectors
#normalize the matrix
for i in range(0,eigenVectors.shape[0]):
    total = 0
    for j in range(0,eigenVectors.shape[1]):
        total += eigenVectors.item((i,j))**2
    z[i]=+z[i]/(total**(1/2))

#find k means paritions
kmeans = KMeans(n_clusters=partitionSize, random_state=0).fit(z)


lables = kmeans.labels_

array = Handler.indexArray(G.nodes(),partitionSize,lables)

#New partition array
partitionArray = []
# get each laplacian matrix
for k in array:
    #print(nx.adjacency_matrix(G,nodelist = k).todense())
    A = nx.laplacian_matrix(G, nodelist = k)
    tempEigenvalues, tempEigenvectors = np.linalg.eig(A.toarray())
    sort = tempEigenvalues.argsort()
    #sort = tempEigenvalues
    if(sort.size>2):
        if 0 in tempEigenvectors:
            counter=collections.Counter(sort)
            print(counter)
            p = list(counter.values())[-1]
            kmeans = KMeans(n_clusters=p+1, random_state=0).fit(tempEigenvectors[:,[list(tempEigenvalues).index(0)]])
            lables = kmeans.labels_
            arrays = Handler.indexArray(k,p+1,lables)
            for i in arrays:
                partitionArray.append(i)
        else:
            partitionArray.append(k)
    else:
        partitionArray.append(k)

print(partitionArray)


matrix = Handler.conectivityMatrix(partitionArray,G)
q = queue.Queue()
partitionQueue = queue.Queue();
partitionQueue.put(partitionArray);
q.put(matrix)
alpha = Handler.alphaCut(matrix,0)
partitionCount = 1
part =[]
part.append(partitionArray)
while(partitionCount!=partitionSize):
    if(q.empty() is False):
        matrix = q.get()
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
                    total += eigenVectors.item((i,j))**2
                z[i]=+z[i]/(total**(1/2))

            #find k means paritions
            kmeans = KMeans(n_clusters=2, random_state=0).fit(z)
            w = 0
            p1,p2 = [],[]
            partition = partitionQueue.get()
            for p in kmeans.labels_:
                if(p==0):
                    p1.append(partition[w])
                else:
                    p2.append(partition[w])
                w+=1
            partitionQueue.put(p1)
            partitionQueue.put(p2)
            q.put(Handler.conectivityMatrix(p1,G))
            q.put(Handler.conectivityMatrix(p2,G))
            part.pop(0)
            part.append(p1)
            part.append(p2)
        
    partitionCount+=1



print(part)


"""

pos=nx.spring_layout(G)
# edges
nx.draw_networkx_edges(G,pos,
                        width=6)
# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

nx.draw_networkx_nodes(G,pos,node_size=700)
plt.show()


#show partitioning nodes

for i in partitionArray:

    
    nx.draw_networkx_nodes(G,pos,nodelist=i,node_size=700)
    nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
    nx.draw_networkx_edges(G,pos,nodelist = i,
                        width=6)
    
    
    plt.show()

"""
'''
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.7]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.7]

pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)

# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=6)
nx.draw_networkx_edges(G,pos,edgelist=esmall,
                    width=6,alpha=0.5,edge_color='b',style='dashed')

# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display
'''
