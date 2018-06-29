import networkx as nx
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

G=nx.Graph()

G.add_edge('a','b',weight=0.8)
G.add_edge('b','c',weight=0.9)
G.add_edge('a','c',weight=0.6)
"""
G.add_edge('a','d',weight=0.1)
G.add_edge('c','f',weight=0.2)
G.add_edge('f','d',weight=0.7)
G.add_edge('e','f',weight=0.8)
G.add_edge('e','d',weight=0.6)
"""
G.add_edge('a','d',weight=0)
G.add_edge('c','f',weight=0)
G.add_edge('f','d',weight=0.3)
G.add_edge('e','f',weight=0.4)
G.add_edge('e','d',weight=0.7)

"""
print("Eigenvalues", np.linalg.eigvals(A))

eigenvalues, eigenvectors = np.linalg.eig(A)
print("First tuple of eig", eigenvalues)
print("Second tuple of eig\n", eigenvectors)

for i in range(len(eigenvalues)):
   print("Left", np.dot(A, eigenvectors[:,i]))
   print(...)
"""

print(G.node())
        

"""
A = nx.laplacian_matrix(G, nodelist = ['a', 'b', 'f','d','e'])
eigenvalues, eigenvectors = np.linalg.eig(A.toarray())


print(eigenvalues)
print(eigenvectors)
"""
