import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from handler import Handler
import collections
import queue


G = nx.read_edgelist('colombo.txt',nodetype=int, data=(('weight',float),))

stri = ""
for i in G.nodes():

    stri+=str(i)+'\n'

text_file = open("nodelist.txt", "w")
text_file.write(stri)



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
