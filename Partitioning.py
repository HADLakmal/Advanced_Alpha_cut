import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

n1, n2 = 10, 10

# Define two random graphs
g1 = nx.gnm_random_graph(n1,20)
g2 = nx.gnm_random_graph(n2,20)
pos1 = nx.graphviz_layout(g1,prog='dot')
pos2 = nx.graphviz_layout(g2,prog='dot')

# Shift graph2
shift = 400
for key in pos2:
    pos2[key] = (pos2[key][0]+shift, pos2[key][1])

# Combine the graphs and remove all edges
g12 = nx.disjoint_union(g1,g2)
for i,j in g12.edges():
    g12.remove_edge(i,j)

# Add the conjoined edges
g12.add_edge(5, 7+n1)
g12.add_edge(2, 3+n1)
g12.add_edge(8, 7+n1)

# Set the new pos for g12
pos12 = pos1.copy()
for node in pos2:
    pos12[node+n2] = pos2[node]

# Show the results, make the conjoined edges darker

import pylab as plt
nx.draw_networkx(g1,pos=pos1,alpha=0.5)
nx.draw_networkx(g2,pos=pos2,alpha=0.5)
nx.draw_networkx_edges(g12,pos=pos12,width=5)
plt.axis('off')
plt.show()
