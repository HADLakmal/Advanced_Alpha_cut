import networkx as nx


'''
with open("Colombo_Nodes.csv") as f:
    lis=[line.split() for line in f]       
    for i,x in enumerate(lis):              
        print(i,x)
'''

H = nx.read_graphml("CombinedMinimizedGraph.graphml")
array = list(H.nodes())



G = H.subgraph(array[:1000])


print(H.edges())
