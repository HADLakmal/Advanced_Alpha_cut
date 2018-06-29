
import numpy as np

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

old_market = (6.9048,79.8561)
#G = ox.graph_from_point(old_market, distance=1000)
G = ox.graph_from_place('Colombo, Colombo District, Western Province, Sri Lanka', network_type='drive')

# quick plot
ox.plot_graph(G , fig_height=10, fig_width=10)
#ox.plot_graph(ox.project_graph(G))
st = "";
for k in G.edges():
    if (k[0]!=k[1])&(1<len(G[k[0]])):
        try:
            weight = (100-int(G[k[0]][k[1]][0]["maxspeed"]))
        except KeyError:
            weight = 30
        except TypeError:
            weight = 30
        st += str(k[0]) + " " + str(k[1]) + " " + str(weight) + "\n"

with open("colombo.txt", "w") as text_file:
    text_file.write(st)