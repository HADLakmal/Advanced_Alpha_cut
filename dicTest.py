
import numpy as np

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import operator

stats = {'a':1000, 'b':3000, 'c': 100}
print(min(stats.items(), key=operator.itemgetter(1)))