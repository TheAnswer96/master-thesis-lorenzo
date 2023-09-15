import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
from math import sin, cos, sqrt, atan2, radians


df = pd.read_csv("Provincia.csv",sep=';', header=None)

city = df[2]
print(city)
exit()
R = 6371

x = []
y = []

for i in range(len(lon)):
	
	 x.append((R * cos(lat[i]) * cos(lon[i])))
	 y.append((R * cos(lat[i]) * sin(lon[i])))


points=np.c_[x, y]
vor = spatial.Voronoi(points)
fig = spatial.voronoi_plot_2d(vor, show_vertices=False, line_colors='green',line_width=2, line_alpha=0.6, point_size=5)
tri = spatial.Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
for i in range(len(lon)):
	plt.text(x[i], y[i], city[i].replace('_', ''), fontsize=6)
plt.show()