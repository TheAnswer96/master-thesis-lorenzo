import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import smopy
import time

"""
############################################################################################################
#                                                                                                          #
#                                              SEZIONE FUNZIONI                                            #
#                                                                                                          #
#                                             version:25-05-2020                                           #
############################################################################################################
"""
def dijkstra(graph, start):
    """
    Implementation of dijkstra using adjacency matrix.
    This returns an array containing the length of the shortest path from the start node to each other node.
    It is only guaranteed to return correct results if there are no negative edges in the graph. Positive cycles are fine.
    This has a runtime of O(|V|^2) (|V| = number of Nodes), for a faster implementation see @see ../fast/Dijkstra.java (using adjacency lists)

    :param graph: an adjacency-matrix-representation of the graph where (x,y) is the weight of the edge or 0 if there is no edge.
    :param start: the node to start from.
    :return: an array containing the shortest distances from the given start node to each other node
    """
    # This contains the distances from the start node to all other nodes
    distances = [float("inf") for _ in range(len(graph))]

    # This contains whether a node was already visited
    visited = [False for _ in range(len(graph))]

    # The distance from the start node to itself is of course 0
    distances[start] = 0

    # While there are nodes left to visit...
    while True:

        # ... find the node with the currently shortest distance from the start node...
        shortest_distance = float("inf")
        shortest_index = -1
        for i in range(len(graph)):
            # ... by going through all nodes that haven't been visited yet
            if distances[i] < shortest_distance and not visited[i]:
                shortest_distance = distances[i]
                shortest_index = i

        # print("Visiting node " + str(shortest_index) + " with current distance " + str(shortest_distance))

        if shortest_index == -1:
            # There was no node not yet visited --> We are done
            return distances

        # ...then, for all neighboring nodes that haven't been visited yet....
        for i in range(len(graph[shortest_index])):
            # ...if the path over this edge is shorter...
            if graph[shortest_index][i] != 0 and distances[i] > distances[shortest_index] + graph[shortest_index][i]:
                # ...Save this path as new shortest path.
                distances[i] = distances[shortest_index] + graph[shortest_index][i]
                # print("Updating distance of node " + str(i) + " to " + str(distances[i]))

        # Lastly, note that we are finished with this node.
        visited[shortest_index] = True
        #print("Visited nodes: " + str(visited))
        #print("Currently lowest distances: " + str(distances))
def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a
    2D diagram to finite regions.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a
    # given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                  vor.ridge_vertices):
        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))
        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an
            # infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(
                np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)
"""
	distance_coor_two_points
	-lat1: latitudine del punto 1
	-lon1: longitudine del punto 1
	-lat2: latitudine del punto 2
	-lon2: longitudine del punto 2
	* la funzione restituisce la distanza in KM(distanza euclidea) fra il punto 1 e il punto 2
"""
def distance_coor_two_points(lat1,lon1,lat2,lon2):
	from math import sin, cos, sqrt, atan2, radians

	# approximate radius of earth in km
	R = 6373.0

	lat1 = radians(lat1)
	lon1 = radians(lon1)
	lat2 = radians(lat2)
	lon2 = radians(lon2)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	distance = R * c

	#print("Result:", distance)
	return distance
"""
	trasform_editable_delauney
	-delauney: lista di liste delle latitudini, longitudini dei vertici di ogni triangolo[[lat0,lon0],...,[latn,lon]]
	-points: lista contenete l'accoppiamento latitudine, longitudine di ogni punto
	-mapp: mappa ottenuta da OpenStreetMap
	* la funzione restituisce la matrice di adiacenza del grafo dedotto dalla triangolazione
"""
def trasform_editable_delauney(delauney,points,mapp):
	res = []
	for tri in delauney.simplices:
		res.append(mapp.to_pixels(points[tri]))
	return res
"""
	city_to_coor
	-lat: lista delle latitudini dei punti di osservazione
	-lon: lista delle longitudini dei punti di osservazione
	-city: lista dei nomi delle città
	* la funzione restituisce il dizionario con:
  	- KEY: il nome della città
  	- VALUE: la tupla latitudine,longitudine della città
"""
def city_to_coor(lat,lon,city):
	dictionary = {}
	for index in range(len(lat)):
		couple = (lat[index],lon[index])
		dictionary[city[index]] = couple
		inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary
"""
	city_to_dict
	-lat: lista delle latitudini dei punti di osservazione
	-lon: lista delle longitudini dei punti di osservazione
	* la funzione restituisce il dizionario con:
  	- KEY: la tupla latitudine,longitudine della città
  	- VALUE: l'indice nella matrice di adiacenza
"""
def city_to_dict(lat,lon):
	dictionary = {}
	for index in range(len(lat)):
		couple = (lat[index],lon[index])
		dictionary[couple] = index
	inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary
"""
	delounay_to_adj_matrix
	-triangles: lista di liste delle latitudini, longitudini dei vertici di ogni triangolo[[lat0,lon0],...,[latn,lon]]
	-dict_latlon_index: dizionario che contiene la corrispondenza coordinate indice matrice
	* la funzione restituisce la matrice di adiacenza del grafo dedotto dalla triangolazione
"""
def delounay_to_adj_matrix(triangles,dict_latlon_index):
	#from itertools import combinations 
	from itertools import permutations 
	N = len(dict_latlon_index.items())
	M_adj = np.zeros((N,N))
	cont = 0
	for tri in triangles:
		perm = permutations(tri, 2) 
		for elem in list(perm):
			row = dict_latlon_index[tuple(elem[0])]
			col = dict_latlon_index[tuple(elem[1])]
			M_adj[row][col] = distance_coor_two_points(elem[0][0],elem[0][1],elem[1][0],elem[1][1])
	return M_adj
"""
	calculate_triangles_edges
	-M_adj: matrice di adiacenza del grafo dedotto dalla triangolazione
	* la funzione restituisce il numero di archi(orientati) presenti nella triangolazione
"""
def calculate_triangles_edges(M_adj):
	edges=0
	for i in range(0,len(M_adj)):
		for j in range(0,len(M_adj)):
			if(M_adj[i][j]!=0):			
				edges=edges+1
	return edges
"""
	PROPRIETA'1
	n: vertici
	k: vertici nel Convex Hull
	- N.archi = 3n - 3 - k
"""
def check_n_edges(points,M_adj):

	result = False
	print("#######################################################")
	hull = spatial.ConvexHull(points)
	N = len(points)
	K = len(hull.simplices)
	te_edges = (3*N-3-K)
	edges = int(calculate_triangles_edges(M_adj)/2)
	print("N.VERTICI: ",N)
	print("N.VERTICI INVILUPPO CONV.: ",K)
	print("N.ARCHI TEORICI: ",te_edges)
	print("N.ARCHI OTTENUTI: ",edges)
	if(edges==te_edges):
		result=True
	print("VERIFICA ARCHI: ",result)
	print("#######################################################")
	return result
"""
	PROPRIETA'2
	n: vertici
	k: vertici nel Convex Hull
	- N.triangoli = 2n - 2 - k
"""
def check_n_triangles(points):

	print("#######################################################")
	tri = spatial.Delaunay(points)
	hull = spatial.ConvexHull(points)
	result = False
	triangles =len(tri.simplices)
	N = len(points)
	K = len(hull.simplices)
	te_triangles=2*N-2-K
	print("N.VERTICI: ",N)
	print("N.VERTICI INVILUPPO CONV.: ",K)
	print("N.TRIANGOLI TEORICI: ",te_triangles)
	print("N.TRIANGOLI OTTENUTI: ",triangles)
	if(triangles==te_triangles):
		result=True
	print("VERIFICA TRIANGOLI: ",result)
	print("#######################################################")
	return result
"""
	print_connection
	-M_adj: matrice di adiacenza del grafo dedotto dalla triangolazione
	-inv_city_lat: dizionario inverso di city_lat
	-inv_lat_index: dizionario inverso di lat_index
	* la funzione restituisce a video tutti gli archi con il loro peso
"""
def print_connection(M_adj,inv_city_lat,inv_lat_index):
	for i in range(0,len(M_adj)):
		for j in range(0,len(M_adj)):
			if(M_adj[i][j]!=0):
				coor1 = inv_lat_index[i]
				city1 = inv_city_lat[coor1]
				coor2 = inv_lat_index[j]
				city2 = inv_city_lat[coor2]
				print("CITY: ",city1," IS CONNECT TO CITY: ",city2," WITH DISTACE: ",M_adj[i][j]," KM")
	return
"""
############################################################################################################
#                                                                                                          #
#                                         CORPO MAIN DEL PROGRAMMA                                         #
#                                                                                                          #
#                                            version:25-05-2020                                            #
############################################################################################################
"""
df = pd.read_csv("corsica.csv",sep=';', header=None)
lon = df[2]
lat = df[1]
city = df[0]
box = (lat.min()-1, lon.min()-1,lat.max()+1, lon.max()+1)
m = smopy.Map(box, z=9)
ax = m.show_mpl(figsize=(30, 15))
for i in range(len(lon)):
	x,y = m.to_pixels(lat[i],lon[i])
	ax.plot(x, y, 'or', ms=6, mew=2)
	if(len(city[i])>6):
		ax.text(x-10, y-10, city[i][:6].replace('_', ''), fontsize=6)
	else:
		ax.text(x-10, y-10, city[i].replace('_', ''), fontsize=6)
points=np.c_[lat, lon]
vor = spatial.Voronoi(points)
tri = spatial.Delaunay(points)
hull = spatial.ConvexHull(points)
city_lat, inv_city_lat = city_to_coor(lat,lon,city)
lat_index, inv_lat_index = city_to_dict(lat,lon)
regions, vertices = voronoi_finite_polygons_2d(vor)
matrix = delounay_to_adj_matrix(points[tri.simplices],lat_index)
property1 = check_n_triangles(points)
property2 = check_n_edges(points,matrix)
print("loading...")
time.sleep(4)
print("loading complete")
time.sleep(2)
if(property1 and property2):
	#print_connection(matrix,inv_city_lat,inv_lat_index)
	hulls = [m.to_pixels(points[si]) for si in hull.simplices]
	cells = [m.to_pixels(vertices[region]) for region in regions]
	triangulation = trasform_editable_delauney(tri,points,m)
	ax.add_collection(mpl.collections.PolyCollection(cells,edgecolors='black',linewidths=(1.3,),alpha=.35))
	ax.add_collection(mpl.collections.PolyCollection(triangulation,edgecolors='orange',linewidths=(1.3,), facecolor=None,alpha=.45))
	ax.add_collection(mpl.collections.PolyCollection(hulls,edgecolors='purple',linewidths=(1.3,), facecolor=None,alpha=.45))
	result = dijkstra(matrix,0)
	lst = []
	for index in range(33):
		print("SP from city: ", city[0]," to city: ",city[index]," is ",result[index],". The euclidean distance from these cities is: ",distance_coor_two_points(lat[0],lon[0],lat[index],lon[index]))
	df1.to_csv("tab1.csv")
	plt.savefig('mappa.png')
	plt.show()
else:
	print("ERROR: TRIANGULATION DOES NOT RESPECT SOME PROPERTIES")

				





