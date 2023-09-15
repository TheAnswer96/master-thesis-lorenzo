import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import smopy
import time
import matplotlib.lines as mlines
import scale_final_graph as sfg

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

def city_to_coor(lat,lon,city):
	dictionary = {}
	for index in range(len(lat)):
		couple = (lat[index],lon[index])
		dictionary[city[index]] = couple
		inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary

def city_to_dict(lat,lon):
	dictionary = {}
	for index in range(len(lat)):
		couple = (lat[index],lon[index])
		dictionary[couple] = index
	inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary

def coor_to_index_dict(points,vertices,regions):
	list_vert = []
	list_city = []
	list_point = []
	seen_vert = set()
	dictionary = {}
	for region in regions:
		for vert in vertices[region]:
			if((vert[0],vert[1]) not in seen_vert):
				list_vert.append((vert[0],vert[1]))
				seen_vert.add((vert[0],vert[1]))
	for point in points:
		list_city.append((point[0],point[1]))
	list_point = list_city + list_vert
	for index in range(len(list_point)):
		key = list_point[index]
		dictionary[key] = index
	inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary

def to_adj_matrix(voronoi,regions,vertices,dict_latlon_index, dict_latlon_index_inv):

	N = len(dict_latlon_index.items())
	M_adj_dist = np.zeros((N,N))
	M_adj_angle = np.zeros((N,N))
	row = 0
	#calcolo pesi degli archi fra vertici di voronoi e punti di interesse
	vertices_per_cells = [vertices[region] for region in regions] 
	for poly in vertices_per_cells:
		for vert in poly:
			row_lat = dict_latlon_index_inv[row][0]
			row_lon = dict_latlon_index_inv[row][1]
			coor_vert = (vert[0],vert[1])
			col = dict_latlon_index[coor_vert]
			M_adj_dist[row][col] = distance_coor_two_points(row_lat,row_lon,vert[0],vert[1])
			M_adj_dist[col][row] = M_adj_dist[row][col]
			M_adj_angle[row][col] = sfg.angleFromCoordinate(row_lat,row_lon,vert[0],vert[1])
			M_adj_angle[col][row] = sfg.angleFromCoordinate(vert[0],vert[1],row_lat,row_lon)
		row = row + 1
	#calcolo pesi degli archi di voronoi
	for vpair in vor.ridge_vertices:
		if vpair[0] >= 0 and vpair[1] >= 0:
			v0 = vor.vertices[vpair[0]]
			v1 = vor.vertices[vpair[1]]
			row1 = dict_latlon_index[(v0[0],v0[1])]
			col1 = dict_latlon_index[(v1[0],v1[1])]
			M_adj_dist[row1][col1] = distance_coor_two_points(v0[0],v0[1],v1[0],v1[1])
			M_adj_dist[col1][row1] = M_adj_dist[row1][col1]
			M_adj_angle[row1][col1] = sfg.angleFromCoordinate(v0[0],v0[1],v1[0],v1[1])
			M_adj_angle[col1][row1] = sfg.angleFromCoordinate(v1[0],v1[1],v0[0],v0[1])
	return M_adj_dist,sfg.aprox_angle(M_adj_angle)
"""
############################################################################################################
#                                                                                                          #
#                                         CORPO MAIN DEL PROGRAMMA                                         #
#                                                                                                          #
#                                            version:31-07-2020                                            #
############################################################################################################
"""
df = pd.read_csv("city_coo_data.csv",sep=';', header=None)
lon = df[2]
lat = df[1]
city = df[0]
hour_wind_data = sfg.wind_data_load(city,2)
city_lat, inv_city_lat = city_to_coor(lat,lon,city)
lat_index, inv_lat_index = city_to_dict(lat,lon)
points=np.c_[lat, lon]
vor = spatial.Voronoi(points)
box = (lat.min()-1, lon.min()-1,lat.max()+1, lon.max()+1)
m = smopy.Map(box, z=9)
ax = m.show_mpl(figsize=(300, 150))
for i in range(len(lon)):
	x,y = m.to_pixels(lat[i],lon[i])
	ax.plot(x, y, 'or', ms=5, mew=2)
for i in range(len(lon)):
	for j in range(i,len(lon)):
		if i != j:
			x,y = m.to_pixels(lat[i],lon[i])
			x1,y1 = m.to_pixels(lat[j],lon[j])
			ax.plot([x,x1], [y,y1], 'orange', linewidth=1)
			ax.text((x+x1)/2 -1 , (y+y1)/2-1, str(round(sfg.distance_coor_two_points(lat[i],lon[i],lat[j],lon[j]),0))+" Km" , fontsize=5.5)
plt.show()
exit()
	#if(len(city[i])>6):
	#	ax.text(x-10, y-10, str(city[i]), fontsize=6)
	#else:
	#	ax.text(x-10, y-10, str(city[i]), fontsize=6)
points=np.c_[lat, lon]
"""
	Voronoi:
	-region: definisce in base al seed a cui fa riferimento l'insieme dei vertici che definisce il poligono

"""
vor = spatial.Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)



lat_index_tot, inv_lat_index_tot = coor_to_index_dict(points,vertices,regions)
M_adj_dist,M_adj_angle = to_adj_matrix(vor,regions,vertices,lat_index_tot,inv_lat_index_tot)
N = len(M_adj_angle)
M_adj_angle_gw = np.zeros((N,N))
M_adj_wind = np.zeros((N,N))

cells = []
cell =[vertices[region] for region in regions]
cells = [m.to_pixels(vertices[region]) for region in regions]

"""for region in vor.regions:
	for i in region:
		cell=[]
		if i >= 0:
			cell.append(m.to_pixels(vor.vertices[i][0],vor.vertices[i][1]))
	cells.append(cell)
"""

#print(cells[0])
#del cells[0]
for array in cell:
	print("\n\n  ")
	for i in array:
		print(lat_index_tot[(i[0],i[1])])
conta = 0
iteratore = 0
k=0
j=0
for array in cells:
	x, y = m.to_pixels(lat[iteratore],lon[iteratore])
	winfo = hour_wind_data[city[iteratore]]
	k = np.where(cells == array)
	conta = 0
	for i in array:
		ax.plot(i[0], i[1], 'bo', ms=3, mew=2)
		x_values = [i[0],x]
		y_values=[i[1],y]
		edge = mlines.Line2D(x_values, y_values)
		ax.add_line(edge)
		print(lat[iteratore],lon[iteratore],cell[iteratore][conta][0],cell[iteratore][conta][1])
		
		ax.text((x_values[0]+x_values[1])/2 -1 , (y_values[0]+y_values[1])/2-1, str(round(sfg.distance_coor_two_points(lat[iteratore],lon[iteratore],cell[iteratore][conta][0],cell[iteratore][conta][1]),0))+" Km" , fontsize=6)
		conta = conta + 1
		

	iteratore = iteratore + 1
print(conta)
"""
    questa parte di codice disegna uno ad uno i lati di voronoi escludendo quelli all'infinito
"""
indice = 0
vertici_visti = set()
for vpair in vor.ridge_vertices:
	if vpair[0] >= 0 and vpair[1] >= 0:
		v0 = vor.vertices[vpair[0]]
		v1 = vor.vertices[vpair[1]]
		x0,y0 = m.to_pixels(v0[0],v0[1])
		x1,y1 = m.to_pixels(v1[0],v1[1])
        # Draw a line from v0 to v1.
		if not((x0,y0) in vertici_visti):
			vertici_visti.add((x0,y0))
			#ax.text(x0, y0, str(lat_index_tot[(v0[0],v0[1])]), fontsize=6)
		if not((x1,y1) in vertici_visti):
			vertici_visti.add((x1,y1))
			#ax.text(x1, y1, str(lat_index_tot[(v1[0],v1[1])]), fontsize=6)
		print("EDGES: ("+str(v0[0])+","+str(v0[1])+"),("+str(v1[0])+","+str(v1[1])+")"+str(lat_index_tot[(v0[0],v0[1])])+","+str(lat_index_tot[(v1[0],v1[1])]))
		ax.plot([x0,x1], [y0,y1], 'green', linewidth=2)
		ax.text((x0+x1)/2 -1 , (y0+y1)/2-1, str(round(sfg.distance_coor_two_points(v0[0],v0[1],v1[0],v1[1]),0))+" Km" , fontsize=6)
DFcity = pd.read_csv("archi.csv",sep=';')
DFcity.columns=['index','lat','lon','num']
edges_lat =DFcity['lat'].tolist()
edges_lon =DFcity['lon'].tolist()
midpoint_name = DFcity['num'].tolist()
DFvor = pd.read_csv("vor.csv",sep=';')
point = DFvor['coor'].tolist()
name = DFvor['name'].tolist()
"""for index in range(len(edges_lat)):
		#parsing delle coordinate delle colonnine collegate da archi di delauney
		x0,x1 = edges_lat[index].replace("[","").replace("]","").split(", ")
		x0 = float(x0)
		x1 = float(x1)
		y0,y1 = edges_lon[index].replace("[","").replace("]","").split(", ")
		y0 = float(y0)
		y1 = float(y1)
		dist = round(sfg.distance_coor_two_points(x0,x1,y0,y1),0)
		x0,y0 = m.to_pixels(float(x0),float(y0))
		x1,y1 = m.to_pixels(float(x1),float(y1))
		ax.text((x0+x1)/2 -1 , (y0+y1)/2-1, str(dist)+" Km", fontsize=6)
		ax.plot([x0,x1], [y0,y1], 'orange', linewidth=1.5)
"""

#print(result)

#for index in range(11):
#	to_city = inv_lat_index_tot[index]
#	str_city = inv_city_lat[to_city]
#	if str_city != None:
#		print("SP from city: ", city[0]," to city: ",str_city," is ",result[index],". The euclidean distance from these cities is: ",distance_coor_two_points(lat[0],lon[0],lat[index],lon[index]))
plt.show()
