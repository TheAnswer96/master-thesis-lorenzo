import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import smopy
import time
import os
import matplotlib.lines as mlines
import scale_final_graph as sfg
from networkx import nx
import random
from collections import Counter

BUDGET_ENERGY = 2500 #totale carica batteria Kj
PAYLOAD = 2 #peso del carico Kg
DRONE_SPEED = 10#velocità del drone media m/s
SOURCE = 0
INDICE_ORARIO = 2


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

def vor_to_name(vert1,vert2,wind):
	dict_wind = {}
	for i in range(len(vert1)):
		dict_wind[(vert1[i],vert2[i])] = wind[i]
	return dict_wind

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

def update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,index,prefixs,payload):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	hour_wind_data = sfg.wind_data_load(city,index)
	iteratore = 0
	for array in cells:
		winfo = hour_wind_data[city[iteratore]]
		for i in array:
			index_vor = lat_index_tot[(i[0],i[1])]
			M_adj_wind[iteratore][index_vor]= winfo[3]
			M_adj_wind[index_vor][iteratore] = winfo[3]
			M_adj_angle_gw[iteratore][index_vor] = winfo[2]
			M_adj_angle_gw[index_vor][iteratore] = winfo[2]
			app = M_adj_angle[iteratore][index_vor]-winfo[2]
			if app < 0:
				M_adj_angle[iteratore][index_vor] = app+360
			else:
				M_adj_angle[iteratore][index_vor] = app 
			app = M_adj_angle[index_vor][iteratore]-winfo[2]
			if app < 0:
				M_adj_angle[index_vor][iteratore] = app+360
			else:
				M_adj_angle[index_vor][iteratore] = app 
		iteratore = iteratore + 1
	for i in range(len(point1)):
		winfo = hour_wind_data[wind_name[i]]
		M_adj_wind[point1[i]][point2[i]] = winfo[3]
		M_adj_angle_gw[point1[i]][point2[i]] = winfo[2]
		app = M_adj_angle[point1[i]][point2[i]]-winfo[2]
		if app < 0:
			M_adj_angle[point1[i]][point2[i]] = app+360
		else:
			M_adj_angle[point1[i]][point2[i]] = app 
	M_adj_angle = sfg.aprox_angle(M_adj_angle)
	M_adj_wind = sfg.approx_wind(M_adj_wind,M_adj_dist)
	for i in range(N):
		for j in range(N):
			if M_adj_dist[i][j] != 0:
				M_adj_energy[i][j] = (M_adj_dist[i][j]*100) * prefixs[(0,10,M_adj_wind[i][j],M_adj_angle[i][j])]
	return M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy

def min_max_energy_matrix(M_dist,prefixs,speed,payload):
	N = len(M_dist)
	payload_weights = [0, 2]
	drone_speeds = [10]
	global_wind_speeds = [0, 5, 10, 15] #sto cambiando -- era [0, 5, 10, 15]
	relative_wind_directions = [0, 45, 135, 180]
	min_speed_direction_p = (10000,0,0)  
	for speed_i in global_wind_speeds:
		for dire in relative_wind_directions:
			if min_speed_direction_p[0] > round(sfg.get_energy(1,payload_weights[1],drone_speeds[0],speed_i,dire),2):
				min_speed_direction_p = (round(sfg.get_energy(1,payload_weights[1],drone_speeds[0],speed_i,dire),2),speed_i,dire)
	min_speed_direction = (10000,0,0)  
	for speed_i in global_wind_speeds:
		for dire in relative_wind_directions:
			if min_speed_direction[0] > round(sfg.get_energy(1,payload_weights[0],drone_speeds[0],speed_i,dire),2):
				min_speed_direction = (round(sfg.get_energy(1,payload_weights[0],drone_speeds[0],speed_i,dire),2),speed_i,dire)
	max_speed_direction_p = (0,0,0)  
	for speed_i in global_wind_speeds:
		for dire in relative_wind_directions:
			if max_speed_direction_p[0] < round(sfg.get_energy(1,payload_weights[1],drone_speeds[0],speed_i,dire),2):
				max_speed_direction_p = (round(sfg.get_energy(1,payload_weights[1],drone_speeds[0],speed_i,dire),2),speed_i,dire)
	max_speed_direction = (0,0,0)  
	for speed_i in global_wind_speeds:
		for dire in relative_wind_directions:
			if max_speed_direction[0] < round(sfg.get_energy(1,payload_weights[0],drone_speeds[0],speed_i,dire),2):
				max_speed_direction = (round(sfg.get_energy(1,payload_weights[0],drone_speeds[0],speed_i,dire),2),speed_i,dire)									
	lower_bound_matrix_p = np.zeros((N,N)) 
	lower_bound_matrix = np.zeros((N,N)) 
	upper_bound_matrix_p = np.zeros((N,N)) 
	upper_bound_matrix = np.zeros((N,N)) 
	for i in range(N):
		for j in range(N):
			if M_dist[i][j] != 0:
				lower_bound_matrix[i][j] = (M_dist[i][j]*100) * prefixs[(0,speed,min_speed_direction[1],min_speed_direction[2])]
				lower_bound_matrix_p[i][j] = (M_dist[i][j]*100) * prefixs[(payload,speed,min_speed_direction_p[1],min_speed_direction_p[2])]
				upper_bound_matrix[i][j] = (M_dist[i][j]*100) * prefixs[(0,speed,max_speed_direction[1],max_speed_direction[2])]
				upper_bound_matrix_p[i][j] = (M_dist[i][j]*100) * prefixs[(payload,speed,max_speed_direction_p[1],max_speed_direction_p[2])]
	return nx.Graph(lower_bound_matrix_p), nx.Graph(lower_bound_matrix), nx.Graph(upper_bound_matrix_p), nx.Graph(upper_bound_matrix)

def Preprocessing_PP(s,lower_M_p,lower_M,upper_M_p,upper_M):
	cicly_consume_min = {}
	cicly_consume_max = {}
	colors = []
	loc = "result/vPreprocessing_PP.txt"
	if(os.path.isfile(loc)):
		os.remove("result/vPreprocessing_PP.txt")
	file = open("result/vPreprocessing_PP.txt","a+")
	N = len(lower_M)
	for i in range(0,12):
		cicly_consume_min[i] = nx.dijkstra_path_length(lower_M_p, source=s, target=i, weight='weight') + nx.dijkstra_path_length(lower_M, source=i, target=s, weight='weight')
		cicly_consume_max[i] = nx.dijkstra_path_length(upper_M_p, source=s, target=i, weight='weight') + nx.dijkstra_path_length(upper_M, source=i, target=s, weight='weight')
		path_l = nx.dijkstra_path(upper_M_p, source=s, target=i, weight='weight') + nx.dijkstra_path(upper_M, source=i, target=s, weight='weight')
		path_s =  nx.dijkstra_path(lower_M_p, source=s, target=i, weight='weight') + nx.dijkstra_path(lower_M, source=i, target=s, weight='weight')
		#print(cicly_consume_min[i], cicly_consume_max[i])

		if  cicly_consume_min[i] > BUDGET_ENERGY:
			colors.append((i,"BLACK"))
			file.write("PATH: "+str(path_s)+" CONSUMPTION MAX: "+str(round(cicly_consume_max[i],2))+" Kj  MIN: "+str(round(cicly_consume_min[i],2))+" Kj\n")
			file.write("VERTEX: "+str(i)+" IS BLACK\n")
		if cicly_consume_max[i] > BUDGET_ENERGY and cicly_consume_min[i] <= BUDGET_ENERGY:
			colors.append((i,"GRAY"))
			file.write("PATH: "+str(path_s)+" CONSUMPTION MAX: "+str(round(cicly_consume_max[i],2))+" Kj  MIN: "+str(round(cicly_consume_min[i],2))+" Kj\n")
			file.write("VERTEX: "+str(i)+" IS GRAY\n")
		if cicly_consume_max[i] < BUDGET_ENERGY:
			colors.append((i,"GREEN"))
			file.write("PATH: "+str(path_s)+" CONSUMPTION MAX: "+str(round(cicly_consume_max[i],2))+" Kj  MIN: "+str(round(cicly_consume_min[i],2))+" Kj\n")
			file.write("VERTEX: "+str(i)+" IS GREEN\n")
	file.close()
	return colors

def delete_edges(G,edges_delete):
	for e in edges_delete:
		u,v = e
		try:
			G.remove_edge(u, v)
		except Exception as e:
			continue
	return G

def run_test_alg_off_sp(M_adj_dist,cells,point1,point2,wind_name,city,destination,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	confronto_energetico = open("result/vconfronto_energetico_off_sp"+str(indice_ora)+".txt","a+")
	consumption_go = 0
	consumption_return= 0
	distance = 0
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,payload)
	G = nx.Graph(M_adj_energy)
	G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
	time = 0
	STATUS = ""
	sp_cost = nx.dijkstra_path_length(G, source=SOURCE, target=destination, weight='weight') 
	sp_path = nx.dijkstra_path(G, source=SOURCE, target=destination, weight='weight') 
	cycle_cost = nx.dijkstra_path_length(G1, source=destination, target=SOURCE, weight='weight') + sp_cost
	confronto_energetico.write("Energy consumption projection: "+str(cycle_cost))
	sp_return = nx.dijkstra_path(G1, source=destination, target=SOURCE, weight='weight') 
	confronto_energetico.write("node->node_s\n")
	node = sp_path.pop(0)
	confronto_energetico.write(str(node))
	node_r = sp_return.pop(0)
	if cycle_cost <= BUDGET_ENERGY:
		while node != destination:
			control_time = time // 15
			if control_time == 0:
				node_next = sp_path.pop(0)
				confronto_energetico.write("->"+str(node_next))
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				#print(str(M_adj_wind[node][node_next]))
				node = node_next
			else:
				indice_orario = (indice_orario + int(control_time)) % 240
				if indice_orario <= 1:
					indice_orario = indice_orario + 2
				M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,payload)
				G = nx.Graph(M_adj_energy)
				node_next = sp_path.pop(0)
				confronto_energetico.write("->"+str(node_next))
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				node = node_next
			#print(M_adj_angle[node][node_next],M_adj_wind_gw[node][node_next],M_adj_angle[node_next][node],M_adj_wind_gw[node_next][node])
		#print(consumption_go,time,distance)
		G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
		confronto_energetico.write("->"+str(node_r))
		if consumption_go <= BUDGET_ENERGY:
			while node_r != SOURCE:
				control_time = time // 15
				if control_time == 0:
					node_r_next = sp_return.pop(0)
					confronto_energetico.write("->"+str(node_r_next))
					consumption_return = consumption_return + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					#print(M_adj_wind[node_r][node_r_next])
					node_r = node_r_next
				else:
					indice_orario = (indice_orario + int(control_time)) % 240
					if indice_orario <= 1:
						indice_orario = indice_orario + 2
					M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,0)
					G1 = nx.Graph(M_adj_energy)
					node_r_next = sp_return.pop(0)
					confronto_energetico.write("->"+str(node_r_next))
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					node_r = node_r_next
				#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
			if (consumption_go+consumption_return) <= BUDGET_ENERGY:
				STATUS="SUCCESS"
				confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
				status_v[STATUS] = status_v[STATUS] +1
			else:
				STATUS="DELIVERED"
				status_v[STATUS] = status_v[STATUS] +1
				confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
		else:
			STATUS="FAIL"
			status_v[STATUS] = status_v[STATUS] +1
			confronto_energetico.write("\nFAIL")
	else:
		STATUS = "CANCELED"
		status_v[STATUS] = status_v[STATUS] +1
		confronto_energetico.write("\nCANCELED")
	confronto_energetico.write("\n WIND CHANGES:"+str(time //4)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	return status_v

def run_test_alg_on_sp(M_adj_dist,cells,point1,point2,wind_name,city,destination,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	confronto_energetico = open("result/vconfronto_energetico_on_sp"+str(indice_ora)+".txt","a+")
	path_flag = True
	consumption_go = 0
	consumption_return= 0
	distance = 0
	egdes_must_delete=[]
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,payload)
	G = nx.Graph(M_adj_energy)
	G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
	time = 0
	STATUS = ""
	sp_cost = nx.dijkstra_path_length(G, source=SOURCE, target=destination, weight='weight') 
	sp_path = nx.dijkstra_path(G, source=SOURCE, target=destination, weight='weight') 
	confronto_energetico.write("node->node_s\n")
	node = sp_path.pop(0)
	confronto_energetico.write(str(node))
	while node != destination and path_flag:
		control_time = time // 15
		#print(time)
		if control_time == 0:
			node_next = sp_path.pop(0)
			confronto_energetico.write("->"+str(node_next))
			consumption_go = consumption_go + G[node][node_next]['weight']
			time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
			distance = distance + M_adj_dist[node][node_next]
			egdes_must_delete.append((node,node_next))
			#print(node,node_next,M_adj_wind[node][node_next],M_adj_angle[node][node_next])
			#print(egdes_must_delete)
			node = node_next
		else:
			indice_orario = (indice_orario + int(control_time)) % 240
			if indice_orario <= 1:
				indice_orario = indice_orario + 2
			#print(egdes_must_delete)
			M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,payload)
			G = nx.Graph(M_adj_energy)
			G = delete_edges(G,egdes_must_delete)
			sp_path = nx.dijkstra_path(G, source=node, target=destination, weight='weight')
			if sp_path != []:
				node = sp_path.pop(0)
				node_next = sp_path.pop(0)
				confronto_energetico.write("->"+str(node_next))
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				egdes_must_delete.append((node,node_next))
				#print(node)
				#print(node,node_next,M_adj_wind[node][node_next],M_adj_angle[node][node_next])
				node = node_next
				#print(node)
			else:
				path_flag = False

			#print(M_adj_angle[node][node_next],M_adj_wind_gw[node][node_next],M_adj_angle[node_next][node],M_adj_wind_gw[node_next][node])
		#print(consumption_go,time,distance)
	
	if consumption_go <= BUDGET_ENERGY and path_flag:
		G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
		sp_return = nx.dijkstra_path(G1, source=destination, target=SOURCE, weight='weight')
		egdes_must_delete = []
		node_r = sp_return.pop(0)
		confronto_energetico.write("->"+str(node_r))
		while node_r != SOURCE and path_flag:
			control_time = time // 15
			if control_time == 0:
				node_r_next = sp_return.pop(0)
				confronto_energetico.write("->"+str(node_r_next))
				consumption_return = consumption_return + G1[node_r][node_r_next]['weight']
				time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node_r][node_r_next]
				egdes_must_delete.append((node_r,node_r_next))
				#print(node_r,node_r_next,M_adj_wind[node_r][node_r_next],M_adj_angle[node_r][node_r_next])
				node_r = node_r_next
			else:
				indice_orario = (indice_orario + int(control_time)) % 240
				if indice_orario <= 1:
					indice_orario = indice_orario + 2
				M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,0)
				G1 = nx.Graph(M_adj_energy)
				G1 = delete_edges(G1,egdes_must_delete)
				sp_return = nx.dijkstra_path(G1, source=node_r, target=SOURCE, weight='weight')
				if sp_return != []:
					node_r = sp_return.pop(0)
					node_r_next = sp_return.pop(0)
					confronto_energetico.write("->"+str(node_r_next))
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					egdes_must_delete.append((node_r,node_r_next))
					#print(node_r,node_r_next,M_adj_wind[node_r][node_r_next],M_adj_angle[node_r][node_r_next])
					node_r = node_r_next
				else: 
					path_flag = False
			#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
		if (consumption_go+consumption_return) <= BUDGET_ENERGY:
			STATUS="SUCCESS"
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			status_v[STATUS] = status_v[STATUS] +1
		else:
			STATUS="DELIVERED"
			status_v[STATUS] = status_v[STATUS] +1
			confronto_energetico.write("\n*(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
	else:
		STATUS="FAIL"
		status_v[STATUS] = status_v[STATUS] +1
	confronto_energetico.write("\n WIND CHANGES:"+str(time // 4)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	return status_v

def run_test_alg_on_g(M_adj_dist,cells,point1,point2,wind_name,city,destination,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	confronto_energetico = open("result/vconfronto_energetico_on_g"+str(indice_ora)+".txt","a+")
	stop_flag = False
	consumption_go = 0
	consumption_return= 0
	distance = 0
	egdes_must_delete=[]
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,payload)
	G = nx.Graph(M_adj_energy)
	G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
	time = 0
	STATUS = ""
	sp_cost = nx.shortest_path_length(G, source=SOURCE, target=destination, weight='weight') 
	sp_path = nx.shortest_path(G, source=SOURCE, target=destination, weight='weight') 
	cycle_cost = nx.shortest_path_length(G1, source=destination, target=SOURCE, weight='weight') + sp_cost 
	node = SOURCE
	while node != destination and not(stop_flag):
		control_time = time // 15
		if control_time == 0:
			G = delete_edges(G,egdes_must_delete)
			neighbors = list(nx.neighbors(G,node))
			if neighbors != []:
				node_next = neighbors.pop(0)
				for n in neighbors:
					if G[node][n]['weight'] < G[node][node_next]['weight']:
						node_next = n
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				egdes_must_delete.append((node,node_next))
				#print(egdes_must_delete)
				node = node_next
			else:
				stop_flag = True
		else:
			indice_orario = (indice_orario + int(control_time)) % 240
			if indice_orario <= 1:
				indice_orario = indice_orario + 2
			#print(egdes_must_delete)
			#print(indice_orario)
			M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,payload)
			G = nx.Graph(M_adj_energy)
			G = delete_edges(G,egdes_must_delete)
			neighbors = list(nx.neighbors(G,node))
			if neighbors != []:
				node_next = neighbors.pop(0)
				for n in neighbors:
					if G[node][n]['weight'] < G[node][node_next]['weight']:
						node_next = n
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				egdes_must_delete.append((node,node_next))
				#print(node)
				node = node_next
				#	print(node)
			else:
				stop_flag = True
			#print(M_adj_angle[node][node_next],M_adj_wind_gw[node][node_next],M_adj_angle[node_next][node],M_adj_wind_gw[node_next][node])
		#print(consumption_go,time,distance)

	G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
	egdes_must_delete = []
	node_r = destination
	if consumption_go <= BUDGET_ENERGY and not(stop_flag):
		while node_r != SOURCE and not(stop_flag):
			control_time = time // 15
			if control_time == 0:
				G1 = delete_edges(G1,egdes_must_delete)
				neighbors = list(nx.neighbors(G1,node_r))
				if neighbors != []:
					node_r_next = neighbors.pop(0)
					for n in neighbors:
						if G1[node_r][n]['weight'] < G1[node_r][node_r_next]['weight']:
							node_r_next = n
					consumption_return = consumption_return + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					egdes_must_delete.append((node_r,node_r_next))
					node_r = node_r_next
				else:
					stop_flag = True
			else:
				indice_orario = (indice_orario + int(control_time)) % 240
				if indice_orario <= 1:
					indice_orario = indice_orario + 2
				#print(indice_orario)
				M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,0)
				G1 = nx.Graph(M_adj_energy)
				G1 = delete_edges(G1,egdes_must_delete)
				neighbors = list(nx.neighbors(G1,node_r))
				if neighbors != []:
					node_r_next = neighbors.pop(0)
					for n in neighbors:
						if G1[node_r][n]['weight'] < G1[node_r][node_r_next]['weight']:
							node_r_next = n
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					egdes_must_delete.append((node_r,node_r_next))
					node_r = node_r_next
				else:
					stop_flag = True
			#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
		if (consumption_go+consumption_return) <= BUDGET_ENERGY:
			STATUS="SUCCESS"
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			status_v[STATUS] = status_v[STATUS] +1
		else:
			STATUS="DELIVERED"
			status_v[STATUS] = status_v[STATUS] +1
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
	else:
		STATUS="FAIL"
		status_v[STATUS] = status_v[STATUS] +1
		confronto_energetico.write("\nFINAL\n")
	confronto_energetico.write("\n WIND CHANGES:"+str(time // 4)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	return status_v

def test_GDP(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,indice_ora,PP_list):
	tic = time.perf_counter()
	fail = 0
	succes = 0
	delivered = 0
	canceled = 0
	final_status = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	status = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	loc = "result/vtest_alg_on_g"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/vtest_alg_on_g"+str(indice_ora)+".txt")
	loc = "result/vconfronto_energetico_on_g"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/vconfronto_energetico_on_g"+str(indice_ora)+".txt")
	print("Starting test_alg_on_g...")
	for destination in PP_list:
		status = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
		status = run_test_alg_on_g(M_adj_dist,cells,point1,point2,wind_name,city,destination,prefixs,PAYLOAD,status,indice_ora)
		fail = fail + status["FAIL"]
		delivered = delivered + status["DELIVERED"]
		succes = succes + status["SUCCESS"]
		canceled = canceled + status["CANCELED"]
	toc = time.perf_counter()
	final_status["FAIL"] = fail
	final_status["CANCELED"] = canceled
	final_status["DELIVERED"] = delivered
	final_status["SUCCESS"] = succes
	print(f"test_alg_on_g complete in {toc - tic:0.4f} seconds.")
	file = open("result/vconfronto_energetico_on_g"+str(indice_ora)+".txt","a+")
	file.write("\n\n"+str(status))

	return final_status

def test_DSP(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,indice_ora,PP_list):
	fail = 0
	succes = 0
	delivered = 0
	tic = time.perf_counter()
	status_o = {"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	final_status = {"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	loc = "result/vtest_alg_on_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/vtest_alg_on_sp"+str(indice_ora)+".txt")
	loc = "result/vconfronto_energetico_on_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/vconfronto_energetico_on_sp"+str(indice_ora)+".txt")
	print("Starting test_alg_on_sp...")
	for destination in PP_list:
		status_o = {"FAIL":0,"DELIVERED":0,"SUCCESS":0}
		status_o = run_test_alg_on_sp(M_adj_dist,cells,point1,point2,wind_name,city,destination,prefixs,PAYLOAD,status_o,indice_ora)
		fail = fail + status_o["FAIL"]
		delivered = delivered + status_o["DELIVERED"]
		succes = succes + status_o["SUCCESS"]
		#print(succes,delivered,fail)
	toc = time.perf_counter()
	final_status["FAIL"] = fail
	final_status["DELIVERED"] = delivered
	final_status["SUCCESS"] = succes
	print(f"test_alg_on_sp complete in {toc - tic:0.4f} seconds.")
	file = open("result/vtest_alg_on_sp"+str(indice_ora)+".txt","a+")
	file.write("\n\n"+str(status_o))
	return final_status

def test_OSP(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,indice_ora,PP_list):
	tic = time.perf_counter()
	fail = 0
	succes = 0
	delivered = 0
	canceled = 0
	final_status = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	status = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	loc = "result/vtest_alg_off_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/vtest_alg_off_sp"+str(indice_ora)+".txt")
	loc = "result/vconfronto_energetico_off_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/vconfronto_energetico_off_sp"+str(indice_ora)+".txt")
	print("Starting test_alg_off_sp...")
	for destination in PP_list:
		status = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
		status = run_test_alg_off_sp(M_adj_dist,cells,point1,point2,wind_name,city,destination,prefixs,PAYLOAD,status,indice_ora)
		fail = fail + status["FAIL"]
		delivered = delivered + status["DELIVERED"]
		succes = succes + status["SUCCESS"]
		canceled = canceled + status["CANCELED"]
	toc = time.perf_counter()
	final_status["FAIL"] = fail
	final_status["CANCELED"] = canceled
	final_status["DELIVERED"] = delivered
	final_status["SUCCESS"] = succes
	print(f"test_alg_off_sp complete in {toc - tic:0.4f} seconds.")
	file = open("result/vtest_alg_off_sp"+str(indice_ora)+".txt","a+")
	file.write("\n\n"+str(status))
	return final_status

def compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,payload,drone_speed):
	N = len(M_adj_dist)
	M_adj_energy = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			if M_adj_dist[i][j] != 0:
				M_adj_energy[i][j] = (M_adj_dist[i][j]*100) * prefixs[(payload,drone_speed,M_adj_wind[i][j],M_adj_angle[i][j])]
	return nx.Graph(M_adj_energy)

def n_random_test(n,M_adj_dist,M_adj_angle,prefixs,cells,point1,point2,wind_name,city):
	#indici_orari_test = random.sample(range(2, 240), n)
	indici_orari_test = [155, 97, 16, 89, 67, 186, 139, 142, 109, 17, 66, 52, 35, 225, 125, 31, 43, 50, 108, 14, 83, 169, 183, 217, 136, 226, 19, 220, 206, 77]
	Gminp, Gmin, Gmaxp, Gmax = min_max_energy_matrix(M_adj_dist,prefixs,DRONE_SPEED,PAYLOAD)
	colors_list = Preprocessing_PP(SOURCE,Gminp,Gmin,Gmaxp,Gmax)
	final_d={"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	final_o = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	final_g = {"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	gray_list= []
	for i,color in colors_list:
		if color == 'GRAY':
			gray_list.append(i)
	print("N° GRAY VERTEX: ",len(gray_list))
	for i in indici_orari_test:
		#final_g = test_GDP(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,i,gray_list)
		final_g =  Counter(final_g)+Counter(test_GDP(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,i,gray_list))
		#final_d =  Counter(final_d)+Counter(test_DSP(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,i,gray_list))
		#final_o =  Counter(final_o)+Counter(test_OSP(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,i,gray_list))
	loc = "result/digest_result.txt"
	if(os.path.isfile(loc)):
		os.remove("result/digest_result.txt")
	file = open("result/digest_result.txt","a+")
	print("VORONOI TEST")
	#print("FINAL RESULT DSP ALGORITHM: "+str(final_d)+"\n")
	#print("FINAL RESULT OSP ALGORITHM: "+str(final_o)+"\n")
	print("FINAL RESULT GDP ALGORITHM: "+str(final_g)+"\n")
	#file.write("FINAL RESULT GDP ALGORITHM:\n"+str(final_g)+"\nFINAL RESULT DSP ALGORITHM:\n"+str(final_d)+"\nFINAL RESULT OSP ALGORITHM:\n"+str(final_o))
	file.write("\nFINAL RESULT DSP ALGORITHM:\n"+str(final_d)+"\nFINAL RESULT OSP ALGORITHM:\n"+str(final_o))
	return
	
def MODrun_test_alg_off_sp(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	confronto_energetico = open("result/confronto_energetico_off_sp"+str(indice_ora)+".txt","a+")
	consumption_go = 0
	consumption_return= 0
	distance = 0
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_energy[0][15] = 12
	M_adj_energy[15][17]= 50
	M_adj_energy[17][1]= 60
	M_adj_energy[15][1] = 3
	G = nx.Graph(M_adj_energy)
	time = 0
	STATUS = ""
	sp_cost = nx.shortest_path_length(G, source=0, target=1, weight='weight') 
	sp_path = nx.shortest_path(G, source=0, target=1, weight='weight') 
	cycle_cost =  sp_cost
	#sp_return = nx.shortest_path(G1, source=1, target=0, weight='weight') 
	node = sp_path.pop(0)
	#node_r = sp_return.pop(0)
	print(sp_path)
	if cycle_cost <= BUDGET_ENERGY:
		while node != 1:
			control_time = time // 4
			if control_time == 0:
				node_next = sp_path.pop(0)
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				#print(str(M_adj_wind[node][node_next]))
				print(time)
				node = node_next
			else:
				indice_orario = (indice_orario + int(control_time)) % 240
				if indice_orario <= 1:
					indice_orario = indice_orario + 2
				M_adj_energy[0][15] = 12
				M_adj_energy[15][17]= 3
				M_adj_energy[17][1]= 2
				M_adj_energy[15][1] = 40
				#M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,payload)
				G = nx.Graph(M_adj_energy)
				node_next = sp_path.pop(0)
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				node = node_next
		print(consumption_go)
			#print(M_adj_angle[node][node_next],M_adj_wind_gw[node][node_next],M_adj_angle[node_next][node],M_adj_wind_gw[node_next][node])
		#print(consumption_go,time,distance)
		"""print(sp_return)
		G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
		if consumption_go <= BUDGET_ENERGY:
			while node_r != SOURCE:
				control_time = time // 15
				if control_time == 0:
					node_r_next = sp_return.pop(0)
					consumption_return = consumption_return + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					#print(M_adj_wind[node_r][node_r_next])
					node_r = node_r_next
				else:
					indice_orario = (indice_orario + int(control_time)) % 240
					if indice_orario <= 1:
						indice_orario = indice_orario + 2
					M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,0)
					G1 = nx.Graph(M_adj_energy)
					node_r_next = sp_return.pop(0)
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					node_r = node_r_next
				#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
			if (consumption_go+consumption_return) <= BUDGET_ENERGY:
				STATUS="SUCCESS"
				confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
				status_v[STATUS] = status_v[STATUS] +1
			else:
				STATUS="DELIVERED"
				status_v[STATUS] = status_v[STATUS] +1
				confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
		else:
			STATUS="FAIL"
			status_v[STATUS] = status_v[STATUS] +1
			confronto_energetico.write("\nFAIL")
	else:
		STATUS = "CANCELED"
		status_v[STATUS] = status_v[STATUS] +1
		confronto_energetico.write("\nABORT")
	confronto_energetico.write("\n WIND CHANGES:"+str(time // 15)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	"""
	return status_v

def MODrun_test_alg_on_sp(M_adj_dist,cells,point1,point2,wind_name,city,destination,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	confronto_energetico = open("result/confronto_energetico_on_sp"+str(indice_ora)+".txt","a+")
	path_flag = True
	consumption_go = 0
	consumption_return= 0
	distance = 0
	egdes_must_delete=[]
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_energy[0][15] = 12
	M_adj_energy[15][17]= 50
	M_adj_energy[17][1]= 60
	M_adj_energy[15][1] = 3
	G = nx.Graph(M_adj_energy)
	time = 0
	STATUS = ""
	sp_cost = nx.shortest_path_length(G, source=SOURCE, target=destination, weight='weight') 
	sp_path = nx.shortest_path(G, source=SOURCE, target=destination, weight='weight') 
	print(sp_path)
	node = sp_path.pop(0)
	while node != destination and path_flag:
		control_time = time // 4
		if control_time == 0:
			print(sp_path)
			node_next = sp_path.pop(0)
			consumption_go = consumption_go + G[node][node_next]['weight']
			time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
			distance = distance + M_adj_dist[node][node_next]
			egdes_must_delete.append((node,node_next))
			#print(node,node_next,M_adj_wind[node][node_next],M_adj_angle[node][node_next])
			#print(egdes_must_delete)
			node = node_next
		else:
			indice_orario = (indice_orario + int(control_time)) % 240
			if indice_orario <= 1:
				indice_orario = indice_orario + 2
			#print(egdes_must_delete)
			M_adj_energy[0][15] = 12
			M_adj_energy[15][17]= 3
			M_adj_energy[17][1]= 2
			M_adj_energy[15][1] = 40
			G = nx.Graph(M_adj_energy)
			G = delete_edges(G,egdes_must_delete)
			sp_path = nx.shortest_path(G, source=node, target=destination, weight='weight')
			print(sp_path) 
			if sp_path != []:
				node = sp_path.pop(0)
				node_next = sp_path.pop(0)
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				egdes_must_delete.append((node,node_next))
				#print(node)
				#print(node,node_next,M_adj_wind[node][node_next],M_adj_angle[node][node_next])
				node = node_next
				#print(node)
			else:
				path_flag = False
		print(consumption_go)
			#print(M_adj_angle[node][node_next],M_adj_wind_gw[node][node_next],M_adj_angle[node_next][node],M_adj_wind_gw[node_next][node])
		#print(consumption_go,time,distance)
	
	"""if consumption_go <= BUDGET_ENERGY and path_flag:
		G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
		sp_return = nx.shortest_path(G1, source=destination, target=SOURCE, weight='weight')
		egdes_must_delete = []
		print(sp_return)
		node_r = sp_return.pop(0)
		while node_r != SOURCE and path_flag:
			control_time = time // 15
			if control_time == 0:
				print(sp_return)
				node_r_next = sp_return.pop(0)
				consumption_return = consumption_return + G1[node_r][node_r_next]['weight']
				time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node_r][node_r_next]
				egdes_must_delete.append((node_r,node_r_next))
				#print(node_r,node_r_next,M_adj_wind[node_r][node_r_next],M_adj_angle[node_r][node_r_next])
				node_r = node_r_next
			else:
				indice_orario = (indice_orario + int(control_time)) % 240
				if indice_orario <= 1:
					indice_orario = indice_orario + 2
				M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_vor(M_adj_dist,cells,point1,point2,wind_name,city,indice_orario,prefixs,0)
				G1 = nx.Graph(M_adj_energy)
				G1 = delete_edges(G1,egdes_must_delete)
				sp_return = nx.shortest_path(G1, source=node_r, target=SOURCE, weight='weight')
				print(sp_return)
				if sp_return != []:
					node_r = sp_return.pop(0)
					node_r_next = sp_return.pop(0)
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*100/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					egdes_must_delete.append((node_r,node_r_next))
					#print(node_r,node_r_next,M_adj_wind[node_r][node_r_next],M_adj_angle[node_r][node_r_next])
					node_r = node_r_next
				else: 
					path_flag = False
			#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
		if (consumption_go+consumption_return) <= BUDGET_ENERGY:
			STATUS="SUCCESS"
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			status_v[STATUS] = status_v[STATUS] +1
		else:
			STATUS="DELIVERED"
			status_v[STATUS] = status_v[STATUS] +1
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
	else:
		STATUS="FAIL"
		status_v[STATUS] = status_v[STATUS] +1
	confronto_energetico.write("\n WIND CHANGES:"+str(time // 15)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	"""
	return status_v
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
DFvor = pd.read_csv("edge_by_edge_wind.csv",sep=';')
point1 = DFvor['vert1'].tolist()
point2 = DFvor['vert2'].tolist()
wind_name = DFvor['wind'].tolist()
vor_to_wind = vor_to_name(point1,point2,wind_name)
points=np.c_[lat, lon]
vor = spatial.Voronoi(points)
box = (lat.min()-1, lon.min()-1,lat.max()+1, lon.max()+1)
m = smopy.Map(box, z=9)
ax = m.show_mpl(figsize=(30, 15))
for i in range(len(lon)):
	x,y = m.to_pixels(lat[i],lon[i])
	ax.plot(x, y, 'or', ms=5, mew=2)
	if(len(city[i])>6):
		ax.text(x-10, y-10, city[i][:6].replace('_', ''), fontsize=6)
	else:
		ax.text(x-10, y-10, city[i].replace('_', ''), fontsize=6)
points=np.c_[lat, lon]
"""
	Voronoi:
	-region: definisce in base al seed a cui fa riferimento l'insieme dei vertici che definisce il poligono

"""
vor = spatial.Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)
prefixs = sfg.compute_prefixes()
lat_index_tot, inv_lat_index_tot = coor_to_index_dict(points,vertices,regions)
M_adj_dist,M_adj_angle = to_adj_matrix(vor,regions,vertices,lat_index_tot,inv_lat_index_tot)
N = len(M_adj_angle)
M_adj_angle_gw = np.zeros((N,N))
M_adj_wind = np.zeros((N,N))
M_adj_energy = np.zeros((N,N))
cells = [vertices[region] for region in regions]
iteratore = 0
for array in cells:
	winfo = hour_wind_data[city[iteratore]]
	for i in array:
		index_vor = lat_index_tot[(i[0],i[1])]
		#print(iteratore,index_vor)
		M_adj_wind[iteratore][index_vor]= winfo[3]
		M_adj_wind[index_vor][iteratore] = winfo[3]
		M_adj_angle_gw[iteratore][index_vor] = winfo[2]
		M_adj_angle_gw[index_vor][iteratore] = winfo[2]
		app = M_adj_angle[iteratore][index_vor]-winfo[2]
		if app < 0:
			M_adj_angle[iteratore][index_vor] = app+360
		else:
			M_adj_angle[iteratore][index_vor] = app 
		app = M_adj_angle[index_vor][iteratore]-winfo[2]
		if app < 0:
			M_adj_angle[index_vor][iteratore] = app+360
		else:
			M_adj_angle[index_vor][iteratore] = app 
	iteratore = iteratore + 1
for i in range(len(point1)):
	winfo = hour_wind_data[wind_name[i]]
	M_adj_wind[point1[i]][point2[i]] = winfo[3]
	M_adj_angle_gw[point1[i]][point2[i]] = winfo[2]
	app = M_adj_angle[point1[i]][point2[i]]-winfo[2]
	if app < 0:
		M_adj_angle[point1[i]][point2[i]] = app+360
	else:
		M_adj_angle[point1[i]][point2[i]] = app 
M_adj_angle = sfg.aprox_angle(M_adj_angle)
M_adj_wind = sfg.approx_wind(M_adj_wind,M_adj_dist)
status_v = {}
for i in range(N):
	for j in range(N):
		if M_adj_dist[i][j] != 0:
			M_adj_energy[i][j] = (M_adj_dist[i][j]*100) * prefixs[(0,10,M_adj_wind[i][j],M_adj_angle[i][j])]
			
Number_of_exe = 30
G = nx.Graph(M_adj_energy)
M_unitary = M_adj_energy
for i in range(len(M_unitary)):
	for j in range(len(M_unitary)):
		if M_adj_energy[i][j] != 0:
			if (M_adj_dist[i][j] - 100) > 0:
				M_unitary[i][j]=1000000000
			else:
				M_unitary[i][j]=1


Gu = nx.Graph(M_adj_dist)
avg = 0
euc = []
path = []
conta = 0
for source in range(1):
	for destination in range(0,12):
		if source != destination:
			path.append(nx.shortest_path_length(Gu, source=source, target=destination, weight='weight'))
			x0= lat[source]
			y0 = lon[source]
			x1 = lat[destination]
			y1 = lon[destination]
			euc.append(distance_coor_two_points(x0,y0,x1,y1))
			conta = conta +1
somma = 0
somma1 = 0 
for i in range(len(euc)):
	print("Distance Euclidean: "+str(euc[i])+" Distance Delaunay: "+str(path[i])+" Ratio: "+str(round(path[i]/euc[i],2)))
	somma = somma + euc[i]
	somma1 = somma1 + path[i]
print("Final ratio: "+str(round(somma1/somma,2)))
print("Numero dei vertici: "+str(Gu.number_of_nodes()))
print("Numero degli archi: "+str(Gu.number_of_edges()))
green = 0
black = 0
gray = 0
for i in range(0,12):

	Gminp, Gmin, Gmaxp, Gmax = min_max_energy_matrix(M_adj_dist,prefixs,DRONE_SPEED,7)
	color = Preprocessing_PP(i,Gminp,Gmin,Gmaxp,Gmax)
	for c in color:
		if c[1] =="GREEN":
			green = green +1
		if c[1] == "BLACK":
			black = black + 1
		if c[1] == "GRAY":
			gray = gray + 1
print("GREEN: "+str(green -12)+" BLACK: "+str(black)+" GRAY: "+str(gray)) 
exit()

#MODrun_test_alg_off_sp(M_adj_dist,cells,point1,point2,wind_name,city,prefixs,2,status_v,1)
#MODrun_test_alg_on_sp(M_adj_dist,cells,point1,point2,wind_name,city,1,prefixs,2,status_v,1)
n_random_test(Number_of_exe,M_adj_dist,M_adj_angle,prefixs,cells,point1,point2,wind_name,city)