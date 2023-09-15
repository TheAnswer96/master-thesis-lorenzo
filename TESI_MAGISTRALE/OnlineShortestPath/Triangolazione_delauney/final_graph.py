import numpy as np
import pandas as pd
import math
from networkx import nx
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import smopy
import time
import matplotlib.lines as mlines
import xlrd 
import os
from sympy import Symbol, nsolve
import sympy as sp
import numpy as np
import mpmath
import time
import random

mpmath.mp.dps = 5
BUDGET_ENERGY = 50000 #totale carica batteria Kj
PAYLOAD = 2 #peso del carico Kg
DRONE_SPEED = 10#velocità del drone media m/s
SOURCE = 0
INDICE_ORARIO = 2

def get_energy(distance, payload_weight, drone_speed, wind_speed, wind_direction):

	# start calculations
	m_package = payload_weight
	m_drone = 7 #peso del drone
	m_battery = 10#peso batteria

	num_rotors = 8#numero rotori
	diameter = 0.432#

	# s_battery = 540000
	# delta = 0.5
	# f = 1.2

	pressure = 100726  # 50 meters above sea level
	R = 287.058
	temperature = 15 + 273.15  # 15 degrees in Kelvin
	rho = pressure / (R*temperature)

	g = 9.81

	# power efficiency
	eta = 0.7

	drag_coefficient_drone = 1.49
	drag_coefficient_battery = 1
	drag_coefficient_package = 2.2

	projected_area_drone = 0.224
	projected_area_battery = 0.015
	projected_area_package = 0.0929

	v_north = drone_speed - wind_speed*np.cos(np.deg2rad(wind_direction))
	v_east = - wind_speed*np.sin(np.deg2rad(wind_direction))
	v_air = np.sqrt(v_north**2 + v_east**2)

	# Drag force
	F_drag_drone = 0.5 * rho * (v_air**2) * drag_coefficient_drone * projected_area_drone
	F_drag_battery = 0.5 * rho * (v_air**2) * drag_coefficient_battery * projected_area_battery
	F_drag_package = 0.5 * rho * (v_air**2) * drag_coefficient_package * projected_area_package

	F_drag = F_drag_drone + F_drag_battery + F_drag_package

	alpha = np.arctan(F_drag / ((m_drone + m_battery + m_package)*g))

	# Thrust
	T = (m_drone + m_battery + m_package)*g + F_drag

	# # Power min hover
	# P_min_hover = (T**1.5) / (np.sqrt(0.5 * np.pi * num_rotors * (diameter**2) * rho))

	# v_i = Symbol('v_i')
	# f_0 = v_i - (2*T / (np.pi * num_rotors * (diameter**2) * rho * sp.sqrt((drone_speed*sp.cos(alpha))**2 + (drone_speed*sp.sin(alpha) + v_i)**2)))
	# induced_speed = float(nsolve(f_0, v_i, 5))
	# print(induced_speed)

	tmp_a = 2*T
	tmp_b = np.pi * num_rotors * (diameter**2) * rho
	tmp_c = (drone_speed*sp.cos(alpha))**2
	tmp_d = drone_speed*sp.sin(alpha)
	tmp_e = tmp_a / tmp_b

	coeff = [1, (2*tmp_d), (tmp_c+tmp_d**2), 0, -tmp_e**2]
	sol = np.roots(coeff)
	induced_speed = float(max(sol[np.isreal(sol)]).real)
	# print(induced_speed)

	# Power min to go forward
	P_min = T*(drone_speed*np.sin(alpha) + induced_speed)

	# expended power
	P = P_min / eta

	# energy efficiency of travel
	mu = P / drone_speed

	# Energy consumed
	E = mu * distance

	# # Range of a drone
	# R = (m_battery * s_battery * delta) / (e * f)
	#diviso 10 per
	return E/1000.

def compute_prefixes():
	distance = 1 #un metro
	payload_weights = [0, 2]
	drone_speeds = [10, 20]
	global_wind_speeds = [0, 5, 10, 15] #sto cambiando -- era [0, 5, 10, 15]
	relative_wind_directions = [0, 45, 135, 180]

	prefix = {}
	for p in payload_weights:
		for v in drone_speeds:
			for ws in global_wind_speeds:
				for wd in relative_wind_directions:
					prefix[(p, v, ws, wd)] = get_energy(distance, p, v, ws, wd)

	return prefix

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

def city_to_coor(lat,lon,city):
	dictionary = {}
	for index in range(len(lat)):
		couple = (lat[index],lon[index])
		dictionary[city[index]] = couple
		inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary

def midpoint(x1,y1,x2,y2):
	lonA = math.radians(y1)
	lonB = math.radians(y2)
	latA = math.radians(x1)
	latB = math.radians(x2)

	dLon = lonB - lonA

	Bx = math.cos(latB) * math.cos(dLon)
	By = math.cos(latB) * math.sin(dLon)

	latC = math.atan2(math.sin(latA) + math.sin(latB), math.sqrt((math.cos(latA) + Bx) * (math.cos(latA) + Bx) + By * By))
	lonC = lonA + math.atan2(By, math.cos(latA) + Bx)
	lonC = (lonC + 3 * math.pi) % (2 * math.pi) - math.pi

	return (math.degrees(latC), math.degrees(lonC))

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

def city_to_dict(lat,lon):
	dictionary = {}
	for index in range(len(lat)):
		couple = (lat[index],lon[index])
		dictionary[couple] = index
	inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary

def cityindex_to_dict(city):
	dictionary = {}
	for index in range(len(city)):
		dictionary[index] = city[index]
	inv_dictionary = {v: k for k, v in dictionary.items()}
	return dictionary,inv_dictionary

def angleFromCoordinate(lat1, long1, lat2, long2):
	dLon = (long2 - long1)

	y = math.sin(dLon) * math.cos(lat2)
	x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

	brng = math.atan2(y, x)

	brng = math.degrees(brng)
	brng = (brng + 360) % 360
	#brng = 360 - brng # count degrees clockwise - remove to make counter-clockwise

	return brng

def aprox_angle(M_adj_angle):
	N = len(M_adj_angle)
	M_aprox = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
				if M_adj_angle[i][j] <= 45 or M_adj_angle[i][j] >= 315:
					M_aprox[i][j] = 0
				elif M_adj_angle[i][j] <= 90 or M_adj_angle[i][j] >= 270:
					M_aprox[i][j] = 45
				elif M_adj_angle[i][j] <= 135 or M_adj_angle[i][j] >= 225:
					M_aprox[i][j] = 135
				else:
					M_aprox[i][j] = 180
	return M_aprox

def approx_wind(M_adj_wind,M_adj_dist):
	N = len(M_adj_wind)
	M_aprox = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			if M_adj_dist[i][j] != 0:
				if M_adj_wind[i][j] <= 4 and M_adj_wind[i][j] >= 0:
					M_aprox[i][j] = 0
				elif M_adj_wind[i][j] <= 9 and M_adj_wind[i][j] > 4:
					M_aprox[i][j] = 5
				elif M_adj_wind[i][j] <= 14 and M_adj_wind[i][j] > 9:
					M_aprox[i][j] = 10
				else:
					M_aprox[i][j] = 15
	return M_aprox
def wind_data_load(city_name, index):
	cities_parameter = {}
	for name in city_name:
		loc = "wind_dataset/"+ str(name) +".xls"
		if(os.path.isfile(loc)):
			wb = xlrd.open_workbook(loc) 
			sheet = wb.sheet_by_index(0) 
			sheet.cell_value(0, 0) 
			cities_parameter[name]= sheet.row_values(index)
		else:
			print("ERR: "+name+" file not found.")
	"""
		l'output si struttura come una lista composta da [id,data,DD,FF,DXI,FXI]
	"""
	return cities_parameter
def print_table1(prefixes):
	distance = 1 #un metro
	payload_weights = [0, 2]
	drone_speeds = [10, 20]
	global_wind_speeds = [0, 5, 10, 15] #sto cambiando -- era [0, 5, 10, 15]
	relative_wind_directions = [0, 45, 135, 180]
	file1 = open("tabella1.txt","w")
	for pl in payload_weights:
		file1.write("\nPAYLOAD WEIGHT: "+str(pl)+" KG\n\n")
		for ds in drone_speeds:
			file1.write("\nDRONE SPEED: "+str(ds)+" m/s\n")
			for speed in global_wind_speeds:
				file1.write("GLOBAL WIND SPEED: "+str(speed)+"\n")
				for dire in relative_wind_directions:
					file1.write("RELATIVE WIND DIRECTION: "+str(dire)+" UNIT CONSUMPTION(1 METER): "+str(round(get_energy(distance,pl,ds,speed,dire),2))+"\n")

	return
def print_table2(M_dist,M_angle,M_wind,now_wind,index_to_name):
	payload_weights = [0, 2]
	drone_speeds = [10, 20]
	N = len(M_dist)
	file1 = open("tabella2.txt","w")
	for pl in payload_weights:
		file1.write("\nPAYLOAD WEIGHT: "+str(pl)+" KG\n")
		for sp in drone_speeds:
			file1.write("\nDRONE SPEED: "+str(sp)+" m/s\n\n")
			for i in range(N):
				for j in range(N):
					if(M_dist[i][j] != 0):
						name1 = index_to_name[i]
						name2 = index_to_name[j]
						file1.write("EDGE BETWEEN:("+ name1+","+name2+")\n")
						time = round((M_dist[i][j]*1000/sp)/60,2)
						file1.write("EDGE LENGTH: "+str(round(M_dist[i][j],2))+"KM, TRAVERSING TIME: "+str(time)+"min, GLOBAL WIND DIRECTION: "+str(now_wind[i][j])+" RELATIVE WIND DIRECTION: "+str(M_angle[i][j])+", mu(e)*lunghezza: "+str(round(M_adj_dist[i][j]*1000*get_energy(1,pl,sp,M_wind[i][j],M_angle[i][j]),2))+"KJ\n")
	return
def plot_table1(prefixes):
	distance = 1 #un metro
	payload_weights = [0, 2]
	drone_speeds = [10, 20]
	global_wind_speeds = [0, 5, 10, 15] #sto cambiando -- era [0, 5, 10, 15]
	relative_wind_directions = [0, 45, 135, 180]
	fig, (ax1,ax2) = plt.subplots(1, 2)
	fig.suptitle('TABLE 1')
	ax1.set_title("PESO DEL CARICO: 2Kg")
	dataxs = []
	datays = []
	for ds in drone_speeds:
		for speed in global_wind_speeds:
			for dire in relative_wind_directions:
				dataxs.append(speed)
				datays.append(round(get_energy(distance,2,ds,speed,dire),2))
		ax1.plot(dataxs,datays, label="wind speed "+str(speed)+"m/s")	
	ax2.set_title("PESO DEL CARICO: 0Kg")
	for ds in drone_speeds:
		for speed in global_wind_speeds:
			for dire in relative_wind_directions:
				dataxs.append(speed)
				datays.append(round(get_energy(distance,0,ds,speed,dire),2))
		ax2.plot(dataxs,datays, label="wind speed "+str(speed)+"m/s")	
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	plt.show()
	return
def Merge(dict1, dict2): 
	res = {**dict1, **dict2} 
	return res 
      
def master_dictionary(dictionary1,dictionary2,dictionary3):
	from collections import defaultdict
	inv_dictionary = {}
	master_dict = {}
	master_dict = Merge(master_dict,dictionary1)
	master_dict = Merge(master_dict,dictionary2)
	master_dict = Merge(master_dict,dictionary3)
	inv_dictionary = {tuple(v): k for k, v in master_dict.items()}

	return master_dict,inv_dictionary
def update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,index,prefixs,payload):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	windinfo=[]
	for i in range(12):
		loc = "wind_dataset/"+ str(M_index_to_name[i]) +".xls"
		if(os.path.isfile(loc)):
			wb = xlrd.open_workbook(loc) 
			sheet = wb.sheet_by_index(0) 
			sheet.cell_value(0, 0) 
			windinfo = sheet.row_values(index)
		else:
			print("ERR: "+name+" file not found.")
		for j in range(N):
			if M_adj_dist[i][j] != 0:
				x0,y0 = M_name_to_coor[M_index_to_name[i]]
				x1,y1 = M_name_to_coor[M_index_to_name[j]]
				if angleFromCoordinate(x0, y0, x1, y1)-winfo[2] < 0:
					M_adj_angle[i][j] = angleFromCoordinate(x0, y0, x1, y1)-winfo[2]+360
				else:
					M_adj_angle[i][j] = angleFromCoordinate(x0, y0, x1, y1)-winfo[2]
				if angleFromCoordinate(x1, y1, x0, y0)-winfo[2] <0:
					M_adj_angle[j][i] = angleFromCoordinate(x1, y1, x0, y0)-winfo[2] +360
				else:
					M_adj_angle[j][i] = angleFromCoordinate(x1, y1, x0, y0)-winfo[2]
				edge_angle1 = M_adj_angle[i][j] + M_adj_wind_gw[i][j]
				edge_angle2 = M_adj_angle[j][i] + M_adj_wind_gw[j][i]
				#print(windinfo[3],str(M_index_to_name[i]))
				M_adj_wind[i][j] = windinfo[3]
				M_adj_wind[j][i] = windinfo[3]
				M_adj_wind_gw[i][j] = windinfo[2]
				M_adj_wind_gw[j][i] = windinfo[2]
				#print(M_adj_angle[i][j],M_adj_angle[j][i])
		M_adj_angle = aprox_angle(M_adj_angle)
		M_adj_wind = approx_wind(M_adj_wind,M_adj_dist)
		for i in range(12):
			for j in range(N):
				if M_adj_dist[i][j] != 0:
					#print(M_adj_angle[i][j],M_adj_angle[j][i])
					M_adj_energy[i][j] = (M_adj_dist[i][j]*1000) * prefixs[(payload,DRONE_SPEED,M_adj_wind[i][j],M_adj_angle[i][j])]
					M_adj_energy[j][i] = (M_adj_dist[j][i]*1000) * prefixs[(payload,DRONE_SPEED,M_adj_wind[j][i],M_adj_angle[j][i])]
				
	return M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy
def min_max_energy_matrix(M_dist,prefixs,speed,payload):
	N = len(M_dist)
	lower_bound_matrix_p = np.zeros((N,N)) 
	lower_bound_matrix = np.zeros((N,N)) 
	upper_bound_matrix_p = np.zeros((N,N)) 
	upper_bound_matrix = np.zeros((N,N)) 
	for i in range(12):
		for j in range(N):
			if M_dist[i][j] != 0:
				lower_bound_matrix[i][j] = (M_dist[i][j]*1000)*prefixs[(0,speed,15,0)]
				lower_bound_matrix_p[i][j] = (M_dist[i][j]*1000)*prefixs[(payload,speed,15,0)]
				upper_bound_matrix[i][j] = (M_dist[i][j]*1000)*prefixs[(0,speed,15,180)]
				upper_bound_matrix_p[i][j] = (M_dist[i][j]*1000)*prefixs[(payload,speed,15,180)]
				lower_bound_matrix[j][i] = (M_dist[j][i]*1000)*prefixs[(0,speed,15,0)]
				lower_bound_matrix_p[j][i] = (M_dist[j][i]*1000)*prefixs[(payload,speed,15,0)]
				upper_bound_matrix[j][i] = (M_dist[j][i]*1000)*prefixs[(0,speed,15,180)]
				upper_bound_matrix_p[j][i] = (M_dist[j][i]*1000)*prefixs[(payload,speed,15,180)]

	return nx.Graph(lower_bound_matrix_p), nx.Graph(lower_bound_matrix), nx.Graph(upper_bound_matrix_p), nx.Graph(upper_bound_matrix)

def Preprocessing_PP(s,lower_M_p,lower_M,upper_M_p,upper_M):
	cicly_consume_min = {}
	cicly_consume_max = {}
	colors = []
	N = len(lower_M)
	colors.append("GREEN")
	for i in range(1,N):
		cicly_consume_min[i] = nx.shortest_path_length(lower_M_p, source=SOURCE, target=i, weight='weight') + nx.shortest_path_length(lower_M, source=i, target=SOURCE, weight='weight')
		cicly_consume_max[i] = nx.shortest_path_length(upper_M_p, source=SOURCE, target=i, weight='weight') + nx.shortest_path_length(upper_M, source=i, target=SOURCE, weight='weight')
		if  cicly_consume_min[i] > BUDGET_ENERGY:
			colors.append("BLACK")
		elif cicly_consume_max[i] > BUDGET_ENERGY and cicly_consume_min[i] <= BUDGET_ENERGY:
			colors.append("GRAY")
		else:
			colors.append("GREEN")
	return colors

def delete_edges(G,edges_delete):
	for e in edges_delete:
		u,v = e
		try:
			G.remove_edge(u, v)
		except Exception as e:
			continue
	return G

def run_test_alg_off_sp(M_adj_dist,M_index_to_name,M_name_to_coor,destination,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	file = open("result/test_alg_off_sp"+str(indice_ora)+".txt","a+")
	confronto_energetico = open("result/confronto_energetico_off_sp"+str(indice_ora)+".txt","a+")
	consumption_go = 0
	consumption_return= 0
	distance = 0
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,0)
	G = nx.Graph(M_adj_energy)
	G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
	time = 0
	STATUS = ""
	file.write("Static-Offline Shortest Path Test:(source,destination,energy_budget,payload_weight,drone_speed)->("+M_index_to_name[SOURCE]+","+M_index_to_name[destination]+","+str(BUDGET_ENERGY)+","+str(payload)+","+str(DRONE_SPEED)+")\n")
	confronto_energetico.write("\nOSP:(source,destination)->("+M_index_to_name[SOURCE]+","+M_index_to_name[destination]+")\n")
	sp_cost = nx.shortest_path_length(G, source=SOURCE, target=destination, weight='weight') 
	sp_path = nx.shortest_path(G, source=SOURCE, target=destination, weight='weight') 
	cycle_cost = nx.shortest_path_length(G1, source=destination, target=SOURCE, weight='weight') + sp_cost
	sp_return = nx.shortest_path(G1, source=destination, target=SOURCE, weight='weight') 
	file.write("\tPROJECTION ENERGY CONSUMPTION SP(SOURCE,DESTINATION): "+str(round(sp_cost,2))+"Kj PROJECTION ENERGY CONSUMPTION CYCLE(SOURCE,DESTINATION): "+str(round(cycle_cost,2))+"Kj\n")
	file.write("STEP BY STEP PATH\n")
	file.write("(HEAD,TAIL,DISTANCE,TIME,CONSUMPTION,Wd,Ws)")
	node = sp_path.pop(0)
	node_r = sp_return.pop(0)
	if cycle_cost <= BUDGET_ENERGY:
		while node != destination:
			control_time = time // 15
			if control_time == 0:
				node_next = sp_path.pop(0)
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				#print(str(M_adj_wind[node][node_next]))
				file.write("->(" + M_index_to_name[node] + "," + M_index_to_name[node_next] + "," + str(round(M_adj_dist[node][node_next],2)) + "," + str(round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)) + "," + str(round(G[node][node_next]['weight'],2)) + "," + str(M_adj_angle[node][node_next]) + "," + str(M_adj_wind[node][node_next])+")")
				node = node_next
			else:
				indice_orario = indice_orario + 1
				M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,payload)
				G = nx.Graph(M_adj_energy)
				node_next = sp_path.pop(0)
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				file.write("->(" + M_index_to_name[node] + "," + M_index_to_name[node_next] + "," + str(round(M_adj_dist[node][node_next],2)) + "," + str(round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)) + "," + str(round(G[node][node_next]['weight'],2)) + "," + str(M_adj_angle[node][node_next]) + "," + str(M_adj_wind[node][node_next])+")")
				node = node_next
			#print(M_adj_angle[node][node_next],M_adj_wind_gw[node][node_next],M_adj_angle[node_next][node],M_adj_wind_gw[node_next][node])
		#print(consumption_go,time,distance)
		G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
		if consumption_go <= BUDGET_ENERGY:
			while node_r != SOURCE:
				control_time = time // 15
				if control_time == 0:
					node_r_next = sp_return.pop(0)
					consumption_return = consumption_return + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					#print(M_adj_wind[node_r][node_r_next])
					file.write("->("+M_index_to_name[node_r]+","+M_index_to_name[node_r_next]+","+str(round(M_adj_dist[node_r][node_r_next],2))+","+str(round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2))+","+str(round(G1[node_r][node_r_next]['weight'],2))+","+str(M_adj_angle[node_r][node_r_next])+","+str(M_adj_wind[node_r][node_r_next])+")")
					node_r = node_r_next
				else:
					indice_orario = indice_orario + 1
					M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,0)
					G1 = nx.Graph(M_adj_energy)
					node_r_next = sp_return.pop(0)
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					file.write("->("+M_index_to_name[node_r]+","+M_index_to_name[node_r_next]+","+str(round(M_adj_dist[node_r][node_r_next],2))+","+str(round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2))+","+str(round(G1[node_r][node_r_next]['weight'],2))+","+str(M_adj_angle[node_r][node_r_next])+","+str(M_adj_wind[node_r][node_r_next])+")")
					node_r = node_r_next
				#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
			if (consumption_go+consumption_return) <= BUDGET_ENERGY:
				STATUS="SUCCESS"
				file.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")")
				confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
				status_v[STATUS] = status_v[STATUS] +1
			else:
				STATUS="DELIVERED"
				status_v[STATUS] = status_v[STATUS] +1
				confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
				file.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round(consumption_go+consumption_return,2))+","+str(round(distance,2))+","+str(round(time,2))+")")
		else:
			STATUS="FAIL"
			file.write("\n(*CONSUMPTION,*DISTANCE,*TIME)->("+str(round(consumption_go,2))+","+str(round(distance,2))+","+str(round(time,2))+")")
			confronto_energetico.write("\nFAIL")
	else:
		STATUS = "CANCELED"
		status_v[STATUS] = status_v[STATUS] +1
		confronto_energetico.write("\nABORT")
	file.write("\nMISSION STATUS: "+STATUS+" WIND CHANGES:"+str(indice_orario - start_orario)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	file.close()
	return status_v

def run_test_alg_on_sp(M_adj_dist,M_index_to_name,M_name_to_coor,destination,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	file = open("result/test_alg_on_sp"+str(indice_ora)+".txt","a+")
	confronto_energetico = open("result/confronto_energetico_on_sp"+str(indice_ora)+".txt","a+")
	path_flag = True
	consumption_go = 0
	consumption_return= 0
	distance = 0
	egdes_must_delete=[]
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,0)
	G = nx.Graph(M_adj_energy)
	G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
	time = 0
	STATUS = ""
	file.write("Dynamic Shortest Path Test:(source,destination,energy_budget,payload_weight,drone_speed)->("+M_index_to_name[SOURCE]+","+M_index_to_name[destination]+","+str(BUDGET_ENERGY)+","+str(payload)+","+str(DRONE_SPEED)+")\n")
	confronto_energetico.write("\nDSP:(source,destination)->("+M_index_to_name[SOURCE]+","+M_index_to_name[destination]+")\n")
	sp_cost = nx.shortest_path_length(G, source=SOURCE, target=destination, weight='weight') 
	sp_path = nx.shortest_path(G, source=SOURCE, target=destination, weight='weight') 
	cycle_cost = nx.shortest_path_length(G1, source=destination, target=SOURCE, weight='weight') + sp_cost 
	file.write("\tPROJECTION ENERGY CONSUMPTION SP(SOURCE,DESTINATION): "+str(round(sp_cost,2))+"Kj PROJECTION ENERGY CONSUMPTION CYCLE(SOURCE,DESTINATION): "+str(round(cycle_cost,2))+"Kj\n")
	file.write("STEP BY STEP PATH\n")
	file.write("(HEAD,TAIL,DISTANCE,TIME,CONSUMPTION,Wd,Ws)")
	#print(sp_path)
	node = sp_path.pop(0)
	while node != destination and path_flag:
		control_time = time // 15
		if control_time == 0:
			#print(sp_path)
			node_next = sp_path.pop(0)
			consumption_go = consumption_go + G[node][node_next]['weight']
			time = time + round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)
			distance = distance + M_adj_dist[node][node_next]
			file.write("->(" + M_index_to_name[node] + "," + M_index_to_name[node_next] + "," + str(round(M_adj_dist[node][node_next],2)) + "," + str(round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)) + "," + str(round(G[node][node_next]['weight'],2)) + "," + str(M_adj_angle[node][node_next]) + "," + str(M_adj_wind[node][node_next])+")")
			egdes_must_delete.append((node,node_next))
			#print(node,node_next,M_adj_wind[node][node_next],M_adj_angle[node][node_next])
			#print(egdes_must_delete)
			node = node_next
		else:
			indice_orario = indice_orario + 1
			#print(egdes_must_delete)
			M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,payload)
			G = nx.Graph(M_adj_energy)
			G = delete_edges(G,egdes_must_delete)
			sp_path = nx.shortest_path(G, source=node, target=destination, weight='weight')
			#print(sp_path) 
			if sp_path != []:
				node = sp_path.pop(0)
				node_next = sp_path.pop(0)
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				file.write("->("+ M_index_to_name[node] +","+ M_index_to_name[node_next] +","+ str(round(M_adj_dist[node][node_next],2)) + "," + str(round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)) + "," + str(round(G[node][node_next]['weight'],2)) + "," + str(M_adj_angle[node][node_next])+","+str(M_adj_wind[node][node_next])+")")
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
		sp_return = nx.shortest_path(G1, source=destination, target=SOURCE, weight='weight')
		egdes_must_delete = []
		#print(sp_return)
		node_r = sp_return.pop(0)
		while node_r != SOURCE and path_flag:
			control_time = time // 15
			if control_time == 0:
				#print(sp_return)
				node_r_next = sp_return.pop(0)
				consumption_return = consumption_return + G1[node_r][node_r_next]['weight']
				time = time + round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node_r][node_r_next]
				file.write("->("+M_index_to_name[node_r]+","+M_index_to_name[node_r_next]+","+str(round(M_adj_dist[node_r][node_r_next],2))+","+str(round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2))+","+str(round(G1[node_r][node_r_next]['weight'],2))+","+str(M_adj_angle[node_r][node_r_next])+","+str(M_adj_wind[node_r][node_r_next])+")")
				egdes_must_delete.append((node_r,node_r_next))
				#print(node_r,node_r_next,M_adj_wind[node_r][node_r_next],M_adj_angle[node_r][node_r_next])
				node_r = node_r_next
			else:
				indice_orario = indice_orario + 1
				M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,0)
				G1 = nx.Graph(M_adj_energy)
				G1 = delete_edges(G1,egdes_must_delete)
				sp_return = nx.shortest_path(G1, source=node_r, target=SOURCE, weight='weight')
				#print(sp_return)
				if sp_return != []:
					node_r = sp_return.pop(0)
					node_r_next = sp_return.pop(0)
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					file.write("->("+M_index_to_name[node_r]+","+M_index_to_name[node_r_next]+","+str(round(M_adj_dist[node_r][node_r_next],2))+","+str(round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2))+","+str(round(G1[node_r][node_r_next]['weight'],2))+","+str(M_adj_angle[node_r][node_r_next])+","+str(M_adj_wind[node_r][node_r_next])+")")
					egdes_must_delete.append((node_r,node_r_next))
					#print(node_r,node_r_next,M_adj_wind[node_r][node_r_next],M_adj_angle[node_r][node_r_next])
					node_r = node_r_next
				else: 
					path_flag = False
			#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
		if (consumption_go+consumption_return) <= BUDGET_ENERGY:
			STATUS="SUCCESS"
			file.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			status_v[STATUS] = status_v[STATUS] +1
		else:
			STATUS="DELIVERED"
			status_v[STATUS] = status_v[STATUS] +1
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			file.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round(consumption_go+consumption_return,2))+","+str(round(distance,2))+","+str(round(time,2))+")")
	else:
		STATUS="FAIL"
		file.write("\n(*CONSUMPTION,*DISTANCE,*TIME)->("+str(round(consumption_go,2))+","+str(round(distance,2))+","+str(round(time,2))+")")
		file.write("\nFAIL")
		status_v[STATUS] = status_v[STATUS] +1
	file.write("\nMISSION STATUS: "+STATUS+" WIND CHANGES:"+str(indice_orario- start_orario)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	file.close()
	return status_v

def run_test_alg_on_g(M_adj_dist,M_index_to_name,M_name_to_coor,destination,prefixs,payload,status_v,indice_ora):
	N = len(M_adj_dist)
	M_adj_angle = np.zeros((N,N))
	M_adj_wind = np.zeros((N,N))
	M_adj_wind_gw = np.zeros((N,N))
	M_adj_energy = np.zeros((N,N))
	file = open("result/test_alg_on_g"+str(indice_ora)+".txt","a+")
	confronto_energetico = open("result/confronto_energetico_on_g"+str(indice_ora)+".txt","a+")
	stop_flag = False
	consumption_go = 0
	consumption_return= 0
	distance = 0
	egdes_must_delete=[]
	control_time = 0
	indice_orario = indice_ora
	start_orario = indice_ora
	M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,0)
	G = nx.Graph(M_adj_energy)
	G1 = compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,0,DRONE_SPEED)
	time = 0
	STATUS = ""
	file.write("Greedy Shortest Path Test:(source,destination,energy_budget,payload_weight,drone_speed)->("+M_index_to_name[SOURCE]+","+M_index_to_name[destination]+","+str(BUDGET_ENERGY)+","+str(payload)+","+str(DRONE_SPEED)+")\n")
	confronto_energetico.write("\nGDP:(source,destination)->("+M_index_to_name[SOURCE]+","+M_index_to_name[destination]+")\n")
	sp_cost = nx.shortest_path_length(G, source=SOURCE, target=destination, weight='weight') 
	sp_path = nx.shortest_path(G, source=SOURCE, target=destination, weight='weight') 
	cycle_cost = nx.shortest_path_length(G1, source=destination, target=SOURCE, weight='weight') + sp_cost 
	file.write("\tPROJECTION ENERGY CONSUMPTION SP(SOURCE,DESTINATION): "+str(round(sp_cost,2))+"Kj PROJECTION ENERGY CONSUMPTION CYCLE(SOURCE,DESTINATION): "+str(round(cycle_cost,2))+"Kj\n")
	file.write("STEP BY STEP PATH\n")
	file.write("(HEAD,TAIL,DISTANCE,TIME,CONSUMPTION,Wd,Ws)")
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
				time = time + round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				file.write("->("+ M_index_to_name[node] +","+ M_index_to_name[node_next] +","+ str(round(M_adj_dist[node][node_next],2)) + "," + str(round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)) + "," + str(round(G[node][node_next]['weight'],2)) + "," + str(M_adj_angle[node][node_next])+","+str(M_adj_wind[node][node_next])+")")
				egdes_must_delete.append((node,node_next))
				#print(egdes_must_delete)
				node = node_next
			else:
				stop_flag = True
		else:
			indice_orario = indice_orario + 1
			#print(indice_orario)
			#print(egdes_must_delete)
			M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,payload)
			G = nx.Graph(M_adj_energy)
			G = delete_edges(G,egdes_must_delete)
			neighbors = list(nx.neighbors(G,node))
			if neighbors != []:
				node_next = neighbors.pop(0)
				for n in neighbors:
					if G[node][n]['weight'] < G[node][node_next]['weight']:
						node_next = n
				consumption_go = consumption_go + G[node][node_next]['weight']
				time = time + round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)
				distance = distance + M_adj_dist[node][node_next]
				file.write("->("+ M_index_to_name[node] +","+ M_index_to_name[node_next] +","+ str(round(M_adj_dist[node][node_next],2)) + "," + str(round((M_adj_dist[node][node_next]*1000/DRONE_SPEED)/60,2)) + "," + str(round(G[node][node_next]['weight'],2)) + "," + str(M_adj_angle[node][node_next])+","+str(M_adj_wind[node][node_next])+")")
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
					time = time + round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					file.write("->("+M_index_to_name[node_r]+","+M_index_to_name[node_r_next]+","+str(round(M_adj_dist[node_r][node_r_next],2))+","+str(round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2))+","+str(round(G1[node_r][node_r_next]['weight'],2))+","+str(M_adj_angle[node_r][node_r_next])+","+str(M_adj_wind[node_r][node_r_next])+")")
					egdes_must_delete.append((node_r,node_r_next))
					node_r = node_r_next
				else:
					stop_flag = True
			else:
				indice_orario = indice_orario + 1
				#print(indice_orario)
				M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_dist,M_index_to_name,M_name_to_coor,indice_orario,prefixs,0)
				G1 = nx.Graph(M_adj_energy)
				G1 = delete_edges(G1,egdes_must_delete)
				neighbors = list(nx.neighbors(G1,node_r))
				if neighbors != []:
					node_r_next = neighbors.pop(0)
					for n in neighbors:
						if G1[node_r][n]['weight'] < G1[node_r][node_r_next]['weight']:
							node_r_next = n
					consumption_go = consumption_go + G1[node_r][node_r_next]['weight']
					time = time + round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2)
					distance = distance + M_adj_dist[node_r][node_r_next]
					file.write("->("+M_index_to_name[node_r]+","+M_index_to_name[node_r_next]+","+str(round(M_adj_dist[node_r][node_r_next],2))+","+str(round((M_adj_dist[node_r][node_r_next]*1000/DRONE_SPEED)/60,2))+","+str(round(G1[node_r][node_r_next]['weight'],2))+","+str(M_adj_angle[node_r][node_r_next])+","+str(M_adj_wind[node_r][node_r_next])+")")
					egdes_must_delete.append((node_r,node_r_next))
					node_r = node_r_next
				else:
					stop_flag = True
			#print(M_adj_angle[node_r][node_r_next],M_adj_wind_gw[node_r][node_r_next],M_adj_angle[node_r_next][node_r],M_adj_wind_gw[node_r_next][node_r])
		if (consumption_go+consumption_return) <= BUDGET_ENERGY:
			STATUS="SUCCESS"
			file.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")")
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			status_v[STATUS] = status_v[STATUS] +1
		else:
			STATUS="DELIVERED"
			status_v[STATUS] = status_v[STATUS] +1
			confronto_energetico.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round((consumption_go+consumption_return),2))+","+str(round(distance,2))+","+str(round(time,2))+")\n")
			file.write("\n(CONSUMPTION,DISTANCE,TIME)->("+str(round(consumption_go+consumption_return,2))+","+str(round(distance,2))+","+str(round(time,2))+")")
	else:
		STATUS="FAIL"
		file.write("\n(*CONSUMPTION,*DISTANCE,*TIME)->("+str(round(consumption_go,2))+","+str(round(distance,2))+","+str(round(time,2))+")")
		status_v[STATUS] = status_v[STATUS] +1
	file.write("\nMISSION STATUS: "+STATUS+" WIND CHANGES:"+str(indice_orario - start_orario)+"\n\n")
	#print(STATUS)
	confronto_energetico.close()
	file.close()
	return status_v
def test_GDP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,indice_ora):
	tic = time.perf_counter()
	status_g = {"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	colors = Preprocessing_PP(SOURCE,Gminp, Gmin, Gmaxp, Gmax)
	loc = "result/test_alg_on_g"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/test_alg_on_g"+str(indice_ora)+".txt")
	loc = "result/confronto_energetico_on_g"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/confronto_energetico_on_g"+str(indice_ora)+".txt")
	print("Starting test_alg_on_g...")
	for destination in range(1,N):
		status_g = run_test_alg_on_g(M_adj_dist,M_index_to_name,M_name_to_coor,destination,prefixs,PAYLOAD,status_g,indice_ora)
	toc = time.perf_counter()
	print(f"test_alg_on_g complete in {toc - tic:0.4f} seconds.")
	file = open("result/test_alg_on_g"+str(indice_ora)+".txt","a+")
	file.write("\n\n"+str(status_g))

	return

def test_DSP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,indice_ora):
	tic = time.perf_counter()
	status_o = {"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	loc = "result/test_alg_on_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/test_alg_on_sp"+str(indice_ora)+".txt")
	loc = "result/confronto_energetico_on_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/confronto_energetico_on_sp"+str(indice_ora)+".txt")
	print("Starting test_alg_on_sp...")
	for destination in range(1,N):
		status_o = run_test_alg_on_sp(M_adj_dist,M_index_to_name,M_name_to_coor,destination,prefixs,PAYLOAD,status_o,indice_ora)
	toc = time.perf_counter()
	print(f"test_alg_on_sp complete in {toc - tic:0.4f} seconds.")
	file = open("result/test_alg_on_sp"+str(indice_ora)+".txt","a+")
	file.write("\n\n"+str(status_o))
	return

def test_OSP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,indice_ora):
	tic = time.perf_counter()
	status = {"CANCELED":0,"FAIL":0,"DELIVERED":0,"SUCCESS":0}
	loc = "result/test_alg_off_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/test_alg_off_sp"+str(indice_ora)+".txt")
	loc = "result/confronto_energetico_off_sp"+str(indice_ora)+".txt"
	if(os.path.isfile(loc)):
		os.remove("result/confronto_energetico_off_sp"+str(indice_ora)+".txt")
	print("Starting test_alg_off_sp...")
	for destination in range(1,N):
		status = run_test_alg_off_sp(M_adj_dist,M_index_to_name,M_name_to_coor,destination,prefixs,PAYLOAD,status,indice_ora)
	toc = time.perf_counter()
	print(f"test_alg_off_sp complete in {toc - tic:0.4f} seconds.")
	file = open("result/test_alg_off_sp"+str(indice_ora)+".txt","a+")
	file.write("\n\n"+str(status))
	return

def compute_energy_matrix(M_adj_dist,M_adj_angle,M_adj_wind,prefixs,payload,drone_speed):
	N = len(M_adj_dist)
	M_adj_energy = np.zeros((N,N))
	for i in range(12):
		for j in range(N):
			if M_adj_dist[i][j] != 0:
				M_adj_energy[i][j] = (M_adj_dist[i][j]*1000) * prefixs[(payload,drone_speed,M_adj_wind[i][j],M_adj_angle[i][j])]
				M_adj_energy[j][i] = (M_adj_dist[j][i]*1000) * prefixs[(payload,drone_speed,M_adj_wind[j][i],M_adj_angle[j][i])]
	return nx.Graph(M_adj_energy)

def n_random_test(n,M_adj_dist,M_adj_angle,M_adj_wind,prefixs):
	indici_orari_test = random.sample(range(2, 90), n)
	for i in indici_orari_test:
		test_GDP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,i)
		test_DSP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,i)
		test_OSP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,i)
	return
if __name__ == "__main__":
	df = pd.read_csv("city_coo_data.csv",sep=';', header=None)
	lon = df[2]
	lat = df[1]
	city = df[0]
	hour_wind_data = wind_data_load(city,INDICE_ORARIO)
	city_coord, coord_city = city_to_coor(lat,lon,city)
	latlon_index, index_latlon = city_to_dict(lat,lon)
	index_name, name_index = cityindex_to_dict(city)
	box = (lat.min()-1, lon.min()-1,lat.max()+1, lon.max()+1)
	m = smopy.Map(box, z=18)
	ax = m.show_mpl(figsize=(300, 150))
	for i in range(len(lon)):
		x,y = m.to_pixels(lat[i],lon[i])
		ax.plot(x, y, 'or', ms=5, mew=2)
		#ax.text(x-10, y-10, city[i].replace('_', ''), fontsize=6)
		"""if(len(city[i])>6):
			ax.text(x-10, y-10, city[i][:6].replace('_', ''), fontsize=6)
		else:
			ax.text(x-10, y-10, city[i].replace('_', ''), fontsize=6)
		"""
	"""
		lettura dai csv degli archi della triangolazione che utilizzo 
	"""
	dictmid_name_to_coor = {} #dizionario che ha come key: coppia dei nomi che individuano il midpoint e come value: le coordinate lat,lon
	dictmid_index_to_name = {} #dizionario che ha come key: indice della matrice in cui si individua il midpoint e come value: il nome
	DFcity = pd.read_csv("archi.csv",sep=';')
	DFcity.columns=['index','lat','lon','num']
	edges_lat =DFcity['lat'].tolist()
	edges_lon =DFcity['lon'].tolist()
	midpoint_name = DFcity['num'].tolist()

	"""
		lettura degli archi ottenuti dalla congiunzione dei centri di voronoi e i vertici
		della cella che utilizzo
	"""
	DFvor = pd.read_csv("vor.csv",sep=';')
	point = DFvor['coor'].tolist()
	name = DFvor['name'].tolist()
	dictvor_name_to_coor={} #dizionario che ha come key: coppia dei nomi che individuano il vertice di vor e come value: le coordinate lat,lon
	dictvor_index_to_name={} #dizionario che ha come key: indice della matrice in cui si individua il vertice di vor e come value: il nome

	N = len(lat) + len(edges_lat) + len(point)
	M_adj_dist = np.zeros((N,N)) #matrice che contiene le distanze in km
	M_adj_angle = np.zeros((N,N)) #matrice che contiene gli angoli relativi di vento in °
	M_adj_wind = np.zeros((N,N)) #matrice che contiene le forze dei venti in m/s
	M_adj_wind_gw = np.zeros((N,N)) #matrice che contiene l'angolo del vento globale in °
	for index in range(len(edges_lat)):
		#parsing delle coordinate delle colonnine collegate da archi di delauney
		x0,x1 = edges_lat[index].replace("[","").replace("]","").split(", ")
		x0 = float(x0)
		x1 = float(x1)
		y0,y1 = edges_lon[index].replace("[","").replace("]","").split(", ")
		y0 = float(y0)
		y1 = float(y1)
		#calcolo del punto di mezzo fra (x0,y0) e (x1,y1) considerando latitudine e longitudine
		x3,y3 = midpoint(float(x0),float(y0),float(x1),float(y1))
		dictmid_name_to_coor[midpoint_name[index]] = [x3,y3]
		dictmid_index_to_name[(index+len(lat))]=midpoint_name[index]
		dist = distance_coor_two_points(float(x0),float(y0),x3,y3)
		winfo = hour_wind_data[coord_city[(x0,y0)]]
		M_adj_dist[latlon_index[(x0,y0)]][index+len(lat)] = dist
		M_adj_dist[index+len(lat)][latlon_index[(x0,y0)]] = dist
		app = angleFromCoordinate(x0, y0, x3, y3)-winfo[2]
		if app < 0:
			M_adj_angle[latlon_index[(x0,y0)]][index+len(lat)] = app+360
		else:
			M_adj_angle[latlon_index[(x0,y0)]][index+len(lat)] = app
		app = angleFromCoordinate(x3, y3, x0, y0)-winfo[2]
		if app < 0:
			M_adj_angle[index+len(lat)][latlon_index[(x0,y0)]] = app+360
		else:
			M_adj_angle[index+len(lat)][latlon_index[(x0,y0)]] = app
		M_adj_wind_gw[latlon_index[(x0,y0)]][index+len(lat)] = winfo[2]
		M_adj_wind_gw[index+len(lat)][latlon_index[(x0,y0)]] = winfo[2]
		M_adj_wind[latlon_index[(x0,y0)]][index+len(lat)] = winfo[3]#il 5 è la raffica OCCHIO
		M_adj_wind[index+len(lat)][latlon_index[(x0,y0)]] = winfo[3]#il 5 è la raffica OCCHIO
		dist = distance_coor_two_points(x1,y1,x3,y3)
		winfo = hour_wind_data[coord_city[(x1,y1)]]
		M_adj_dist[latlon_index[(x1,y1)]][index+len(lat)] = dist
		M_adj_dist[index+len(lat)][latlon_index[(x1,y1)]] = dist
		app = angleFromCoordinate(x1, y1, x3, y3)-winfo[2]
		if app < 0:
			M_adj_angle[latlon_index[(x1,y1)]][index+len(lat)] = app +360
		else:
			M_adj_angle[latlon_index[(x1,y1)]][index+len(lat)] = app
		app = angleFromCoordinate(x3, y3, x1, y1)-winfo[2]
		if app < 0:
			M_adj_angle[index+len(lat)][latlon_index[(x1,y1)]] = app + 360
		else:
			M_adj_angle[index+len(lat)][latlon_index[(x1,y1)]] = app
		M_adj_wind_gw[latlon_index[(x0,y0)]][index+len(lat)] = winfo[2]
		M_adj_wind_gw[index+len(lat)][latlon_index[(x0,y0)]] = winfo[2]
		M_adj_angle[latlon_index[(x1,y1)]][index+len(lat)] = winfo[3] #il 5 è la raffica OCCHIO
		M_adj_angle[index+len(lat)][latlon_index[(x1,y1)]] = winfo[3]#il 5 è la raffica OCCHIO

		x0,y0 = m.to_pixels(float(x0),float(y0))
		x1,y1 = m.to_pixels(float(x1),float(y1))
		x3,y3 = m.to_pixels(float(x3),float(y3))
		ax.text((x0+x3)/2 -1 , (y0+y3)/2-1, str(round(dist,0))+" Km" , fontsize=6)
		ax.text((x1+x3)/2 -1 , (y1+y3)/2-1, str(round(dist,0))+" Km" , fontsize=6)
		ax.plot([x0,x1], [y0,y1], 'green', linewidth=1.5)
		ax.plot(x3, y3, 'bo', ms=5, mew=2)

	for index in range(len(point)):
		#parsing del vertice delle celle di voronoi utilizzato come appoggio
		x0,y0 = point[index].replace("[","").replace("]","").split(", ")
		x0 = float(x0)
		y0 = float(y0)
		px0,py0 = m.to_pixels(float(x0),float(y0))
		dictvor_name_to_coor[name[index]] = (x0,y0)
		dictvor_index_to_name[index + 30] = name[index]
		#parsing dei nomi delle colonnine collegate a tale punto
		name1,name2 = name[index].replace("(","").replace(")","").split(",")
		x1,y1 = city_coord[name1]
		dist2 = round(distance_coor_two_points(x0,y0,x1,y1),0)
		dist = distance_coor_two_points(x0,y0,x1,y1)
		winfo = hour_wind_data[name1]
		M_adj_dist[index + 30][latlon_index[(x1,y1)]] = dist
		M_adj_dist[latlon_index[(x1,y1)]][index + 30] = dist
		app = angleFromCoordinate(x0, y0, x1, y1) - winfo[2]
		if app < 0:
			M_adj_angle[index + 30][latlon_index[(x1,y1)]] = app + 360
		else:
			M_adj_angle[index + 30][latlon_index[(x1,y1)]] = app
		app = angleFromCoordinate(x1, y1, x0, y0) - winfo[2]
		if app < 0:
			M_adj_angle[latlon_index[(x1,y1)]][index + 30] = app + 360
		else:
			M_adj_angle[latlon_index[(x1,y1)]][index + 30] = app
		M_adj_wind_gw[latlon_index[(x1,y1)]][index + 30] = winfo[2]
		M_adj_wind_gw[index + 30][latlon_index[(x1,y1)]] = winfo[2]
		M_adj_wind[index + 30][latlon_index[(x1,y1)]] = winfo[3]#il 5 è la raffica OCCHIO
		M_adj_wind[latlon_index[(x1,y1)]][index + 30] = winfo[3]#il 5 è la raffica OCCHIO
		x1,y1 = m.to_pixels(x1,y1)
		x2,y2 = city_coord[name2]
		winfo = hour_wind_data[name2]
		dist1 = distance_coor_two_points(x0,y0,x2,y2)
		M_adj_dist[index + 30][latlon_index[(x2,y2)]] = dist1
		M_adj_dist[latlon_index[(x2,y2)]][index + 30] = dist1
		app = angleFromCoordinate(x0, y0, x2, y2)-winfo[2]
		if app < 0:
			M_adj_angle[index + 30][latlon_index[(x2,y2)]] = app + 360
		else:
			M_adj_angle[index + 30][latlon_index[(x2,y2)]] = app
		app = angleFromCoordinate(x2, y2, x0, y0) -winfo[2]
		if app < 0:
			M_adj_angle[latlon_index[(x2,y2)]][index + 30] = app + 360
		else:
			M_adj_angle[latlon_index[(x2,y2)]][index + 30] = app
		M_adj_wind_gw[index + 30][latlon_index[(x2,y2)]] = winfo[2]
		M_adj_wind_gw[latlon_index[(x2,y2)]][index + 30] = winfo[2]
		M_adj_wind[index + 30][latlon_index[(x2,y2)]] = winfo[3]#il 5 è la raffica OCCHIO
		M_adj_wind[latlon_index[(x2,y2)]][index + 30] = winfo[3]#il 5 è la raffica OCCHIO
		dist1 = round(distance_coor_two_points(x0,y0,x2,y2),0)
		x2,y2 = m.to_pixels(x2,y2)
		ax.text((px0+x2)/2 -1 , (py0+y2)/2-1, str(dist1)+" Km" , fontsize=6)
		ax.text((px0+x1)/2 -1 , (py0+y1)/2-1, str(dist2)+" Km" , fontsize=6)
		ax.plot([px0,x1], [py0,y1], 'orange', linewidth=1.5)
		ax.plot([px0,x2], [py0,y2], 'orange', linewidth=1.5)
	
	points=np.c_[lat, lon]
	M_name_to_coor, M_coor_to_name = master_dictionary(city_coord,dictmid_name_to_coor,dictvor_name_to_coor)
	M_index_to_name, M_name_to_index = master_dictionary(index_name,dictmid_index_to_name,dictvor_index_to_name)
	M_adj_angle = aprox_angle(M_adj_angle)
	M_adj_wind = approx_wind(M_adj_wind,M_adj_dist)
	M_adj_wind_gw = aprox_angle(M_adj_wind_gw)
	prefixs = compute_prefixes()
	print_table1(prefixs)
	print_table2(M_adj_dist,M_adj_angle,M_adj_wind,M_adj_wind_gw,M_index_to_name)
	"""for i in range(N):
		for j in range(N):
			if M_adj_dist[i][j] != 0:
				print(M_index_to_name[i]," TO ",M_index_to_name[j]," GLOBAL DIRECTION ",M_adj_wind_gw[i][j]," EDGE ANGLE ",M_adj_angle[i][j] + M_adj_wind_gw[i][j]," RELATIVE ANGLE ",M_adj_angle[i][j]," WIND SPEED ",M_adj_wind[i][j]," CONSUMPTION ",M_adj_energy[i][j])
	"""
	#M_adj_angle,M_adj_wind_gw,M_adj_wind,M_adj_energy = update_wind_graph(M_adj_angle,M_adj_dist,M_adj_wind,M_adj_wind_gw,M_adj_energy,M_index_to_name,M_name_to_coor, indice_orario_venti, prefixs,10,0)	
	
	"""
		SEZIONE DI TEST
	"""
	colors = []
	M_adj_energy = np.zeros((N,N))
	for i in range(12):
		for j in range(N):
			if M_adj_dist[i][j] != 0:
				M_adj_energy[i][j] = (M_adj_dist[i][j]*1000) * prefixs[(PAYLOAD,DRONE_SPEED,M_adj_wind[i][j],M_adj_angle[i][j])]
				M_adj_energy[j][i] = (M_adj_dist[j][i]*1000) * prefixs[(PAYLOAD,DRONE_SPEED,M_adj_wind[j][i],M_adj_angle[j][i])]
	Gminp, Gmin, Gmaxp, Gmax = min_max_energy_matrix(M_adj_dist,prefixs,DRONE_SPEED,PAYLOAD)
	G = nx.Graph(M_adj_energy)
	#for edges in nx.edges(Gminp):
	#	a , b = edges
	#	print(M_index_to_name[a]," TO ",M_index_to_name[b]," WEIGHT ",G[a][b]['weight']," distance ",M_adj_dist[a][b]," ",M_adj_dist[b][a])
	#indice_ora = 50
	n = 1
	#test_GDP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,indice_ora)
	#test_DSP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,indice_ora)
	#test_OSP(M_adj_dist,M_index_to_name,M_name_to_coor,prefixs,indice_ora)
	#n_random_test(n,M_adj_dist,M_adj_angle,M_adj_wind,prefixs)

	

	vor = spatial.Voronoi(points)
	regions, vertices = voronoi_finite_polygons_2d(vor)
	cells = [m.to_pixels(vertices[region]) for region in regions]
	ax.add_collection(mpl.collections.PolyCollection(cells,edgecolors='black',linewidths=(1.3,),alpha=.35))
	plt.show()
	