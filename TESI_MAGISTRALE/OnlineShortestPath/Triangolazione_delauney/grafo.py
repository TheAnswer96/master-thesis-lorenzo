import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import smopy
import time
import matplotlib.lines as mlines

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

def trasform_editable_delauney(delauney,points,mapp):
	res = []
	for tri in delauney.simplices:
		res.append(mapp.to_pixels(points[tri]))
	return res

if __name__ == "__main__":
	df = pd.read_csv("city_coo_data.csv",sep=';', header=None)
	lon = df[2]
	lat = df[1]
	city = df[0]
	box = (lat.min()-1, lon.min()-1,lat.max()+1, lon.max()+1)
	m = smopy.Map(box, z=18)
	ax = m.show_mpl(figsize=(300, 150))
	for i in range(len(lon)):
		x,y = m.to_pixels(lat[i],lon[i])
		ax.plot(x, y, 'or', ms=4, mew=2)
		"""if(len(city[i])>6):
			ax.text(x-10, y-10, city[i].replace('_', ' '), fontsize=10)
		else:
			ax.text(x-10, y-10, city[i].replace('_', ' '), fontsize=10)
		"""

	points=np.c_[lat, lon]
	vor = spatial.Voronoi(points)
	tri = spatial.Delaunay(points)
	conta = 0
	equa = set()
	archi_lat = []
	archi_lon = []
	archi_num = []
	for triangolo in tri.simplices:
		x0,y0 = m.to_pixels(lat[triangolo[0]],lon[triangolo[0]])
		x1,y1 = m.to_pixels(lat[triangolo[1]],lon[triangolo[1]])
		x2,y2 = m.to_pixels(lat[triangolo[2]],lon[triangolo[2]])
		if (x0,x1,y0,y1) not in equa and (x1,x0,y1,y0) not in equa :
			equa.add((x0,x1,y0,y1))
			archi_lat.append([lat[triangolo[0]],lat[triangolo[1]]])
			archi_lon.append([lon[triangolo[0]],lon[triangolo[1]]])
			archi_num.append(conta)
			#ax.plot([x0,x1], [y0,y1], 'green', linewidth=2)
			xmean = (x0+x1) / 2
			ymean = (y0+y1) / 2
			#ax.text(xmean+2, ymean+2, conta, fontsize=10)
			conta = conta +1
		if (x0,x2,y0,y2) not in equa and (x2,x0,y2,y0) not in equa :
			equa.add((x0,x2,y0,y2))
			archi_lat.append([lat[triangolo[0]],lat[triangolo[2]]])
			archi_lon.append([lon[triangolo[0]],lon[triangolo[2]]])
			archi_num.append(conta)
			#ax.plot([x0,x2], [y0,y2], 'green', linewidth=2)
			xmean = (x0+x2) / 2
			ymean = (y0+y2) / 2
			#ax.text(xmean+2, ymean+2, conta, fontsize=10)
			conta = conta +1
		if (x1,x2,y1,y2) not in equa and (x2,x1,y2,y1) not in equa :
			equa.add((x1,x2,y1,y2))
			archi_lat.append([lat[triangolo[1]],lat[triangolo[2]]])
			archi_lon.append([lon[triangolo[1]],lon[triangolo[2]]])
			archi_num.append(conta)
			#ax.plot([x1,x2], [y1,y2], 'green', linewidth=2)
			xmean = (x1+x2) / 2
			ymean = (y1+y2) / 2
			#ax.text(xmean+2, ymean+2, conta, fontsize=10)
			conta = conta +1
	triangulation = trasform_editable_delauney(tri,points,m)
	regions, vertices = voronoi_finite_polygons_2d(vor)
	cells = [m.to_pixels(vertices[region]) for region in regions]
	ax.add_collection(mpl.collections.PolyCollection(cells,edgecolors='black',linewidths=(1.3,),alpha=.35))
	#ax.add_collection(mpl.collections.PolyCollection(triangulation,edgecolors='orange',linewidths=(1.3,), facecolor=None,alpha=.45))
	#list_of_tuples = list(zip(archi_lat, archi_lon,archi_num))
	#edges = pd.DataFrame(list_of_tuples, columns=['lat','lon','num'])
	#edges.to_csv("archi.csv",sep=';')
	plt.show()