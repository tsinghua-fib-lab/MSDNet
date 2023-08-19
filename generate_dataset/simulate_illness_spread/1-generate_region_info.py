import shapefile
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from tqdm import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pickle
from utils import *

args = initialization()
Network = args.Network
shape_file = args.shp
region_file_dir = '../../data/{0}/region_file'.format(Network)
sf = shapefile.Reader("{0}/{1}.shp".format(region_file_dir, shape_file))
shapes = sf.shapes()
length_district = len(shapes)
print(length_district)
length_shapes = []
info = []

for i in range(length_district):
    length_shapes.append(len(shapes[i].points))
max_points = max(length_shapes)
points = np.zeros((length_district, max_points, 2))
for i in range(length_district):
    for j in range(len(shapes[i].points)):
        points[i, j] = shapes[i].points[j]

sio.savemat('{0}/streets_xa.mat'.format(region_file_dir), {'streets_xa': points})

Region_info = loadmat("{0}/streets_xa.mat".format(region_file_dir))  # load region information
Region_edge = np.array(Region_info['streets_xa'])  # let street in the region to be the edge
region_center = []
for region in tqdm(Region_edge):
    not_zero_points = []
    for region_points in region:
        if np.any(region_points):
            not_zero_points.append(region_points.tolist())
    np.array(not_zero_points)
    region_center.append(np.average(not_zero_points, axis=0))

sio.savemat('{0}/centers_xa.mat'.format(region_file_dir), {'centers_xa': region_center})

sf = shapefile.Reader("{0}/{1}.shp".format(region_file_dir, shape_file))

shapes = sf.shapes()

f = open('{0}/aois.pkl'.format(region_file_dir), 'rb')
content = pickle.load(f)
content = list(content)
content = np.array(content)

last_info = []
former_info = []
former_info = content[len(content) - 33326:len(content)]

print(former_info)
print("--------------------------------------------------------------------")
last_info = np.delete(content, range(len(content) - 33326, len(content)), 0)
print(last_info)

population = np.zeros(len(shapes))
polygons = []

for i in range(len(shapes)):  # for every large region
    # build big regions
    polygon = Polygon(shapes[i].points)
    polygons.append(polygon)

for i in trange(len(last_info)):  # for every Points in small regions
    # create a Point
    point = Point(last_info[i].get('geo'))
    external = last_info[i].get('external')
    flag = False
    # if (i % 1000 == 0):
    #     print(i)
    for j in range(len(shapes)):  # for every big regions
        if polygons[j].contains(point):  # if the Point belongs to the region
            # add population
            population[j] += external.get('population')
            flag = True
            break
    if not flag:
        print('no such region!')

for i in trange(len(former_info)):  # for every Points in small regions
    # create a Point
    geo = former_info[i].get('geo')
    geo = np.array(geo)
    average_geo = np.average(geo, axis=0)
    print(average_geo)
    point = Point(average_geo)
    external = former_info[i].get('external')
    # if (i % 1000 == 0):
    #     print(i)
    flag = False
    for j in range(len(shapes)):  # for every big regions
        if polygons[j].contains(point):  # if the Point belongs to the region
            # add population
            population[j] += external.get('population')
            flag = True
            break
    if not flag:
        print('no such region!')

population.reshape(-1, 1)
sio.savemat('{0}/population_xa.mat'.format(region_file_dir), {'population_xa': population})

print("finished!")
