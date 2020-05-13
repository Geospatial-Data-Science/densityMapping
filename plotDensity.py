from pylab import *
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neighbors import KernelDensity
from matplotlib import path
import random
from mpl_toolkits.basemap import Basemap 
import numpy as np
import pandas as pd
from rasterio.transform import Affine
import rasterio
import os 

os.chdir('/home/mmann1123/Documents/github/densityMapping')

def pnpoly(x, y, xyverts):
    """
    Return 1 if x,y is inside the polygon, 0 otherwise.
    A point on the boundary may be treated as inside or outside.

    :param x: Longitude
    :type x: float 

    :param y: Latitude
    :type y: float 

    :param xyverts: a sequence of x,y vertices making up a polygon
    :type xyverts: float 
    """
    p = path.Path(xyverts)
    return p.contains_point([x, y])  

def points_inside_poly(xypoints, xyverts):
    """
    Returns a boolean ndarray, True for points inside the polygon.
    A point on the boundary may be treated as inside or outside.

    :param xypoints: a sequence of N x,y pairs.
    :type xypoints: np.c_[x, y] 

    :param xyverts: sequence of x,y vertices of the polygon.
    :type yxyverts: np.array([x,y],[x,y])
    """

    p = path.Path(xyverts)
    return p.contains_points(xypoints)


def points_in_polys(points, polys):
    """
    Iterates across multiple polygons to create a binary mask, returns True for points inside of polygons
    This method masks off the water (where data will be unreliable).

    :param points: a sequence of N x,y pairs.
    :type points: np.c_[x, y] 

    :param polys: sequence of x,y vertices of the polygon.
    :type polys: np.array([x,y],[x,y])

    """
    result = []
    for i, poly in enumerate(polys):
        if i == 0:
            mask = points_inside_poly(points, poly)
        else:
            mask = mask | points_inside_poly(points, poly)
    return np.array(mask)


 
def generateSimulatedData():
    """
    This generates some (not very realistic) simulated data.
    """
    lats = list(35 + (np.random.random([2000,]) - .5) * 10)
    longs = list(-100 + (np.random.random([1000,]) - .5) * 20) + list(-80 + (np.random.random([1000,]) - .5) * 20)
    is_pizza_shop = [1 for i in range(1000)] + [0 for i in range(1000)] 
    d = pd.DataFrame({'lat':lats, 'lon':longs, 'shop_is_pizza_shop':is_pizza_shop})
    d.to_csv('sample_data.csv', index = False)


def kernel_density_nearest_neighbor(filename, col_of_interest = None,  maskOffWater = True, 
                                    NN_neighbors = 20, res = .2, NN_weights = lambda x:np.exp(-(x ** 2) / 3 ** 2),
                                    KD_kernel='gaussian', KD_bandwidth = 3,
                                    min_lat = 0, max_lat = 90, min_lon = -130, max_lon = -65,):

    """
    Generates kernel density or nearest neighbor estimates from point data where lat and lon are provided.
    Defaults to generating kernel point density, if `col_of_interest` provided, creates nearest neighbor (NN) interpolation. 


    :param filename: csv file containing lat and lon of data
    :type filename: path

    :param col_of_interest: If not None, function uses this data column to generate NN interpolation
    :type col_of_interest: string

    :param maskOffWater: Indicates whether water should be masked from analysis
    :type maskOffWater: Boolean

    :param NN_neighbors: Number of neighbors to included in NN interpolation
    :type NN_neighbors: Boolean

    :param res: Resolution in degrees of the output raster
    :type res: float

    :param weights: Neighborhood weights specification for NN interpolation 
    :type weights: lambda function

    :param min_lat: Minumum lat of bounding box for analysis
    :type min_lat: Int

    :param max_lat: Maximum lat of bounding box for analysis
    :type max_lat: Int

    :param min_lon: Minumum lon of bounding box for analysis
    :type min_lon: Int

    :param max_lon: Maximum lon of bounding box for analysis
    :type max_lon: Int    
    """


    d = pd.read_csv(filename)
    if not (('lat' in d.columns) and ('lon' in d.columns)):
        raise Exception('Error: dataset must contain lat and lon')

    if (col_of_interest is not None) and (col_of_interest not in d.columns):
        raise Exception('Error: dataset must contain column labeled %s' % col_of_interest)
        if not set(d[col_of_interest]) == set([0, 1]):
            raise Exception("Error: column %s must contain 0s and 1s" % col_of_interest)

    #Filter for events with locations. 
    geolocated = d.dropna(subset = ['lat', 'lon']) 
    idxs = (geolocated['lat'] > min_lat) & (geolocated['lat'] < max_lat) 
    idxs = idxs &  (geolocated['lon'] > min_lon) & (geolocated['lon'] < max_lon) 
    geolocated = geolocated.loc[idxs]

    #Fit the appropriate model: k-nearest neighbors if col_of_interest is not None, otherwise Kernel Density Estimation. 
    if col_of_interest is not None:
        model = KNeighborsClassifier(n_neighbors = NN_neighbors, weights = NN_weights)
        print('Total number of points', len(geolocated), 'in column of interest nonzero', geolocated[col_of_interest].sum())
        model.fit(geolocated[['lat', 'lon']], geolocated[col_of_interest])   
    else:
        print('Total number of points', len(geolocated))
        model = KernelDensity(kernel=KD_kernel, bandwidth = KD_bandwidth).fit(geolocated[['lat', 'lon']])

    #Create a grid of points at which to predict. 
    x = np.arange(min_lat, max_lat, res)
    y = np.arange(min_lon, max_lon, res)
    X, Y = meshgrid(x, y)
    numel = len(X) * len(X[0, :])
    Z = np.zeros(X.shape)
    unraveled_x = X.reshape([numel, 1])
    unraveled_y = Y.reshape([numel, 1])
    data_to_eval = np.hstack([unraveled_x, unraveled_y])

        #Make predictions using appropriate model. 
    if col_of_interest is not None:
        density = model.predict_proba(data_to_eval)[:, 1]

    else:
        density = np.exp(model.score_samples(data_to_eval))


    # get map for coastline masking
    m = Basemap(llcrnrlat = min_lat, urcrnrlat = max_lat, llcrnrlon = min_lon, urcrnrlon=max_lon, resolution='l', fix_aspect = False)
    if maskOffWater:
        m.drawcoastlines()
        x, y = m(data_to_eval[:,1], data_to_eval[:,0])
        loc = np.c_[x, y]
        polys = [p.boundary for p in m.landpolygons]
        on_land = points_in_polys(loc, polys) 
        density[~on_land] = np.NaN

    # weird transform due to earlier reshaping
    density =   np.rot90(density.reshape( X.shape))[::-1,:] 

    # create raster
    transform = Affine.translation(x[0] - res / 2, y[0] - res / 2) * Affine.scale(res, res)

    with rasterio.open(
        './new.tif',
        'w',
        driver='GTiff',
        height=density.shape[0],
        width=density.shape[1],
        count=1,
        dtype=density.dtype,
        crs='+proj=latlong',
        transform=transform) as dst:
            dst.write(density, 1)

if __name__ == '__main__':
    generateSimulatedData()
    filename = 'sample_data.csv'
    kernel_density_nearest_neighbor(filename = 'sample_data.csv')
