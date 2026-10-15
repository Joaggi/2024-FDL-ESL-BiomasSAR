import matplotlib.pyplot as plt
import tifffile as tiff
from shapely.geometry import Point, Polygon
import xarray as xr
from matplotlib.patches import Polygon as MplPolygon
import numpy as np 

def crop_lidar_edge_zone(data):
    """
    Crop the edge of zero value areas in LiDAR CHM dataset.
    
    :param array: 2D numpy array
    :return: Cropped 2D numpy array
    """
    # Define the coordinates of the polygon (example polygon coordinates)
    polygon_coords = [(50, 40), (350, 0), (710, 0), (710, 250), (50, 300)]
    polygon = Polygon(polygon_coords)

    # Access the correct coordinate variables (assuming they are named 'x' and 'y')
    rows, cols = data.shape
    mask = np.zeros(data.shape, dtype=bool)

    # Generate grid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))

    # Check if each point is inside the polygon
    for i in range(rows):
        for j in range(cols):
            point = Point(x_coords[i, j], y_coords[i, j])
            if polygon.contains(point):
                mask[i, j] = True
    
    # Create a new array by extracting only the data in the polygon
    clipped_data = np.full(data.shape, np.nan)
    clipped_data[mask] = data[mask]

    return clipped_data