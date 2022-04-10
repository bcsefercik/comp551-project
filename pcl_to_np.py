import open3d as o3d
import numpy as np

pcd_path = '/home/onurberk/Desktop/development/comp551-project/_gitignore/pcd_files/001_1.pcd'
pcd_obj = o3d.io.read_point_cloud(pcd_path)
#o3d.visualization.draw_geometries([pcd_obj])

print(np.asarray(pcd_obj.points))
print(np.asarray(pcd_obj.colors))


#http://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.PointCloud.html

"""

colors
    RGB colors of points.

    Type
    float64 array of shape (num_points, 3), range [0, 1] , use numpy.asarray() to access data

points
    Points coordinates.

    Type
    float64 array of shape (num_points, 3), use numpy.asarray() to access data
"""
