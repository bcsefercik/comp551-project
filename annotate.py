import open3d as o3d
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances


class Annotator:

    def __init__(self):
        self.metrics = None

    def classify(self, points_p, points_c, metric='distance_changed', removal_th=0.015):
        '''
            Classifies 3D points into arm and background points
        '''

        # Needs too much memory -> is not feasible
        #dists = distance_matrix(points_c[i,:].reshape(1,-1), points_p)

        mask = np.ones(points_c.shape[0])

        for i in range(points_c.shape[0]):
        
            dists = euclidean_distances(points_c[i,:].reshape(1,-1), points_p)
            
            if dists.min() <= removal_th:
                mask[i] = 0
            
        #      arm points    , background points
        return points_c[mask], points_c[np.invert(mask)]


pcd_path  = '../data/001_1.pcd'
pcd_obj_p = o3d.io.read_point_cloud(pcd_path)
points_p  = np.asarray(pcd_obj_p.points)

pcd_path  = '../data/001_3.pcd'
pcd_obj_c = o3d.io.read_point_cloud(pcd_path)
points_c  = np.asarray(pcd_obj_c.points)

ann       = Annotator()

arm_points, bg_points = ann.classify(points_p,points_c)

