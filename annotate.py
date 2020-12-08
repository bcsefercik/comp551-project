import open3d as o3d
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
import pickle as pkl
import time

import sensor_msgs.point_cloud2 as pc2
import rospy
import pickle

from sensor_msgs.msg import PointCloud2, PointField




class Annotator:

    def __init__(self, ):
        self.metrics     = None
        self.bg_samples  = None

    def set_bg(self, samples):
        '''
            set bg samples of the annotator
                
            Note: some preprocessing steps on bg samples can be added here (e.g. adding aggregating frames)
        '''
        self.bg_samples = samples

    def o3d_classify(self, bg_cloud, target_cloud, metric='distance_changed', removal_th=0.01):
        '''
            Classifies 3D points into arm and background points
        '''
        dists = target_cloud.compute_point_cloud_distance(bg_cloud)

        dists = np.asarray(dists)

        mask = dists > removal_th

        ind = np.where(mask == True)[0]

        return ind


visualize = False

pcd_path  = './_gitignore/pcd_files/bg_data/bg_1.pcd'
pcd_obj_p = o3d.io.read_point_cloud(pcd_path)

pcd_path  = './_gitignore/pcd_files/data_samples/data_30.pcd'
pcd_obj_c = o3d.io.read_point_cloud(pcd_path)

ann       = Annotator()

start = time.time()
arm_ind  = ann.o3d_classify(pcd_obj_p ,pcd_obj_c)
end = time.time()
print("elapsed time: ", end - start)

arm_cloud = pcd_obj_c.select_down_sample(arm_ind)
bg_cloud  = pcd_obj_c.select_down_sample(arm_ind, invert=True)

o3d.io.write_point_cloud("arm_cloud.pcd", arm_cloud, write_ascii=False, compressed=False, print_progress=True)
o3d.io.write_point_cloud("bg_cloud.pcd", bg_cloud, write_ascii=False, compressed=False, print_progress=True)

if visualize:
    print("visualizing the arm points.")
    o3d.visualization.draw_geometries([arm_cloud])

    print("visualizing the background points.")
    o3d.visualization.draw_geometries([bg_cloud])
    raw_input()


