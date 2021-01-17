#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    annotate_p3.py: Data class and Annotator class for labeling data, storing, surface reconstruction, and filtering.

    
'''
import sys
assert sys.version_info[0] >= 3


import open3d as o3d
import numpy as np
import pickle as pkl
import copy
import logging as log
import sklearn.preprocessing as preprocessing
import pickle
from os import listdir
from os.path import isfile,isdir, join
import trimesh

from numpy.random import default_rng


def visualize_res(arm_xyz, arm_rgb, ee_position, ee_orientation,  visualize=True, save=False):
        '''
            This function visualizes the pcd and ee pose or saves them.
        '''
        arm_pcd        = o3d.geometry.PointCloud()
        arm_pcd.points = o3d.utility.Vector3dVector(arm_xyz)
        arm_pcd.colors = o3d.utility.Vector3dVector(arm_rgb)
        
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        ee_frame = copy.deepcopy(frame).translate(ee_position)
        ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

        o3d.visualization.draw_geometries([arm_pcd, ee_frame])


arm_file_path = '/home/fnegahbani20/workplace/COMP551/comp551-project/_gitignore/pickle_samples/moving_67.pickle'


filehandler = open(arm_file_path, 'rb')

xyz_origin, rgb, label, instance_label, pose = pickle.load(filehandler, encoding='bytes')

filehandler.close()

arm_points     = xyz_origin[label==1] 
rgb            = rgb[label==1]
rgb[:,0]       = preprocessing.minmax_scale(rgb[:,0], feature_range = (0,1), axis = 0)
rgb[:,1]       = preprocessing.minmax_scale(rgb[:,1], feature_range = (0,1), axis = 0)
rgb[:,2]       = preprocessing.minmax_scale(rgb[:,2], feature_range = (0,1), axis = 0)

visualize_res(arm_points, rgb, (pose[0], pose[1], pose[2]), (pose[3], pose[4], pose[5], pose[6]))