import numpy as np
import pickle as pkl
import copy
import logging as log
import sklearn.preprocessing as preprocessing
import pickle
from os import listdir
from os.path import isfile,isdir, join

import os
import argparse

import open3d as o3d
import ipdb


def visualize_res(arm_xyz, arm_rgb, ee_position, ee_orientation,  visualize=True, save=False):
        '''
            This function visualizes the pcd and ee pose or saves them.
        '''
        arm_pcd        = o3d.geometry.PointCloud()
        arm_pcd.points = o3d.utility.Vector3dVector(arm_xyz)
        arm_pcd.colors = o3d.utility.Vector3dVector(arm_rgb)



        frame    = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        ee_frame = copy.deepcopy(frame).translate(ee_position)
        ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

        o3d.visualization.draw_geometries([arm_pcd, ee_frame])
        o3d.visualization.draw_geometries([arm_pcd, ee_frame])


arm_file_path = '/home/bcs/Desktop/MSc/repos/comp551_project/dataset/toy/train/moving_8.pickle'


filehandler = open(arm_file_path, 'rb')

xyz_origin, rgb, label, instance_label, pose = pickle.load(filehandler, encoding='bytes')

filehandler.close()

arm_points     = xyz_origin[label==1]
rgb            = rgb[label==1]
rgb[:,0]       = preprocessing.minmax_scale(rgb[:,0], feature_range = (0,1), axis = 0)
rgb[:,1]       = preprocessing.minmax_scale(rgb[:,1], feature_range = (0,1), axis = 0)
rgb[:,2]       = preprocessing.minmax_scale(rgb[:,2], feature_range = (0,1), axis = 0)

visualize_res(arm_points, rgb, (pose[0], pose[1], pose[2]), (pose[3], pose[4], pose[5], pose[6]))
#0.50330381, -0.27686713,  0.00213259,  0.35378082,  0.82501708,0.26996977,  0.3482847


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcl', type=str)
    args = parser.parse_args()

    if os.path.isdir(args.pcl):
        file_names = os.listdir(args.pcl)
        files = [os.path.join(args.pcl, fn) for fn in file_names if fn[-7:] == ".picke"]
    else:
        files = [args.pcl]

    ipdb.set_trace()
