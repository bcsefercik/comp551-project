#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    annotate_p3.py: Data class and Annotator class for labeling data, storing, surface reconstruction, and filtering.


"""
import sys

assert sys.version_info[0] >= 3

import os
import open3d as o3d
import numpy as np
import pickle as pkl
import copy
import logging as log
import sklearn.preprocessing as preprocessing
import pickle
from os import listdir
from os.path import isfile, isdir, join

from numpy.random import default_rng

import ipdb


def get_roi_mask(
    points,
    initial_mask=None,
    min_x=-500,
    max_x=500,
    min_y=-500,
    max_y=500,
    min_z=-500,
    max_z=500,
    offset=0.0,
):
    max_x += offset
    max_y += offset
    max_z += offset
    min_x -= offset
    min_y -= offset
    min_z -= offset

    roi_mask = points[:, 0] > -500 if initial_mask is None else initial_mask

    roi_mask = np.logical_and(points[:, 0] < max_x, roi_mask)  # x
    roi_mask = np.logical_and(points[:, 0] > min_x, roi_mask)
    roi_mask = np.logical_and(points[:, 1] < max_y, roi_mask)  # y
    roi_mask = np.logical_and(points[:, 1] > min_y, roi_mask)
    roi_mask = np.logical_and(points[:, 2] < max_z, roi_mask)  # z
    roi_mask = np.logical_and(points[:, 2] > min_z, roi_mask)

    return roi_mask


# LIMITS = {"min_x": -0.6, "max_x": 0.4, "max_z": 1.1, "min_y": -0.5}  # p2
# LIMITS = {"min_x": -0.5, "max_x": 0.3, "max_z": 1.3, "min_y": -0.5}  # p3
LIMITS = {
    "min_x": -0.6,
    "max_x": 0.43,
    "max_y": 0.3,
    "min_z": 0,
    "max_z": 1.25,
}  # test_p1


def filter_mesh(mesh, iterations, visualize=False, verbose=False):
    """ """
    mesh_out = mesh.filter_smooth_simple(number_of_iterations=iterations)
    mesh_out.compute_vertex_normals()
    if visualize:
        o3d.visualization.draw_geometries([mesh_out])

    return mesh_out


class Data:
    """
    A helper class to keep information about a sample of data and annotations provided by the annotator class.
    """

    def __init__(self, pcd_path):
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        self.arm_ind = None
        self.bg_ind = None
        self.arm_pcd = None
        self.arm_rec_pcd = None
        self.label_mask = None
        self.pose = None
        self.robot2ee_pose = None
        self.joint_angles = None

    def convert_to_pointgroup(self):
        """
        Converts the current data to pointgroup data
        """
        xyz = np.asarray(self.pcd.points)
        rgb = np.asarray(self.pcd.colors)
        # rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(-1, 1), axis=0)
        # rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(-1, 1), axis=0)
        # rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(-1, 1), axis=0)
        label = np.zeros(len(self.pcd.points))
        label[self.arm_ind] = 1
        instance_label = np.zeros(len(self.pcd.points))
        return [xyz, rgb, label, instance_label]

    def convert_to_robotNet(self, version="v2"):

        """
        Converts the current data to pointgroup data
        """
        xyz = np.asarray(self.pcd.points)
        rgb = np.asarray(self.pcd.colors)

        # rgb[:,0] = preprocessing.minmax_scale(rgb[:,0], feature_range=(-1, 1), axis = 0)
        # rgb[:,1] = preprocessing.minmax_scale(rgb[:,1], feature_range=(-1, 1), axis = 0)
        # rgb[:,2] = preprocessing.minmax_scale(rgb[:,2], feature_range=(-1, 1), axis = 0)

        label = np.zeros(len(self.pcd.points))
        label[self.arm_ind] = 1
        instance_label = np.zeros(len(self.pcd.points))
        pose = self.pose

        if version == "v2":
            return {
                "points": xyz,
                "rgb": rgb,
                "labels": label,
                "instance_labels": instance_label,
                "pose": pose,
                "robot2ee_pose": self.robot2ee_pose,
                "joint_angles": self.joint_angles,
            }
        else:
            return [xyz, rgb, label, instance_label, pose]

    def surface_reconstruct_arm_pcd(
        self,
        visualize=False,
        verbose=False,
        sampling_method="poisson_disk",
        filter_it=0,
    ):
        """ """

        if verbose:
            log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        else:
            log.basicConfig(format="%(levelname)s: %(message)s")

        log.info("Compute the normal of the point cloud.")

        self.arm_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10)
        )

        log.info("run Poisson surface reconstruction.")

        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            (
                arm_mesh,
                densities,
            ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                self.arm_pcd, depth=10
            )

        if visualize:
            log.info("visualizing mesh.")
            o3d.visualization.draw_geometries([arm_mesh])

        densities = np.asarray(densities)

        if visualize:
            from matplotlib import pyplot as plt

            log.info("visualizing estimated densities.")
            density_colors = plt.get_cmap("plasma")(
                (densities - densities.min()) / (densities.max() - densities.min())
            )
            density_colors = density_colors[:, :3]
            density_mesh = o3d.geometry.TriangleMesh()
            density_mesh.vertices = arm_mesh.vertices
            density_mesh.triangles = arm_mesh.triangles
            density_mesh.triangle_normals = arm_mesh.triangle_normals
            density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
            o3d.visualization.draw_geometries([density_mesh])

        log.info("removing low density vertices.")
        vertices_to_remove = densities < np.quantile(densities, 0.08)
        arm_mesh.remove_vertices_by_mask(vertices_to_remove)

        if visualize:
            log.info("visualizing mesh after removing low dencity verticies.")
            o3d.visualization.draw_geometries([arm_mesh])

        arm_mesh.compute_vertex_normals()

        if filter_it != 0:
            arm_mesh = filter_mesh(arm_mesh, filter_it)

        if sampling_method == "poisson_disk":
            self.arm_rec_pcd = arm_mesh.sample_points_poisson_disk(
                number_of_points=200000
            )
        elif sampling_method == "uniform":
            self.arm_rec_pcd = arm_mesh.sample_points_uniformly(number_of_points=200000)
        else:
            log.error("Undefined sampling method")
            exit

        if visualize:
            log.info("visualizing the sampled point cloud.")
            o3d.visualization.draw_geometries([arm_mesh])

    def update_arm_pcl_from_ind(self):
        """ """
        self.arm_pcd = self.pcd.select_by_index(self.arm_ind)

        # Selecting background
        # self.bg_pcd.select_by_index(arm_ind, invert=True)

    def remove_outliers(self, mode="statistical"):
        """ """
        if self.arm_pcd != None:
            log.error(
                "This operation changes the main pcd file. You cannot perform this after annotation."
            )
            exit

        if mode == "statistical":
            cl, ind = self.pcd.remove_statistical_outlier(
                nb_neighbors=40, std_ratio=2.0
            )

        elif mode == "radious":
            cl, ind = self.pcd.remove_radius_outlier(nb_points=20, radius=0.01)
            self.pcd = self.pcd.select_by_index(ind)

        else:
            log.error("Undefined sampling method")
            exit

        self.pcd = self.pcd.select_by_index(ind)

    def write_rec_arm_to_pcd(self, path="rec_arm_cloud.pcd"):
        """ """
        o3d.io.write_point_cloud(
            path,
            self.arm_rec_pcd,
            write_ascii=False,
            compressed=False,
            print_progress=True,
        )

    def write_arm_to_pcd(self, path="arm_cloud.pcd"):
        """ """
        o3d.io.write_point_cloud(
            path, self.arm_pcd, write_ascii=False, compressed=False, print_progress=True
        )

    def write_pointgroup_element(self, data, path):
        """
        Please refer to the pointgroup/data/alivev1_inst.py
        """

        data = self.convert_to_pointgroup()
        pickle.dump(data, open(path + ".pickle", "wb"))
        del data

    def write_robotNet_element(self, data, path):
        """
        Please refer to the robotnet/alive/data/alivev1_inst.py
        """

        data = self.convert_to_robotNet()
        pickle.dump(data, open(path + ".pickle", "wb"))
        del data

    def write_labels_np(self, path):
        """ """
        assert self.arm_ind.any() != None
        assert self.bg_ind.any() != None
        assert self.label_mask.any() != None

        dic = {}
        dic["arm_ind"] = self.arm_ind
        dic["bg_ind"] = self.bg_ind
        dic["label_mask"] = self.label_mask

        with open(path + ".pkl", "wb") as f:
            pkl.dump(dic, f, pkl.HIGHEST_PROTOCOL)


class Annotator:
    """
    All the operations of annotaion is going to be integrated
    to this class (e.g. background aggregation, background subtraction, reconstruction, ...)
    """

    def __init__(self, load_bg_from_file=None):

        if load_bg_from_file is None:
            log.warning(
                "Don't forget to set the background pcl with set_background later."
            )
        else:
            # ipdb.set_trace()
            self.bg_pcl = o3d.io.read_point_cloud(load_bg_from_file)

    def set_background(self, bg_data):
        """
        set background samples of the annotator

        Note: some preprocessing steps on bg samples can be added here (e.g. adding aggregating frames)
        """
        self.bg_samples = bg_data[0]

    def pose_to_arr(self, pose):
        return np.array(
            [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ]
        )

    def annotate_batch(
        self,
        folder_name=None,
        output_dir=None,
        percentages=[0.6, 0.2, 0.2],
        bg_pcl=None,
        conversion_type="pointgroup",
        pose_data=None,
        write_pcd=False,
    ):

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if bg_pcl is not None:
            self.bg_pcl = bg_pcl

        files = [
            f
            for f in listdir(folder_name)
            if isfile(join(folder_name, f)) and f[-4:] == ".pcd"
        ]
        n_sample = len(files)

        # assert n_sample == pose_data.shape[0]

        samples = np.random.choice(3, n_sample, p=percentages)
        int2str = ["train", "val", "test"]
        for i in range(n_sample):

            data = Data(folder_name + files[i])
            # ipdb.set_trace()
            data.arm_ind, data.bg_ind, data.label_mask = self.distance_annotate(
                data.pcd, self.bg_pcl, removal_th=0.02, clip_depth=True, max_depth=1.0
            )

            print("Sample no: ", i, " ", len(data.arm_ind))

            if len(data.arm_ind) < 256:
                print("|>No arm data")
                continue
            single_pose_data = np.load(
                folder_name + files[i][:-4] + ".npy", allow_pickle=True
            )
            # data.pose = pose_data[i]
            data.pose = single_pose_data
            if os.path.isfile(folder_name + files[i][:-4] + "_joint_angles.npy"):
                joint_angles = np.load(
                    folder_name + files[i][:-4] + "_joint_angles.npy", allow_pickle=True
                )
                data.joint_angles = joint_angles

            if os.path.isfile(folder_name + files[i][:-4] + "_robot2ee_pose.npy"):
                data.robot2ee_pose  = np.load(
                    folder_name + files[i][:-4] + "_robot2ee_pose.npy", allow_pickle=True
                )

            # data.pose = self.pose_to_arr(data.pose)
            data.update_arm_pcl_from_ind()

            output = output_dir
            # if not os.path.isdir(output):
            #     os.mkdir(output)
            output += files[i][:-4]

            if write_pcd:
                output_pcd = output_dir + "arm_pcl/"
                if not os.path.isdir(output_pcd):
                    os.mkdir(output_pcd)
                output_pcd += files[i]
                data.write_arm_to_pcd(path=output_pcd)

            if conversion_type == "pointgroup":
                data.write_pointgroup_element(data, output)
            elif conversion_type == "robotNet":
                data.write_robotNet_element(data, output)
            del data

    def annotate_single(self, target_data, bg_pcl=None):
        """
        Args:
            param1 (Data)  : The data sample to annotate.
        """
        if bg_pcl is None:
            bg_pcl = self.bg_pcl

        (
            target_data.arm_ind,
            target_data.bg_ind,
            target_data.label_mask,
        ) = self.distance_annotate(
            target_data.pcd, bg_pcl, removal_th=0.02, clip_depth=True, max_depth=1.0
        )
        target_data.update_arm_pcl_from_ind()

        return target_data

    def distance_annotate(
        self, target_cloud, bg_pcl, removal_th=0.02, clip_depth=True, max_depth=1
    ):
        """
        Classifies 3D points into arm and background points
        Args:
            param1 (Pointcloud)      : The background Pointcloud.
            param2 (Pointcloud)      : The target Pointcloud set that is being annotated.

        Returns:
            nparray: returns array of integer containing indices of the robot arm points.
        """
        dists = np.asarray(target_cloud.compute_point_cloud_distance(bg_pcl))
        roi_mask = dists > removal_th
        # ipdb.set_trace()

        if clip_depth:

            points = np.asarray(target_cloud.points)
            roi_mask = get_roi_mask(points, initial_mask=roi_mask, **LIMITS)

        return np.where(roi_mask == True)[0], np.where(roi_mask == False)[0], roi_mask


if __name__ == "__main__":

    # set some parameters
    visualize = True
    remove_outliers = False
    reconstruct = False
    verbose = True

    common_path = sys.argv[1] + "/"

    # common_path = "/home/bcs/Desktop/MSc/repos/comp551_project/dataset/new/p1/half_light/"

    file_dir = common_path + "pcd_ee/"

    # file_dir += "99.pdc"

    isDirectory = isdir(file_dir)

    # poses = np.load(common_path + 'ee_poses.npy', allow_pickle = True)
    poses = None

    if isDirectory:
        ann = Annotator(load_bg_from_file=common_path + "combined_bg.pcd")
        output_dir = common_path + "labeled/"
        ann.annotate_batch(
            folder_name=file_dir,
            output_dir=output_dir,
            percentages=[1, 0, 0],
            conversion_type="robotNet",
            pose_data=poses,
            write_pcd=False,
        )
        log.info("Visualizing are done")
        sys.exit()

    target_data = Data(file_dir)

    log.info("Loading and annotating data.")

    ann = Annotator(load_bg_from_file=common_path + "combined_bg.pcd")
    target_data = ann.annotate_single(target_data)

    if remove_outliers:
        target_data.remove_outliers(mode="statistical")

    if visualize:
        log.info("visualizing the arm points.")
        o3d.visualization.draw_geometries([target_data.arm_pcd])
        o3d.visualization.draw_geometries([target_data.pcd])

        if reconstruct:
            log.info("visualizing the reconstructed points.")
            o3d.visualization.draw_geometries([bg_cloud])

    log.info("Saving the arm PCD.")
    target_data.write_arm_to_pcd("arm_cloud.pcd")

    if reconstruct:

        log.info("Reconstructing the arm points.")
        target_data.surface_reconstruct_arm_pcd(
            visualize=False, verbose=verbose, sampling_method="uniform", filter_it=0
        )

        log.info("Saving the arm PCD.")
        target_data.write_rec_arm_to_pcd("rec_arm_cloud.pcd")

    target_data.write_labels_np("label_001")
