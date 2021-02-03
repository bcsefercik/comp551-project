import copy
import pickle
import os
import argparse

import open3d as o3d
import sklearn.preprocessing as preprocessing


def visualize_res(arm_xyz, arm_rgb, ee_position, ee_orientation, save=False):
    '''
        This function visualizes the pcd and ee pose or saves them.
    '''
    arm_pcd = o3d.geometry.PointCloud()
    arm_pcd.points = o3d.utility.Vector3dVector(arm_xyz)
    arm_pcd.colors = o3d.utility.Vector3dVector(arm_rgb)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    ee_frame = copy.deepcopy(frame).translate(ee_position)
    ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

    o3d.visualization.draw_geometries([arm_pcd, ee_frame])


def load_data(arm_file_path):
    with open(arm_file_path, 'rb') as filehandler:
        xyz_origin, rgb, label, instance_label, pose = pickle.load(filehandler, encoding='bytes')

    arm_points = xyz_origin[label == 1]

    rgb = rgb[label == 1]
    rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
    rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
    rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

    return arm_points, rgb, pose


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcl', type=str)
    args = parser.parse_args()

    if os.path.isdir(args.pcl):
        file_names = os.listdir(args.pcl)
        files = [os.path.join(args.pcl, fn) for fn in file_names if fn[-7:] == ".pickle"]
    else:
        files = [args.pcl]

    i = 0
    file_count = len(files)

    while i < file_count:
        filename = files[i].split('/')[-1]
        arm_points, rgb, pose = load_data(files[i])

        visualize_res(
            arm_points,
            rgb,
            (pose[0], pose[1], pose[2]),
            (pose[3], pose[4], pose[5], pose[6])
        )

        command = input(f"({filename}) $: ")

        # Parse file commands and apply
        if command == "rm":
            os.remove(files[i])
            del files[i]
            file_count = len(files)

            print(f"Removed {filename}.")

            i -= 1

        i += 1
