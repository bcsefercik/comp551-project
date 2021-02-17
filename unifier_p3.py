"""
Given a folder with multiple pcd files it loads them and combine them to generate
one pcd file
"""

from os import listdir
from os.path import isfile, join
import open3d as o3d
import numpy as np
from depth_registered_to_pcl import write_pcd


common_path = '/home/onurberk/Desktop/development/comp551-project/_gitignore/Dataset/p1/half_light/'

path = common_path + 'background/pcd'
file_names = [f for f in listdir(path) if isfile(join(path, f)) and f[-4:] == ".pcd"]
# file_names = ['BG_1.pcd', 'BG_5.pcd','BG_9.pcd']
n_initial  = 0
n_next     = 0



def load_pcd(file_name):
    return o3d.io.read_point_cloud(file_name)

def count_points(pcd_obj):
    return np.asarray(pcd_obj.points).shape[0]

def update_frame(curr_background, next_background, removal_th=0.01):

    dists                   = np.asarray(next_background.compute_point_cloud_distance(curr_background))
    point_mask              = dists > removal_th
    point_idx               = np.where(point_mask == True)[0]
    next_background_refined = next_background.select_by_index(point_idx)
    pcd_combined = curr_background + next_background_refined
    del curr_background
    del next_background
    del next_background_refined
    return pcd_combined


def find_added_percentage(n_initial,n_final):
    n_added    = n_final - n_initial
    percentage = (n_added * 100) / (n_initial)
    return percentage

background  = load_pcd( path + '/' + file_names[0])
n_next      = count_points(background)
percentages = [100] 
for i in range(1,len(file_names)):
    n_initial  = n_next
    next_frame = load_pcd(path + '/' + file_names[i])
    background = update_frame(background, next_frame)
    n_next     = count_points(background)
    print("Total Number of Points: ", n_next)
    percentages.append(find_added_percentage(n_initial,n_next))
    print("Currently: ", i, '/', str(len(file_names)-1) )
print(percentages)
apply_downsampling = False

if apply_downsampling == True:
    print("Starting downsampling")
    voxel_size = 0.02
    print("Voxel size: ", voxel_size)
    background = background.voxel_down_sample(voxel_size=voxel_size)
    print("Final Number of Points: ",count_points(background))



output_file = common_path + 'background/combined_bg2.pcd'
o3d.io.write_point_cloud(output_file, background, write_ascii=False, compressed=False, print_progress=True)
