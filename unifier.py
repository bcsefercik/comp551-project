"""
Given a folder with multiple pcd files it loads them and combine them to generate
one pcd file
"""

from os import listdir
from os.path import isfile, join
import open3d as o3d
import numpy as np
from depth_registered_to_pcl import write_pcd



path = '/home/onurberk/Desktop/development/comp551-project/_gitignore/pcd_files/background_000'
file_names = [f for f in listdir(path) if isfile(join(path, f))]
n_initial  = 0
n_next     = 0



def load_pcd(file_name):
    return o3d.io.read_point_cloud(file_name)

def count_points(pcd_obj):
    return np.asarray(pcd_obj.points).shape[0]

def update_frame(pcd_obj_1,pcd_obj_2):
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined = pcd_obj_1 + pcd_obj_2
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
    background = update_frame(background,next_frame)
    n_next     = count_points(background)
    print("Total Number of Points: ", n_next)
    #percentages.append(find_added_percentage(n_initial,n_next))
    print("Currently: ", i, '/', str(len(file_names)-1) )
#print(percentages)
apply_downsampling = False

if apply_downsampling == True:
    print("Starting downsampling")
    voxel_size = 0.02
    print("Voxel size: ", voxel_size)
    background = background.voxel_down_sample(voxel_size=voxel_size)
    print("Final Number of Points: ",count_points(background))



output_file = '/home/onurberk/Desktop/development/comp551-project/_gitignore/pcd_files/unified/unified_background_000.pcd'
o3d.io.write_point_cloud(output_file, background, write_ascii=False, compressed=False, print_progress=True)
