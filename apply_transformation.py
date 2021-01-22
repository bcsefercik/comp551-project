import os
import pickle 
import PyKDL
import numpy as np
from tf.transformations import quaternion_from_euler

folder_name       = '_gitignore/Dataset/p1/full_light/robotNet/train_raw_pose_p2/'
output_name       = '_gitignore/Dataset/p1/full_light/robotNet/PLZ/'
file_names        = [name for name in os.listdir(folder_name) if os.path.isdir(folder_name)]
#robot_to_kinect   = [0.524, 0.31, 0.936, 0.33650, 0.358586, -0.603734, 0.627441]
robot_to_kinect   = [0.524, 0.31, 0.936, -0.023, 0.963, -0.268, 0.001]

base_to_camera_translation = PyKDL.Vector(robot_to_kinect[0], robot_to_kinect[1], robot_to_kinect[2]) 
base_to_camera_rotation    = PyKDL.Rotation.Quaternion(robot_to_kinect[3], robot_to_kinect[4], robot_to_kinect[5], robot_to_kinect[6])
base_to_camera_frame       = PyKDL.Frame(base_to_camera_rotation,base_to_camera_translation)

def apply_transform(ee_pose):
    '''
        ee_pose :  (translation in meters) x , y, z  (ortientation) x, y, z, w
    '''
    ee_translation  = PyKDL.Vector(ee_pose[0], ee_pose[1], ee_pose[2])
    ee_rotation     = PyKDL.Rotation.Quaternion(ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6])
    ee_frame        = PyKDL.Frame(ee_rotation, ee_translation)

    ee_kinect             = base_to_camera_frame.Inverse() * ee_frame
    ee_kinect_rotation    = ee_kinect.M
    ee_kinect_translation = ee_kinect.p
    r, p, y               = ee_kinect_rotation.GetRPY()
    rx, ry, rz, rw        = quaternion_from_euler(r, p, y)

    return np.array([ee_kinect_translation[0],ee_kinect_translation[1], ee_kinect_translation[2],rx, ry, rz, rw])

i = 1
for file_name in file_names:
    print(file_name,'/' ,len(file_names))
    filehandler = open(folder_name + file_name,'rb')
    xyz_origin, rgb, label, instance_label,pose = pickle.load(filehandler)
    print(pose)
    #exit()
    filehandler.close()
    pose = apply_transform(pose)
    print(file_name, pose)
    robotNetObject = (xyz_origin, rgb, label, instance_label,pose)
    filehandler = open(output_name+file_name,"wb")
    pickle.dump(robotNetObject, filehandler, protocol=2)
    filehandler.close()
    i = i+1