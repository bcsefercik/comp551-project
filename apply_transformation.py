import os
import pickle 
import PykDL

folder_name       = 'RobotNet/dataset/alivev1_raw_pose/train/'
output_name       = 'RobotNet/dataset/alivev1/train'
file_names        = [name for name in os.listdir(folder_name) if os.path.isdir(folder_name)]


robot_to_kinect   = [0.524, 0.31, 0.936, 0.33650, 0.358586, -0.603734, 0.627441]

robot_to_kinect_translation = PyKDL.Vector(robot_to_kinect[0], robot_to_kinect[1], robot_to_kinect[2]) 
robot_to_kinect_rotation    = PyKDL.Vector(robot_to_kinect[3], robot_to_kinect[4], robot_to_kinect[5])
robot_to_kinect_frame       = PyKDL.Frame(robot_to_kinect_translation, robot_to_kinect_rotation)


def apply_transform(ee_pose):
    '''
        ee_pose :  (translation in meters) x , y, z  (ortientation) x, y, z, w
    '''
    ee_translation  = PyKDL.Vector(ee_pose[0], ee_pose[1], ee_pose[2])      
    ee_rotation     = PyKDL.Rotation.Quaternion(ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6])
    ee_frame        = PyKDL.Frame(ee_rotation, ee_translation)

    ee_kinect             = ee_frame * robot_to_kinect_frame.Inverse()
    ee_kinect_rotation    = ee_kinect.M
    ee_kinect_translation = ee_kinect.p

    return ee_kinect_translation[0],ee_kinect_translation[1], ee_kinect_translation[2], ee_kinect_rotation[0], ee_kinect_rotation[1],ee_kinect_rotation[2], ee_kinect_rotation[3]

for file_name in file_names:
    with open(folder_name +file_name, 'rb') as f:
        xyz_origin, rgb, label, instance_label,pose = pickle.load(f)
        pose = apply_transform(pose)

