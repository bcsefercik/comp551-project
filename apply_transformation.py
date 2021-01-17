import os
import pickle 
import PyKDL
import numpy
from tf.transformations import quaternion_from_euler

folder_name       = '_gitignore/Dataset/p1/full_light/robotNet/train_raw_pose_p2/'
output_name       = '_gitignore/Dataset/p1/full_light/robotNet/train/'
file_names        = [name for name in os.listdir(folder_name) if os.path.isdir(folder_name)]

robot_to_kinect   = [0.524, 0.31, 0.936, 0.33650, 0.358586, -0.603734, 0.627441]

robot_to_kinect_translation = PyKDL.Vector(robot_to_kinect[0], robot_to_kinect[1], robot_to_kinect[2]) 
<<<<<<< HEAD
robot_to_kinect_rotation    = PyKDL.Rotation.Quaternion(robot_to_kinect[3], robot_to_kinect[4], robot_to_kinect[5], robot_to_kinect[6])
robot_to_kinect_frame       = PyKDL.Frame(robot_to_kinect_rotation,robot_to_kinect_translation)
=======
robot_to_kinect_rotation    = PyKDL.Vector(robot_to_kinect[3], robot_to_kinect[4], robot_to_kinect[5])
robot_to_kinect_frame       = PyKDL.Frame(robot_to_kinect_translation, robot_to_kinect_rotation)
>>>>>>> 25af3f55a75b45dcc2515fcc9b77325883003864

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
    r, p, y               = ee_kinect_rotation.GetRPY()
    rx, ry, rz, rw        = quaternion_from_euler(r, p, y)

    return np.array([ee_kinect_translation[0],ee_kinect_translation[1], ee_kinect_translation[2],rx, ry, rz, rw])

i = 1
for file_name in file_names:
    print(i,'/' ,len(file_names))
    filehandler = open(folder_name + file_name,'rb')
    xyz_origin, rgb, label, instance_label,pose = pickle.load(filehandler)
    filehandler.close()
    pose = apply_transform(pose)
    robotNetObject = (xyz_origin, rgb, label, instance_label,pose)
    filehandler = open(output_name+file_name,"wb")
    pickle.dump(robotNetObject, filehandler, protocol=2)
    filehandler.close()
    i = i+1