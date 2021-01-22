import os
import pickle 
import PyKDL
import numpy as np
from tf.transformations import quaternion_from_euler

folder_name       = '_gitignore/Dataset/p1/full_light/robotNet/train_raw_pose_p2/'
output_name       = '_gitignore/Dataset/p1/full_light/robotNet/TEST_PCL_CONVERTED/'
file_names        = [name for name in os.listdir(folder_name) if os.path.isdir(folder_name)]
robot_to_kinect   = [0.524, 0.31, 0.936, 0.33650, 0.358586, -0.603734, 0.627441]

base_to_camera_translation = PyKDL.Vector(robot_to_kinect[0], robot_to_kinect[1], robot_to_kinect[2]) 
base_to_camera_rotation    = PyKDL.Rotation.Quaternion(robot_to_kinect[3], robot_to_kinect[4], robot_to_kinect[5], robot_to_kinect[6])
base_to_camera_frame       = PyKDL.Frame(base_to_camera_rotation,base_to_camera_translation)

def apply_transform(xyz_camera):
    '''
        xyz_camera :  pcd points in camera frame
    '''
    points = []
    for point in xyz_camera:
        p = PyKDL.Vector(point[0],point[1],point[2])
        p = base_to_camera_frame.Inverse() * p

        points.append([p.x(),p.y(),p.z()])
    #xyz_base             = base_to_camera_frame.Inverse() * xyz_camera

    return np.array(points)

i = 1
print "OUTPUT FOLDER: ", output_name
print "Input FOLDER: ", folder_name
for file_name in file_names:
    print(i,'/' ,len(file_names))
    filehandler = open(folder_name + file_name,'rb')
    xyz_origin, rgb, label, instance_label,pose = pickle.load(filehandler)
    filehandler.close()

    xyz_base = apply_transform(xyz_origin)

    print(file_name, pose)
    robotNetObject = (xyz_base, rgb, label, instance_label,pose)
    filehandler = open(output_name+file_name,"wb")
    pickle.dump(robotNetObject, filehandler, protocol=2)
    filehandler.close()
    i = i+1