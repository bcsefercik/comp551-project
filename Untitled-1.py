#!/usr/bin/env python
"""
The units can be seen in the below address:
https://www.ros.org/reps/rep-0103.html#units
"""

import roslib, rospy, sys, moveit_commander, tf
from ar_track_alvar_msgs.msg import AlvarMarkers
from panda_manipulation.manipulator import Manipulator
from geometry_msgs.msg import TransformStamped, Quaternion, Pose
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler
import PyKDL
from scipy.spatial.transform import Rotation as R
import numpy as np

def main():
    sys.argv.append("robot_description:=/robot_description")
    sys.argv.append("joint_states:=/joint_states")
    sys.argv.append("move_group:=/move_group")
    sys.argv.append("pickup:=/pickup")
    sys.argv.append("place:=/place")
    sys.argv.append("execute_trajectory:=/execute_trajectory")
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('kinect_tf', anonymous=True)

    robot                         = Manipulator(False)
    artag_pose_topic              = '/ar_pose_marker'
    artag_msg                     = rospy.wait_for_message(artag_pose_topic, AlvarMarkers, timeout=5)
    

    marker                        = artag_msg.markers[0]
    marker_posest                 = marker.pose
    marker_pose                   = marker_posest.pose
    marker_posest.header.frame_id = marker.header.frame_id

    #print 50*'*'
    #print "marker_pose is", marker_pose

    ee_posest                     = robot.arm.get_FK()
    ee_pose                       = ee_posest.pose

    print 50*'*'
    print "ee_pose is", ee_pose
   
    
    marker_translation            = PyKDL.Vector(marker_pose.position.x, marker_pose.position.y, marker_pose.position.z)
    marker_rotation               = PyKDL.Rotation.Quaternion(marker_pose.orientation.x, marker_pose.orientation.y, marker_pose.orientation.z, marker_pose.orientation.w)
    marker_frame                  = PyKDL.Frame(marker_rotation, marker_translation)


    _marker_pose = np.array( [marker_pose.position.x, marker_pose.position.y, marker_pose.position.z])
    _marker_quat = np.array( [marker_pose.orientation.x, marker_pose.orientation.y, marker_pose.orientation.z, marker_pose.orientation.w])
    _marker_rot  = R.from_quat(  _marker_quat ).as_dcm()
    
    print 50*'*'
    print "AR Euler rotations"
    print R.from_quat(  _marker_quat ).as_euler('zyx', degrees=True)

,

    _marker_transformation = [ [_marker_rot[0][0], _marker_rot[0][1], _marker_rot[0][2], _marker_pose[0]],
                               [_marker_rot[1][0], _marker_rot[1][1], _marker_rot[1][2], _marker_pose[1]],
                               [_marker_rot[2][0], _marker_rot[2][1], _marker_rot[2][2], _marker_pose[2]],
                               [0,             0,             0,             1        ]]
    _marker_transformation = np.array(_marker_transformation)

    print 50*'*'
    print "marker transf is"
    print _marker_transformation
    
    T_C_B  = _marker_transformation

    _ee_pose  = np.array( [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z])
    _ee_quat  = np.array([ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w] )
    _ee_rot   = R.from_quat(  _ee_quat ).as_dcm()
    _ee_transformation = [ [_ee_rot[0][0], _ee_rot[0][1], _ee_rot[0][2], _ee_pose[0]],
                           [_ee_rot[1][0], _ee_rot[1][1], _ee_rot[1][2], _ee_pose[1]],
                           [_ee_rot[2][0], _ee_rot[2][1], _ee_rot[2][2], _ee_pose[2]],
                           [0,             0,             0,             1        ]]
    _ee_transformation = np.array(_ee_transformation)
    T_A_B = _ee_transformation


    #T_A_C = np.dot(T_A_B,np.linalg.inv(T_C_B))
    T_A_C = np.dot(T_A_B,T_C_B)
    print 50*'*'
    print "base to kinect our"
    print(T_A_C)

    ee_translation                = PyKDL.Vector(ee_pose.position.x, ee_pose.position.y, ee_pose.position.z)
    ee_rotation                   = PyKDL.Rotation.Quaternion(ee_pose.orientation.x,ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w)
    ee_frame                      = PyKDL.Frame(ee_rotation, ee_translation)

    # Make transformation
    tf_frame       = ee_frame * marker_frame.Inverse()
    tf_rotation    = tf_frame.M
    tf_translation = tf_frame.p
    r, p, y        = tf_rotation.GetRPY()
    rx, ry, rz, rw = quaternion_from_euler(r, p, y)

    print 50*'*'
    print "Base to camera transformation"
    print tf_translation[0], tf_translation[1], tf_translation[2], rx, ry, rz, rw

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    