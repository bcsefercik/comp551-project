"""
Process the pickle file to extract its content
to pickle file
COMP551 TEAM
"""

user_name                 = 'onurberk'
does_this_machine_has_ros = True

import numpy as np
import pickle

import sys
import logging
import time
import os
import signal
import copy
import json
import math
import argparse

development_path   = '/home/' + user_name + '/Desktop/development'
robot_control_path = '/home/' + user_name + '/Desktop/development/Tubitak'

sys.path.insert(0,robot_control_path)
sys.path.insert(0, development_path)

if does_this_machine_has_ros == True:
    import rospy
    import rospkg
    import rosnode
    from roslib import message
    from sensor_msgs.msg import JointState
    from perpy.camera import Kinect
    from std_msgs.msg import String, Float32MultiArray
    from Tubitak.robot_control import Agent
    from geometry_msgs.msg import Pose, PoseStamped



robot = Agent(sim=True, data_type='poses')
robot.initialize_robot()



if __name__ == "__main__":

    input_folder = '_gitignore/Dataset/p1/full_light/'
    input_file   = 'perception_joint_states'
    input_ext    = '.pickle'
    input_full   = input_folder + input_file + input_ext
    input_file   = open(input_full,'rb')

    output_folder = '_gitignore/Dataset/p1/full_light/'
    output_file   = 'ee_poses'
    ee_poses = []


    i = 1
    while True:
        try:
            _,joint_states = pickle.load(input_file)
            ee_poses.append(robot.robot.arm.get_FK(joint_states.position))
            print("Output: ",i)
            i = i+1

        except EOFError:
            print("Done...")
            break

    output_full = output_folder + output_file
    np.save(output_full,ee_poses)