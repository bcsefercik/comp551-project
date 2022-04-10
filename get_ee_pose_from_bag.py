"""
Process the bag file to extract its content
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


import rosbag
bag = rosbag.Bag('/home/onurberk/Desktop/development/comp551-project/_gitignore/Dataset/p1/full_light/random_movement.bag')
ee_poses = []
i = 1
for topic, msg, t in bag.read_messages(topics=['/franka_state_controller/joint_states']):
    print(i)
    ee_poses.append(robot.robot.arm.get_FK(msg.position))
    i += 1
np.save('ee_poses',ee_poses)

bag.close()
