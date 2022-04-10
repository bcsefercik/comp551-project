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

development_path   = '../'
robot_control_path = '../Tubitak'
# robot_control_path = '../'

sys.path.insert(0, robot_control_path)
sys.path.insert(0, development_path)

if does_this_machine_has_ros == True:
    import rospy
    import rospkg
    import rosnode

    from perpy.camera import Kinect
    from std_msgs.msg import String, Float32MultiArray
    from Tubitak.robot_control import Agent
    from geometry_msgs.msg import Pose, PoseStamped


file_name = 'joint_states.npy'
if len(sys.argv) > 1:
    file_name = sys.argv[1]

joint_states = np.load(file_name, allow_pickle=True)
print("shape:", joint_states.shape)

robot = Agent(sim=False, data_type='poses')
robot.initialize_robot()
robot.robot.arm.error_recovery()

real_joint_states = list()
durations         = list()
curr = 1
for i in range(len(joint_states)):
    real_joint_states.append(joint_states[i].position)
    durations.append(curr)
    curr += 0.1

real_joint_states = np.array(real_joint_states)
durations = np.array(durations)

current_joints = robot.robot.arm.get_joints()
starting_pos   = real_joint_states[0]

print("Going to initial position")

robot.robot.arm.execute_trajectory(np.array([current_joints,starting_pos]), [10,11,12])

print("Starting...")

robot.robot.arm.execute_trajectory(real_joint_states, durations)

print('Done')
