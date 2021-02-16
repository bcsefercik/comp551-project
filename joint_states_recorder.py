"""
Record the movement of the robot
COMP551 TEAM
"""

import rospy
from roslib import message
import pickle
from sensor_msgs.msg import JointState
import numpy as np

joint_states = list()
def callback(data):
    global joint_states
    joint_states.append(data)


def listener():

    rospy.init_node('joint_state_recorder')
    rospy.Subscriber('/franka_state_controller/joint_states',JointState,callback)
    rospy.spin()

def myhook():
    global joint_states
    joint_states2 = np.array(joint_states)
    np.save('joint_states', joint_states2)
    print('shutdown time!')

if __name__ == '__main__':
    rospy.on_shutdown(myhook)
    print("Starting recording...")
    listener()
