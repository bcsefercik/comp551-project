import message_filters
from sensor_msgs.msg import PointCloud2, JointState
import rospy
from roslib import message
import pickle


output_folder = '_gitignore/Dataset/p1/full_light/'
output_file   = 'perception_joint_states2'
output_ext    = '.pickle'
output_full   = output_folder + output_file + output_ext

output = open(output_full,'wb')

i = 0
def callback(depth_registered_sub, joint_state_sub):
    global i
    print('Frame No: ',i)
    i = i + 1
    data = [depth_registered_sub,joint_state_sub]
    pickle.dump(data,output)

rospy.init_node('depth_registered_joint_state_recorder')
depth_registered_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
joint_state_sub = message_filters.Subscriber('/franka_state_controller/joint_states', JointState)

ts = message_filters.ApproximateTimeSynchronizer([depth_registered_sub, joint_state_sub], 10000, slop=0.01)
ts.registerCallback(callback)
rospy.spin()

