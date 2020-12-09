"""
Process the bag file to extract its content
to pickle file
COMP551 TEAM
"""

import rospy
from roslib import message
import pickle
from sensor_msgs.msg import PointCloud2


output_folder = '_gitignore/depth_registered_files/'
output_file   = 'moving_001'
output_ext    = '.pickle'
output_full   = output_folder + output_file + output_ext

output = open(output_full,'wb')
i      = 1
def callback(data):
    global i
    print('Frame No: ',i)
    i = i + 1
    pickle.dump(data,output)

def listener():
    rospy.init_node('depth_registered_recorder')
    rospy.Subscriber('/camera/depth_registered/points',PointCloud2,callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
