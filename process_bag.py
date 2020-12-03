"""
Process the bag file to extract its content
to pickle file
COMP551 TEAM
"""

import rospy
import pickle
import sensor_msgs.point_cloud2 as pc2

output_folder = '_gitignore/bag_files/processsed/'
output_file   = '001'
output_full   = output_folder + output_file

output = open(output_full,'wb')

def callback(data):
    print('Get data')
    pickle.dump(data.data,output)

def listener():
    rospy.init_node('depth_registered_recorder')
    rospy.Subscriber('/camera/depth_registered/points',pc2,callback)
    rospy.spin()


if __name__ == '__main__':
    listener()

