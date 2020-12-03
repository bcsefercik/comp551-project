"""
Process the bag file to extract its content
to given files into more readable format
COMP551 TEAM
"""

import rospy
import pickle


output_folder = '_gitignore/bag_files/processsed/'
output_file   = '001'
output_full   = output_folder + output_file

output = open(output_full,'wb')

def callback(data):
    print('Get data')
    pickle.dump(data.data,output)

def listener():
    rospy.init_node('depth_registered_recorder')
    rospy.Subscriber('/camera/depth_registered/points',,callback)
    rospy.spin()


if __name__ == '__main__':
    listener()

