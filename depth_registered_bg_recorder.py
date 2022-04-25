"""
Process the bag file to extract its content
to pickle file
COMP551 TEAM
"""
import sys
import argparse

import rospy
from roslib import message
import pickle
from sensor_msgs.msg import PointCloud2


output_folder = 'tmp'
output_file   = 'background'
output_ext    = '.pickle'
output_full   = output_folder + '/' + output_file + output_ext

output = None
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
    parser = argparse.ArgumentParser(description='Convert background bag playback to pickle.')
    parser.add_argument('--outfolder', default=output_folder, type=str)
    parser.add_argument('--outfile', default=output_file, type=str)

    args = parser.parse_args()

    output_folder = args.outfolder
    output_file = args.outfile
    output_full   = output_folder + '/' + output_file + output_ext
    output = open(output_full,'wb')

    listener()
