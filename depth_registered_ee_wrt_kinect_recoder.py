import argparse

import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import rospy
from roslib import message
import pickle


output = None

i = 0
def callback(depth_registered_sub, ee_wrt_kinect_state_sub):
    global i
    print('Frame No: ',i)
    i = i + 1
    data = [depth_registered_sub,ee_wrt_kinect_state_sub]
    pickle.dump(data,output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert background bag playback to pickle.')
    parser.add_argument('--outfolder', default='tmp', type=str)
    parser.add_argument('--outfile', default='background', type=str)

    args = parser.parse_args()

    output_folder = args.outfolder
    output_file   = args.outfile
    output_ext    = '.pickle'
    output_full   = output_folder + '/' + output_file + output_ext
    output = open(output_full,'wb')

    rospy.init_node('depth_registered_joint_state_recorder')
    depth_registered_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
    ee_wrt_kinect_state_sub = message_filters.Subscriber('/robot/ee_pose_wrt_kinect', PoseStamped)
    # ee_wrt_kinect_state_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
    print('Starting...')
    ts = message_filters.ApproximateTimeSynchronizer([depth_registered_sub, ee_wrt_kinect_state_sub], queue_size=100, slop=0.1)
    ts.registerCallback(callback)
    rospy.spin()

