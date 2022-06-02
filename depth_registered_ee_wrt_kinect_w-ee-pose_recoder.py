import argparse

import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import rospy
from roslib import message
import pickle
import pdb


output = None
output_seq2ja = None
output_ja2jointangle = None

i = 0
def callback(depth_registered_sub, ee_wrt_kinect_state_sub, ee_pose):
    global i
    i += 1
    print('Frame No: ',i, 'Seq', ee_wrt_kinect_state_sub.header.seq)
    # pdb.set_trace()
    data = [depth_registered_sub,ee_wrt_kinect_state_sub, ee_pose]
    pickle.dump(data,output)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert background bag playback to pickle.')
    parser.add_argument('--outfolder', default='tmp', type=str)
    parser.add_argument('--outfile', default='ee', type=str)

    args = parser.parse_args()

    output_folder = args.outfolder
    output_file   = args.outfile
    output_ext    = '.pickle'
    output_full   = output_folder + '/' + output_file + output_ext
    output = open(output_full,'wb')

    rospy.init_node('depth_registered_joint_state_recorder')
    depth_registered_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
    ee_wrt_kinect_state_sub = message_filters.Subscriber('/robot/ee_pose_wrt_kinect', PoseStamped)
    ee_pose_state_sub = message_filters.Subscriber('/robot/ee_pose', PoseStamped)
    # ee_wrt_kinect_state_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
    print('Starting...')
    ts = message_filters.ApproximateTimeSynchronizer([depth_registered_sub, ee_wrt_kinect_state_sub, ee_pose_state_sub], queue_size=100, slop=0.1)
    ts.registerCallback(callback)

    rospy.spin()

