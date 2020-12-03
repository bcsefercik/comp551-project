import sensor_msgs.point_cloud2 as pc2
import rospy
import pickle

from sensor_msgs.msg import PointCloud2, PointField


input_folder = '_gitignore/bag_files/processsed/'
input_file   = '001'
input_ext    = 'pickle'
output_full  = output_folder + output_file + pickle
input_file   = pickle.load(f)


def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB
    
        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message
            
        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data


objs = []
while 1:
    try:
       objs.append(pickle.load(f))
    except EOFError:
        print("Starting transformation...")

for obj in objs:
    pcl_obj = ros_to_pcl(obj)
    print(pcl_obj)
    return
