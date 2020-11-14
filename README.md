# comp551-project


# Kinect Recording
* Recording
1. Start roscore, and then run:
    roslaunch freenect_launch freenect.launch depth_registration:=true

2. Start to rosbag to record:
    rosbag record camera/depth_registered/image_raw camera/depth_registered/camera_info camera/rgb/image_raw camera/rgb/camera_info

* Playback:
1. Set simulation time:
    rosparam set /use_sim_time true
2. Close any running freenect driver and re-run as:
    roslaunch freenect_launch freenect.launch load_driver:=false

3. If you want to visualization: 
    run rviz, create a PointCloud2 display of the topic /camera/depth_registered/points with the fixed frame 
    /camera_link 

4. If you dont see any thing try to run this to whether any messages are published:
    rostopic hz /camera/depth_registered/points

5. finally play the bag file:
    rosbag play --clock kinect.bag

    This way you can access the points generated in topic /camera/depth_registered/points which has the type of PointCloud2
    and containing both RGB and XYZ information. In order to process the data please refer to another function.


! Please refer to [here](http://wiki.ros.org/openni_launch/Tutorials/BagRecordingPlayback) to more information.
