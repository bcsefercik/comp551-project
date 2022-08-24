# comp551-project

# PointGroup

Please refer to the readme inside the pointgroup

# Combining Multiple PointClouds

We can combine multiple pointcloud pcd files into one big pointcloud.The code also allows you to downsample.

# Kinect Recording/Playback

There are possible ways to record the kinect data, first one, the hardest, memory efficient
and faster method is to save the four different topic and generate the depth data later.
The steps for that described below:

* Recording (Robot EE wrt Kinect and Kinect) (AKA Robot EE + PointCloud)
1. Start roscore, and then run:
    ```roslaunch freenect_launch freenect.launch depth_registration:=true```

2. Start Lfd to record robot end effector

    ```roslaunch lfd_data lfd_nodes.launch```

3. Start to rosbag to record:
    ```rosbag record camera/depth_registered/image_raw camera/depth_registered/camera_info camera/rgb/image_raw camera/rgb/camera_info  /robot/ee_pose_wrt_kinect /robot/ee_pose /robot/joint_angles /franka_state_controller/joint_states --split --duration=210```

* Recording (Kinect) (AKA Background)
1. Start roscore, and then run:
    ```roslaunch freenect_launch freenect.launch depth_registration:=true```

2. Start to rosbag to record:
    ```rosbag record camera/depth_registered/image_raw camera/depth_registered/camera_info camera/rgb/image_raw camera/rgb/camera_info ```

* Collect Data for Prediction
```rosbag record camera/depth_registered/image_raw camera/depth_registered/camera_info camera/rgb/image_raw camera/rgb/camera_info  /robot/ee_pose_wrt_kinect /robot/ee_pose /robot/joint_angles /robot/joint_states2 /franka_state_controller/joint_states```




* Playback (Recorded Movement)

1. Please run the replay.py

* Playback (A Bag File):
1. Set simulation time:
    ```rosparam set /use_sim_time true```
2. Close any running freenect driver and re-run as:
    ```roslaunch freenect_launch freenect.launch load_driver:=false```

3. If you want to viz: 
    run rviz, create a PointCloud2 display of the topic /camera/depth_registered/points with the fixed frame 
    /camera_link 

4. If you dont see any thing try to run this to whether any messages are published:
    ```rostopic hz /camera/depth_registered/points```

5. finally play the bag file:
    ```rosbag play --rate 0.5 --clock kinect.bag```

    This way you can access the points generated in topic /camera/depth_registered/points which has the type of PointCloud2
    and containing both RGB and XYZ information. In order to process the data please refer to another function.

    You can change the rate to play it faster or slower

6. After that you need to follow the second method to access the data.

! Please refer to [here](http://wiki.ros.org/openni_launch/Tutorials/BagRecordingPlayback) to more information.

The second method which will end up with less collected data, will start by automatically saving the depth registered data 
while robot moves.

* Recording

1. You need to run the depth_registered_recorder.py, this way you will save the files into specified folder.If you are
    using the first method, you have to do this to collect data. 

* Accessing the data

1. Run the pcd_bg_extractor.py to get the pcl data, at this step you have the pcl data and you can look at them using
    pcl_viewer for making them numpy array please refer to second step

2. In order to make the pcl to numpy array please use pcl_to_numpy.py. 


## TODO
- [x] Pointgroup server setup
- [ ] Data Collection
    - [x] Data Recorder and conver codes
    - [x] Labeling Code
      - [x] Background subtraction
      - [x] Mesh reconstruction

  

