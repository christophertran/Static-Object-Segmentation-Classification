# Static-Object-Segmentation-Classification

## Environment Setup
Instructions created following comments from [this](https://github.com/ros2/rosbag2/issues/139) thread
1. Ensure you have ROS2 Foxy installed. [Source](https://docs.ros.org/en/foxy/index.html)
2. Ensure you have ROS1 Noetic installed. [Source](http://wiki.ros.org/noetic)
3. Open a new terminal and install the rosbag_v2 plugin
    ```txt
    $ sudo apt install ros-foxy-rosbag2-bag-v2-plugins
    ```
4. Now you can playback ROS1 bags using ROS2 by following the instructions under [Terminal 1](###terminal-1)

## How to Run

### Terminal 1
This terminal will be used to playback ROS `.bag` files
```txt
# Source ROS1 install
$ source /opt/ros/noetic/setup.bash


# Source ROS2 install
$ source /opt/ros/foxy/setup.bash


# Playback ROS1 bag using ROS2
# -s specifies the storage identifier
# -l specifies loop playback of bag file
$ ros2 bag play ~/Path/To/Bag -s rosbag_v2 -l
```

### Terminal 2
This terminal will be used to build and run the ROS2 nodes
```txt
# Open dev_ws
$ cd ~/dev_ws/


# Source ROS2 install
$ source /opt/ros/foxy/setup.bash


# Build static_object_segmentation_classification package using colcon
# --symlink-install prevents having to build again after changing python scripts
$ colcon build --symlink-install --packages-select static_object_segmentation_classification


# Source overlay
$ source install/setup.bash


# Run static_object_segmentation_node
$ ros2 run static_object_segmentation_classification static_object_segmentation_node
```

### Terminal 3 (optional)
This terminal will be used for running rviz2
```txt
# Source ROS2 install
$ source /opt/ros/foxy/setup.bash


# Visualize using rviz2 (using the config file provided)
# Add a "&" to the end of the command to launch rviz2 as a background process
$ ros2 run rviz2 rviz2 -d ~/dev_ws/src/Static-Object-Segmentation-Classification/config/config.rviz
```

## Topic
### /cepton/points
**Type: sensor_msgs/msg/PointCloud2**

- Using PointCloud2 message with Python [Source](https://github.com/ros2/common_interfaces/blob/master/sensor_msgs_py/sensor_msgs_py/point_cloud2.py)

- Structure of PointCloud2 message. [Source](https://github.com/ros2/common_interfaces/blob/master/sensor_msgs/msg/PointCloud2.msg)
    ```txt
    # This message holds a collection of N-dimensional points, which may
    # contain additional information such as normals, intensity, etc. The
    # point data is stored as a binary blob, its layout described by the
    # contents of the "fields" array.
    #
    # The point cloud data may be organized 2d (image-like) or 1d (unordered).
    # Point clouds organized as 2d images may be produced by camera depth sensors
    # such as stereo or time-of-flight.

    # Time of sensor data acquisition, and the coordinate frame ID (for 3d points).
    std_msgs/Header header

    # 2D structure of the point cloud. If the cloud is unordered, height is
    # 1 and width is the length of the point cloud.
    uint32 height
    uint32 width

    # Describes the channels and their layout in the binary data blob.
    PointField[] fields

    bool    is_bigendian # Is this data bigendian?
    uint32  point_step   # Length of a point in bytes
    uint32  row_step     # Length of a row in bytes
    uint8[] data         # Actual point data, size is (row_step*height)

    bool is_dense        # True if there are no invalid points
    ```

- PointCloud2 message example content.
    ```
    [INFO] [1647673225.463304580] [static_object_segmentation_node]: 
    header: std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1645138913, nanosec=186668000), frame_id='cepton_10627')
    fields: 
        [
            sensor_msgs.msg.PointField(name='timestamp', offset=0, datatype=8, count=1), 
            sensor_msgs.msg.PointField(name='image_x', offset=8, datatype=7, count=1), 
            sensor_msgs.msg.PointField(name='distance', offset=12, datatype=7, count=1), 
            sensor_msgs.msg.PointField(name='image_z', offset=16, datatype=7, count=1), 
            sensor_msgs.msg.PointField(name='intensity', offset=20, datatype=7, count=1), 
            sensor_msgs.msg.PointField(name='return_type', offset=24, datatype=2, count=1), 
            sensor_msgs.msg.PointField(name='flags', offset=25, datatype=2, count=1), 
            sensor_msgs.msg.PointField(name='x', offset=28, datatype=7, count=1), 
            sensor_msgs.msg.PointField(name='y', offset=32, datatype=7, count=1), 
            sensor_msgs.msg.PointField(name='z', offset=36, datatype=7, count=1)
        ] 
    width: 21456 
    height: 1 
    point_step: 40 
    row_step: 858240
    ```