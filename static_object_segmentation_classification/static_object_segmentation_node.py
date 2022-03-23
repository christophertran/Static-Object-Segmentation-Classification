import sys
import os

import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


SUB_TOPIC = "cepton/points"

PUB_TOPIC = "static_object_segmentation_node/labels"


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("static_object_segmentation_node")

        # This is for visualization of the received point cloud.
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([255, 255, 255])

        self.view_control = self.vis.get_view_control()

        self.pcd_as_numpy_array = np.asarray([])
        self.o3d_pcd = o3d.geometry.PointCloud()
        self.outlier_o3d_pcd = o3d.geometry.PointCloud()

        # Set up a subscription to the SUB_TOPIC topic with a
        # callback to the function 'listener_callback'
        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,  # Msg type
            SUB_TOPIC,  # topic
            self.listener_callback,  # Function to call
            10,  # QoS
        )

        # Set up a publisher to the PUB_TOPIC topic
        # with a callback to the function 'timer_callback'
        self.pcd_publisher = self.create_publisher(
            sensor_msgs.PointCloud2,  # Msg type
            PUB_TOPIC,  # topic
            10,  # QoS
        )

        timer_period = 1 / 10  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def listener_callback(self, msg):
        # self.get_logger().info(
        #     f"header: {msg.header} "
        #     + f"fields: {msg.fields} "
        #     + f"width: {msg.width} "
        #     + f"height: {msg.height} "
        #     + f"point_step: {msg.point_step} "
        #     + f"row_step: {msg.row_step}"
        # )

        # Parameters for voxel_down_sample function
        # voxel_size defines the size of the voxel for downsampling
        voxel_size = 0.05

        # Parameters for segment_plane function
        # distance_threshold defines the maximum distance a point can have
        # to an estimated plane to be considered an inlier
        distance_threshold = 0.25
        # ransac_n defines the number of points that are randomly samples to estimate a plane
        ransac_n = 5
        # num_iterations defines how often a random plane is sampled and verified
        num_iterations = 50

        # Parameters for cluster_dbscan function
        # eps defines the distance to neighbors in a cluster
        eps = 0.50
        # min_points defines the minimum numebr of points required to form a cluster
        min_points = 20

        # Parameters for changing viewpoint of visualizer
        # viewcontrol_front defines the front vector of the visualizer
        viewcontrol_front = [
            -0.089934221049818977,
            -0.99314925303098789,
            -0.074608291014828354,
        ]
        # viewcontrol_lookat defines the lookat vector of the visualizer
        viewcontrol_lookat = [
            9.2692756652832031,
            99.291183471679688,
            9.1466994285583496,
        ]
        # viewcontrol_up defines the up vector of the visualizer
        viewcontrol_up = [
            -0.03194282842538148,
            -0.07199697911102805,
            0.99689321931241603,
        ]
        # viewcontrol_zoom defines the zoom of the visualizer
        viewcontrol_zoom = 0.2999999999999996

        # Convert the 'msg', which is of type PointCloud2 to a numpy array
        # Convert the numpy array to a open3d PointCloud
        self.o3d_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                np.asarray(list(read_points(msg, field_names=["x", "y", "z"])))
            )
        )

        # Voxel downsampling uses a regular voxel grid to create
        # a uniformly downsampled point cloud from an input point cloud.
        self.o3d_pcd = self.o3d_pcd.voxel_down_sample(voxel_size=0.05)

        # Segment plane in attempt to remove ground plane
        plane_model, inliers = self.o3d_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        # Outliers = Points not part of ground plane
        self.outlier_o3d_pcd = self.o3d_pcd.select_by_index(inliers, invert=True)

        # Cluster points using dbscan
        labels = np.asarray(
            self.outlier_o3d_pcd.cluster_dbscan(eps=eps, min_points=min_points)
        )

        # Apply different colors to clusters
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        self.outlier_o3d_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Output that is being published, it is a point cloud and each point is labeled
        # and either in a cluster labeled [0, n] or is labeled -1 for noise.
        self.pcd_as_numpy_array = np.append(
            np.asarray(self.outlier_o3d_pcd.points), labels.reshape(-1, 1), axis=1
        )

        # This is for visualization of the received point cloud.
        self.vis.clear_geometries()
        self.vis.add_geometry(self.outlier_o3d_pcd)

        # Move viewpoint camera
        self.view_control.set_front(viewcontrol_front)
        self.view_control.set_lookat(viewcontrol_lookat)
        self.view_control.set_up(viewcontrol_up)
        self.view_control.set_zoom(viewcontrol_zoom)

        self.vis.poll_events()
        self.vis.update_renderer()

        # o3d.visualization.draw_geometries(
        #     [self.o3d_pcd],
        #     front=viewcontrol_front,
        #     lookat=viewcontrol_lookat,
        #     up=viewcontrol_up,
        #     zoom=viewcontrol_zoom,
        # )

    def timer_callback(self):
        header = std_msgs.Header(frame_id="map")

        # This function has be modified to add a "label" field to the message
        self.pcd_publisher.publish(create_cloud_xyz32(header, self.pcd_as_numpy_array))


"""
Serialization of sensor_msgs.PointCloud2 messages.
Author: Tim Field
ROS 2 port by Sebastian Grans
File originally ported from:
https://github.com/ros/common_msgs/blob/f48b00d43cdb82ed9367e0956db332484f676598/
sensor_msgs/src/sensor_msgs/point_cloud2.py
"""

from collections import namedtuple
import ctypes
import math
import struct
import sys

from sensor_msgs.msg import PointCloud2, PointField


_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a sensor_msgs.PointCloud2 message.
    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param skip_nans: If True, then don't return any point with a NaN value.
                      (Type: Bool, Default: False)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: empty list)
    :return: Generator which yields a list of values for each point.
    """
    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )

    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def create_cloud(header, fields, points):
    """
    Create a sensor_msgs.msg.PointCloud2 message.
    :param header: The point cloud header. (Type: std_msgs.msg.Header)
    :param fields: The point cloud fields.
                   (Type: iterable of sensor_msgs.msg.PointField)
    :param points: The point cloud points. List of iterables, i.e. one iterable
                   for each point, with the elements of each iterable being the
                   values of the fields for that point (in the same order as
                   the fields parameter)
    :return: The point cloud as sensor_msgs.msg.PointCloud2
    """
    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))

    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))

    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    for p in points:
        pack_into(buff, offset, *p)
        offset += point_step

    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=cloud_struct.size,
        row_step=cloud_struct.size * len(points),
        data=buff.raw,
    )


def create_cloud_xyz32(header, points):
    """
    Create a sensor_msgs.msg.PointCloud2 message with (x, y, z) fields.
    :param header: The point cloud header. (Type: std_msgs.msg.Header)
    :param points: The point cloud points. (Type: Iterable)
    :return: The point cloud as sensor_msgs.msg.PointCloud2.
    """
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="label", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    return create_cloud(header, fields, points)


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"

    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                "Skipping unknown PointField datatype [%d]" % field.datatype,
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def main(args=None):
    rclpy.init(args=args)

    segmentation_node = SegmentationNode()

    rclpy.spin(segmentation_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    segmentation_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
