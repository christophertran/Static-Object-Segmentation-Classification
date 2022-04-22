import sys
import os

import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import vision_msgs.msg as vision_msgs
import geometry_msgs.msg as geometry_msgs

import numpy as np
import pyquaternion as pyq
import open3d as o3d
import matplotlib.pyplot as plt

from collections import defaultdict

SUB_TOPIC = "static_object_segmentation_node/labels"

PUB_TOPIC = "static_object_classification_node/bounding_boxes"


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("static_object_classification_node")

        # This is for visualization of the received point cloud.
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([255, 255, 255])

        self.view_control = self.vis.get_view_control()

        self.pcd_as_numpy_array = np.asarray([])
        self.o3d_pcd = o3d.geometry.PointCloud()

        self.o3d_bboxs = []

        self.bboxs = []

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
            vision_msgs.BoundingBox3DArray,  # Msg type
            PUB_TOPIC,  # topic
            10,  # QoS
        )

        timer_period = 1 / 10  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def listener_callback(self, msg):
        # Parameters for changing viewpoint of visualizer
        # viewcontrol_front defines the front vector of the visualizer
        viewcontrol_front = [
            -0.24,
            -1.4,
            0,
        ]
        # viewcontrol_lookat defines the lookat vector of the visualizer
        viewcontrol_lookat = [
            6,
            60,
            0.6,
        ]
        # viewcontrol_up defines the up vector of the visualizer
        viewcontrol_up = [
            0.2,
            1,
            1,
        ]
        # viewcontrol_zoom defines the zoom of the visualizer
        viewcontrol_zoom = 0.5

        # Convert the 'msg', which is of type PointCloud2 to a numpy array
        points_and_labels = np.asarray(
            list(read_points(msg, field_names=["x", "y", "z", "label"]))
        )

        # ["x", "y", "z"] point cloud coordinates
        points = points_and_labels[:, :3]

        # ["label"] of specific point cloud coordinate
        # Each label is either -1 for noise, or [0, n] where each
        # point is related to a specific cluster of points.
        labels = points_and_labels[:, 3:].flatten()
        for x in labels:
        	x = 1
        	
        dictionary = defaultdict(list)
        for key, value in zip(labels, points):
            dictionary[key].append(value)

        self.o3d_bboxs = []
        for key in dictionary:
            dictionary[key] = np.asarray(dictionary[key])

            # Must have more than 4 points (rows) to create a bounding box
            if dictionary[key].shape[0] >= 20:
                self.o3d_bboxs.append(
                    o3d.geometry.AxisAlignedBoundingBox().create_from_points(
                        o3d.utility.Vector3dVector(dictionary[key])
                    )
                )

        """
        vision_msgs/BoundingBox3DArray.msg
            std_msgs/Header header
            vision_msgs/BoundingBox3D[] boxes

        vision_msgs/BoundingBox3D.msg
            # A 3D bounding box that can be positioned and rotated about its center (6 DOF)
            # Dimensions of this box are in meters, and as such, it may be migrated to
            #   another package, such as geometry_msgs, in the future.

            # The 3D position and orientation of the bounding box center
            geometry_msgs/Pose center

            # The size of the bounding box, in meters, surrounding the object's center
            #   pose.
            geometry_msgs/Vector3 size
        
        geometry_msgs/Pose.msg
            # A representation of pose in free space, composed of position and orientation. 
            Point position
            Quaternion orientation

        geometry_msgs/Vector3.msg
            # This represents a vector in free space. 
            # It is only meant to represent a direction. Therefore, it does not
            # make sense to apply a translation to it (e.g., when applying a 
            # generic rigid transformation to a Vector3, tf2 will only apply the
            # rotation). If you want your data to be translatable too, use the
            # geometry_msgs/Point message instead.

            float64 x
            float64 y
            float64 z

        geometry_msgs/Point.msg
            # This contains the position of a point in free space
            float64 x
            float64 y
            float64 z

        geometry_msgs/Quaternion.msg
            # This represents an orientation in free space in quaternion form.
            float64 x
            float64 y
            float64 z
            float64 w
        """

        # TODO: Complete this loop to create a vision_msgs/BoundingBox3DArray

        self.bboxs = []
        for o3d_bbox in self.o3d_bboxs:
            # Items needed to create Point
            o3d_bbox_center = o3d_bbox.get_center()
            x1, y1, z1 = o3d_bbox_center[0], o3d_bbox_center[1], o3d_bbox_center[2]

            # Items needed to create Quarternion
            pyq_quaternion = pyq.Quaternion(
                matrix=o3d_bbox.get_oriented_bounding_box().R
            )
            x2, y2, z2, w2 = (
                pyq_quaternion[0],
                pyq_quaternion[1],
                pyq_quaternion[2],
                pyq_quaternion[3],
            )

            # Items needed to create Pose
            position = geometry_msgs.Point(x=x1, y=y1, z=z1)
            orientation = geometry_msgs.Quaternion(x=x2, y=y2, z=z2, w=w2)

            # Items needed to create Vector3
            o3d_bbox_extent = o3d_bbox.get_extent()
            x3, y3, z3 = o3d_bbox_extent[0], o3d_bbox_extent[1], o3d_bbox_extent[2]

            # Items needed to create BoundingBox3D
            center = geometry_msgs.Pose(position=position, orientation=orientation)
            size = geometry_msgs.Vector3(x=x3, y=y3, z=z3)

            # Items needed to create BoundingBox3DArray
            bbox = vision_msgs.BoundingBox3D(center=center, size=size)

            self.bboxs.append(bbox)
            
            # Classification:
            width = x3
            height = y3

            # classification = -1
            
            # Finding constraints for bounding box, since we only had center and size before
            xmin = x1 - x3/2
            xmax = x1 + x3/2
            ymin = y1 - y3/2
            ymax = y1 + y3/2
            zmin = z1 - z3/2
            zmax = z1 + z3/2
            				
            # TODO: Assign appropriate thresholds for classification
            # w and h are in meters
            # Human: [1.2m-2.1m x 0.3m-0.9m]
            # Traffic Lights: [1m-1.4m x 0.25-0.45m]
            # Street Signs: [1.0m-1.8m x 1.0m-1.8m]
            # Cars: [Average range in width: 1.4m-1.9m] [Average range in height: 1.3m-2.0m]
            
            if (1.2 < height < 2.1) and (0.3 < width < 0.9):
                for point in points:
                    if (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax) and (zmin <= point[2] <= zmax):
                        # reset label for each point to 1
            elif (1.0 < height < 1.4) and (0.25 < width < 0.45):
                for point in points:
                    if (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax) and (zmin <= point[2] <= zmax):
                        # reset label for each point to 2
            elif (1.0 < height < 1.8) and (1.0 < width < 1.8):
                for point in points:
                    if (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax) and (zmin <= point[2] <= zmax):
                        # reset label for each point to 3
            elif (1.3 < height < 2.0) and (1.4 < width < 1.9):
                for point in points:
                    if (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax) and (zmin <= point[2] <= zmax):
                        # reset label for each point to 4
            else:
                print("Other Classification")
            
#             label_list = list(dictionary.keys())
#             point_list = list(dictionary.values())
            
#             for point in points:
#             	xmin = x1 - x3/2
#             	xmax = x1 + x3/2
#             	ymin = y1 - y3/2
#             	ymax = y1 + y3/2
#             	zmin = z1 - z3/2
#             	zmax = z1 + z3/2
            	
#             	if(xmin <= point[0] <= xmax):	
#             		if( ymin <= point[1] <= ymax):	
#             			if(zmin <= point[2] <= zmax):
#             				if(classification >= 0):
#             					print(".")	
#             					for label, tmp_point in dictionary.items():
#             						if point == tmp_point:
#             							print(f"Label: {label}")
            					
            
        # Convert the numpy array to a open3d PointCloud
        self.o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        # Apply different colors to clusters
        max_label = labels.max()
        print(f"point cloud has {int(max_label + 1)} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels = 0] = 0
        colors[labels >= 0] = 0
        self.o3d_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # This is for visualization of the received point cloud.
        self.vis.clear_geometries()
        self.vis.add_geometry(self.o3d_pcd)

        # Draw bounding boxes
        for o3d_bbox in self.o3d_bboxs:
            self.vis.add_geometry(o3d_bbox)

        # Move viewpoint camera
        self.view_control.set_front(viewcontrol_front)
        self.view_control.set_lookat(viewcontrol_lookat)
        self.view_control.set_up(viewcontrol_up)
        self.view_control.set_zoom(viewcontrol_zoom)

        self.vis.poll_events()
        self.vis.update_renderer()

    def timer_callback(self):
        # TODO: Implement this callback function that will publish the final message with classifications
        header = std_msgs.Header(frame_id="map")

        # This function has be modified to add a "label" field to the message
        self.pcd_publisher.publish(
            vision_msgs.BoundingBox3DArray(header=header, boxes=self.bboxs)
        )


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
