#!/usr/bin/env python

"""
@file		odometry_publisher.py
@date   	Mar 22, 2019
@author 	Kiin Na (kina4147@etri.re.kr)
@brief		This file include ROS node for publishing odometry. This publishes relative position between two frames (from_frame_id 			and to_frame_id) from tf.

Copyright (C) 2019  <Kiin Na>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import rospy
from nav_msgs.msg import Odometry
import tf2_ros
from tf.transformations import *

if __name__ == "__main__":
    
    rospy.init_node("odometry_publisher")

    from_frame_id = 'map'
    to_frame_id = 'mobile'
    period = 30
    pose_topic = '/robot_odom'
    if rospy.has_param('~from_frame_id'):
        from_frame_id = rospy.get_param('~from_frame_id')

    if rospy.has_param('~to_frame_id'):
        to_frame_id = rospy.get_param('~to_frame_id')

    if rospy.has_param('~period'):
        period = rospy.get_param('~period')

    if rospy.has_param('~pose_topic'):
	pose_topic = rospy.get_param('~pose_topic')

    pose_pub = rospy.Publisher(pose_topic, Odometry, queue_size=10)
    rate = rospy.Rate(period)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    while not rospy.is_shutdown():
        try:
            trans = tf_buffer.lookup_transform(from_frame_id, to_frame_id, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        odom = Odometry()
        odom.pose.pose.position.x = trans.transform.translation.x
        odom.pose.pose.position.y = trans.transform.translation.y
        odom.pose.pose.position.z = trans.transform.translation.z
        odom.pose.pose.orientation.x = trans.transform.rotation.x
        odom.pose.pose.orientation.y = trans.transform.rotation.y
        odom.pose.pose.orientation.z = trans.transform.rotation.z
        odom.pose.pose.orientation.w = trans.transform.rotation.w
        odom.header.stamp = rospy.Time()
        odom.header.frame_id = from_frame_id
        odom.child_frame_id = to_frame_id
        pose_pub.publish(odom)
        rate.sleep()
                          

