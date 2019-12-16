#!/usr/bin/env python  

"""
@file		tf_broadcaster.py
@date   	Mar 22, 2019
@author 	Kiin Na (kina4147@etri.re.kr)
@brief		This file include ROS node for adding transform to TF. This subscribes odometry messages and sends transform between 			child_frame_id and frame_id of Odometry.

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
import tf
from nav_msgs.msg import Odometry 

def odom_cb(msg):
    br = tf.TransformBroadcaster()
    br.sendTransform((msg.pose.pose.position.x, msg.pose.pose.position.y, 0), (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w), msg.header.stamp + rospy.Duration(5.0), msg.child_frame_id, msg.header.frame_id)
    


if __name__ == '__main__':
    rospy.init_node('tf_broadcaster')
    rospy.Subscriber('/robot_odom', Odometry, odom_cb)
    rospy.spin()
