#!/usr/bin/env python

"""
@file		fixed_agent_pose_publisher.py
@date   	Mar 22, 2019
@author 	Kiin Na (kina4147@etri.re.kr)
@brief		This file include ROS node for publishing fixed agent pose. This loads the stored pose from text file and publish fixed agent pose in the type of nav_msgs Odometry.
 
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


import os
import rospy
from nav_msgs.msg import Odometry


if __name__ == "__main__":

    rospy.init_node("fixed_agent_pose_publisher")

    try:
        period = rospy.get_param('~period', 10)
        robot_frame_id = rospy.get_param('~robot_frame_id', 'fixed_01')
        map_frame_id = rospy.get_param('~map_frame_id', 'map')
        world_frame_id = rospy.get_param('~world_frame_id', 'world')
        pub_pose_topic = rospy.get_param('~pose_topic', '/pose')
        pose_filename = rospy.get_param('~pose_filename', robot_frame_id + '_to_map.txt')
    except KeyError:
        print "ros parameters are not set"


    pub_pose = rospy.Publisher(pub_pose_topic, Odometry, queue_size=10)
    rate = rospy.Rate(period)

    file_dir = os.path.dirname(__file__)
    file_path = os.path.join(file_dir, "../data", pose_filename)
    pos_x, pos_y, q = None, None, None

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            a2m = f.read().split()
            pos_x, pos_y, q_x, q_y, q_z, q_w = float(a2m[0]), float(a2m[1]), float(a2m[2]), float(a2m[3]), float(a2m[4]), float(a2m[5])
            q = [q_x, q_y, q_z, q_w]
    else:
        print ("[ERROR] There is no fixed agent pose file.")

    print(pos_x, pos_y, q)
    while not rospy.is_shutdown():
        odom = Odometry()
        odom.header.frame_id = map_frame_id
        odom.header.stamp = rospy.Time.now()
        odom.child_frame_id = robot_frame_id
        odom.pose.pose.position.x = pos_x
        odom.pose.pose.position.y = pos_y
        odom.pose.pose.position.z = 0

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = 0
        odom.twist.twist.linear.y = 0
        odom.twist.twist.angular.z = 0

        pub_pose.publish(odom)

        rate.sleep()

