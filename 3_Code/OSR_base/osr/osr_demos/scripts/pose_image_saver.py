#!/usr/bin/env python

"""
@file		pose_image_saver.py
@date   	Mar 22, 2020
@author 	Kiin Na (kina4147@etri.re.kr)
@brief		this subscribes and save PoseImage message 

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
import cv2
from cv_bridge import CvBridge
import copy
from osr_msgs.msg import PoseImage
# from osr_msgs.msg import Abnormal

class MessageSubscriber:
    def __init__(self, sub_pose_image_topic):
        self.sub_pose = rospy.Subscriber(sub_pose_image_topic, PoseImage, self.pose_image_callback)
        self.pose_image_msg = None

    def pose_image_callback(self, msg):
        self.pose_image_msg = copy.deepcopy(msg)

    def get_messages(self):
        return self.pose_image_msg


def main():
    rospy.init_node("pose_image_saver")
    try:
        period = rospy.get_param('~period', 1)
        sub_pose_image_topic = rospy.get_param('~sub_pose_topic', "/pose_image")
	save_path = rospy.get_param('~save_path', '/home/parallels/data')
    except KeyError:
        rospy.logerr( "[ERROR] ROS parameters cannot be loaded." )

    # Subscriber Definition
    message_subscriber = MessageSubscriber(sub_pose_image_topic)
    if not os.path.isdir(save_path):
	print(save_path)
	rospy.loginfo("[INFO] Save directory is not existed, so it is created.")
	os.makedirs(save_path)

    bridge = CvBridge()
    rate = rospy.Rate(period)
    while not rospy.is_shutdown():
        pose_image_msg = message_subscriber.get_messages()
        if pose_image_msg is None:
            rospy.logerr("[ERROR] Data is not subscribed yet.")
            continue

        # Code Here #################################
	filename = '{:.2f}_{:.2f}_{:.3f}_{:.0f}.png'.format(pose_image_msg.pos_x, pose_image_msg.pos_y, pose_image_msg.rot_z, rospy.get_time())
	filepath = os.path.join(save_path, filename)
	cv_image = bridge.imgmsg_to_cv2(pose_image_msg.image, desired_encoding='passthrough')
	cv2.imwrite(filepath, cv_image)
        #############################################
		

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
