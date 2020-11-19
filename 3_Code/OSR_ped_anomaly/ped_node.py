#!/usr/bin/env python3

""" #
@file		ped_node.py
@date   	Aug 13, 2020
@author 	Jin-ha Lee (jh_lee@etri.re.kr)
@brief		This file include ROS node for publishing fixed agent anomaly score for pedestrain.

Copyright (C) 2019  <Jin-ha Lee>

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
import copy
from nav_msgs.msg import Odometry
from osr_msgs.msg import Abnormal, Tracks
from sensor_msgs.msg import Image
from model_anomaly import test_patches, init_ped_anoamly
import matplotlib.pyplot as plt
import argparse, sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from cv_bridge import CvBridge, CvBridgeError

class MessageSubscriber:  # Merge topics here
    def __init__(self, sub_pose_topic):
        self.sub_pose = rospy.Subscriber(sub_pose_topic, Odometry, self.pose_callback)
        self.pose_msg = None

    def pose_callback(self, msg):
        self.pose_msg = copy.deepcopy(msg)

    def get_messages(self):
        return self.pose_msg

tracks = None
image = None

def tracks_cb(msg):
    """
    :param msg: Ros topic(bounding box information)
    :return: Extracted bounding box
    """
    global tracks
    # print('tracks_cb')
    tmp = []
    for track in msg.tracks:
        x = track.bbox_pose.x
        y = track.bbox_pose.y
        w = track.bbox_pose.width
        h = track.bbox_pose.height
        tmp.append([x, y, w, h])
    tracks = tmp
def image_cb(msg):
    """
    :param msg: Ros topic(Full-frame image from agent)
    :return: Extracted image
    """
    global image
    # print('image_cb')
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    # (rows, cols, channels) = cv_image.shape
    image = cv_image
def crop_from_topic(tracks, image):
    patches = []
    # cv2.imwrite('./data/frame/img_{0}.jpg'.format(rospy.get_time()), image)  # save full frame img
    for box in tracks:
        crop = image[box[1]:box[1]+box[3], box[0]:box[0] + box[2]].copy()  # crop patch
        patches.append(crop)
    return patches

def opt_parser_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', default=512, type=int, help='latent space size')
    parser.add_argument('--start_number', default=True, type=bool, help='Flag to load model')
    parser.add_argument('--input_size', default=64, type=int, help='default 64x64')
    parser.add_argument('--input_channel', default=3, type=int, help='default RGB channel')
    parser.add_argument('--load', default=True, type=bool, help='Load model or not')
    parser.add_argument('--resume', default=True, type=bool, help='Load model or not')
    opt = parser.parse_args()

    return opt

def main():
    rospy.init_node("ped_node")
    try:
        period = rospy.get_param('~period', 10)
        robot_id = rospy.get_param('~robot_id', 1)  # Fix123 ~ 789
        pub_abnormal_topic = rospy.get_param('~pub_abnormal_topic', '/osr/abnormal_pedestrian')
        sub_pose_topic = rospy.get_param('~sub_pose_topic', "/robot_odom")
    except KeyError:
        rospy.logerr("[ERROR] ROS parameters cannot be loaded.")

    # Publisher Definition
    pub_abnormal = rospy.Publisher(pub_abnormal_topic, Abnormal, queue_size=1)
    # Subscriber Definition
    message_subscriber = MessageSubscriber(sub_pose_topic)
    sub_track = rospy.Subscriber("osr/tracks", Tracks, tracks_cb)
    sub_image = rospy.Subscriber("osr/image_color", Image, image_cb)

    # ped_anomaly module ###########
    opt = opt_parser_test()  # setup parameters
    model = init_ped_anoamly(opt)  # init anomaly detection model
    cur_status = 0
    max_score = 0
    ################################

    rate = rospy.Rate(period)
    while not rospy.is_shutdown():
        pose_msg = message_subscriber.get_messages()
        report_id = int(rospy.get_time())
        # if pose_msg is None:
        #     rospy.logerr("[ERROR] Data is not subscribed yet.")
        #     continue

        # Code Here #################################
        if tracks:
            # print("Got sth!")
            patches = crop_from_topic(tracks, image)
            max_score = test_patches(model, opt, patches)
            print("Report_id:{0} Score: {1}".format(report_id, max_score))  #TODO: why occationally shows score?
            if max_score > 0.6:
                cur_status = 2
            elif max_score > 0.3:
                cur_status = 1
            else:
                cur_status = 0
        #############################################

        abnormal = Abnormal()
        abnormal.agent_id = robot_id
        abnormal.report_id = report_id
        abnormal.pos_x = 0#pose_msg.pose.pose.position.x
        abnormal.pos_y = 0#pose_msg.pose.pose.position.y

        # input the result from this module
        abnormal.status = cur_status  # Abnormal Status (0: NORMAL, 1: CAUTION, 2: ALERT)
        abnormal.type = 4
        abnormal.detail = 0  # FOR Unfamiliar Pedestrian (type = 4), 7: Unfamiliar Pedestrian
        abnormal.score = int(max_score * 255)
        pub_abnormal.publish(abnormal)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass