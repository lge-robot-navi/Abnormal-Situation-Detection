#!/usr/bin/env python

import rospy
from grid_map_msgs.msg import GridMap
import numpy as np

layered_map = None
def sub_osr_map_cb(msg):
    global layered_map
    # print osr_map_msg
    # layered_map = None
    if len(msg.layers) > 0:
        layered_map = np.zeros((msg.data[0].layout.dim[1].size, msg.data[0].layout.dim[0].size, len(msg.layers)), 'float32')
    for idx, layer in enumerate(msg.layers):
        map_layer = msg.data[idx].data
        map_layer = np.reshape(map_layer, (msg.data[idx].layout.dim[1].size, msg.data[idx].layout.dim[0].size))
        layered_map[:, :, idx] = map_layer


def main():
    global layered_map
    rospy.init_node('osr_map_receiver')

    try:
        period = rospy.get_param('~period', 10)
        map_topic = rospy.get_param('~map_topic', '/mobile_agent_map')
    except KeyError:
        print "ros parameters are not set"

    sub_osr_map = rospy.Subscriber(map_topic, GridMap, sub_osr_map_cb)
    rate = rospy.Rate(period)
    while not rospy.is_shutdown():
        if layered_map is None:
            continue
        current_layered_map = layered_map
        print (current_layered_map.shape)
        rate.sleep()


if __name__ == "__main__":
    main()

