"""
@file       mapimg_publisher.py
@date       Aug 26, 2020
@author     Jiho Chang(changjh@etri.re.kr)
@brief      This file include ROS node for publishing image of environment map
 
Copyright (C) 2020  <Jiho Chang>

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

import paho.mqtt.client as mqtt
import time
import numpy as np
import struct
from PIL import Image

import rospy
from sensor_msgs.msg import Image

def mqtt_bridge():
    rospy.init_node('image_listener')
    image_topic = "/osr_map_report1"
    subscriber_high = rospy.Subscriber(image_topic, Image, image_callback_search)
    image_topic = "/osr_map_report2"
    subscriber_obj = rospy.Subscriber(image_topic, Image, image_callback_height)
    image_topic = "/osr_map_report3"
    subscriber_temp = rospy.Subscriber(image_topic, Image, image_callback_temp)
    image_topic = "/osr_map_report4"
    subscriber_temp = rospy.Subscriber(image_topic, Image, image_callback_objprob)
    image_topic = "/osr_map_report5"
    subscriber_temp = rospy.Subscriber(image_topic, Image, image_callback_heightprob)
    # register shutdown callback and spin
    rospy.on_shutdown(client.disconnect)
    rospy.on_shutdown(client.loop_stop)
    rospy.spin()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)

def close_mqtt():
    client.loop_stop()
    client.disconnect()

def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))

def on_publish(client, userdata, mid):
    print("In on_pub callback mid= ", mid)

def image_callback_search(msg):
    print("Received an search image!")
    msg_header = struct.pack('IHHH', int(time.time()),msg.height, msg.width, 3)
    #print("msg header:", msg_header)
    img_array = np.fromstring(msg.data,np.uint8)
    byteArr = bytearray(img_array)
    client.publish("/mams/ph/etri/map/search",msg_header+byteArr,0)  

def image_callback_height(msg):
    print("Received an height image!")
    msg_header = struct.pack('IHHH', int(time.time()),msg.height, msg.width, 3)
    img_array = np.fromstring(msg.data,np.uint8)
    byteArr = bytearray(img_array)
    client.publish("/mams/ph/etri/map/height",msg_header+byteArr,0)  

def image_callback_temp(msg):
    print("Received an temprture image!")
    msg_header = struct.pack('IHHH', int(time.time()),msg.height, msg.width, 3)
    img_array = np.fromstring(msg.data,np.uint8)
    byteArr = bytearray(img_array)
    client.publish("/mams/ph/etri/map/temperature",msg_header+byteArr,0)  

def image_callback_objprob(msg):
    print("Received an object probability image!")
    msg_header = struct.pack('IHHH', int(time.time()),msg.height, msg.width, 1)
    img_array = np.fromstring(msg.data,np.uint8)
    byteArr = bytearray(img_array)
    client.publish("/mams/ph/etri/map/object_probability",msg_header+byteArr,0) 

def image_callback_heightprob(msg):
    print("Received an height probability image!")
    msg_header = struct.pack('IHHH', int(time.time()),msg.height, msg.width, 1)
    img_array = np.fromstring(msg.data,np.uint8)
    byteArr = bytearray(img_array)
    client.publish("/mams/ph/etri/map/height_probability",msg_header+byteArr,0)   

# configure and connect to MQTT broker
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_publish = on_publish
client.connect('220.81.76.111', 1883)
# configure and connect to MQTT broker
client.loop_start()

def main():
    mqtt_bridge()
    close_mqtt()

if __name__ == '__main__':
    main()

