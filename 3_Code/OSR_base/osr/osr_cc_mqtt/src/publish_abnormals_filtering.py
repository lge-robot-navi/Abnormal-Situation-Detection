# -*- coding: utf-8 -*- 

"""
@file       publish_abnormals_filtering.py
@date       Aug 26, 2020
@author     Jiho Chang(changjh@etri.re.kr)
@brief      This file include ROS node for publishing predefiend abnormals with filtering
 
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
import json
import time
import numpy as np
from random import *

import rospy
from osr_msgs.msg import Abnormals, Abnormal

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

ringbf_caution = RingBuffer(100)
ringbf_warning = RingBuffer(100)

h11 = -0.111429818
h12 = -0.0853415504
h13 = 36.1190186
h21 = -0.399244487
h22 = -0.305760771
h23 = 129.415985
h31 = -0.00308493176
h32 = -0.00236267666
h33 = 1.0

def mqtt_bridge():
    rospy.init_node('abnormals_listener')
    abnormal_map_topic = "/osr_map_abnormals"
    subscriber_high = rospy.Subscriber(abnormal_map_topic, Abnormals, abnormals_callback_map)
    abnormal_agent_topic = "/osr/agent_abnormals"
    subscriber_high = rospy.Subscriber(abnormal_agent_topic, Abnormals, abnormals_callback_agent)
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

def isthere(abnormals_bf, abnormal_el):
    for buf_ab in abnormals_bf[::-1]:
        if (abs(buf_ab.report_id - abnormal_el.report_id) < 60):
            if (buf_ab.status == abnormal_el.status) and (buf_ab.agent_id == abnormal_el.agent_id) and (buf_ab.type == abnormal_el.type) and (buf_ab.detail == abnormal_el.detail):
                tmp_distance = np.linalg.norm(np.array((buf_ab.pos_x, buf_ab.pos_y))-np.array((abnormal_el.pos_x, abnormal_el.pos_y)))
                if (tmp_distance < 5):
                    return 1
    return 0

def abnormals_callback_map(msg):
    global stack_caution, stack_warning
    print("Received an map abnormals!")
    total_status = 0
    tmp_pubmsg = msg.abnormals[0]
    #Detect for higher abnormals
    for lst in msg.abnormals:
        if (lst.status != 0) and (lst.status > total_status) and (not isthere(ringbf_warning.get(), lst)):            
            total_status = lst.status
            tmp_pubmsg = lst
            ringbf_warning.append(lst)
    #print(ringbf_warning.data)

    if total_status == 0:
        client.publish('/mams/ph/etri/event', json.dumps({"eventid": tmp_pubmsg.report_id, "status":total_status, "agentid":0, "abnormalType":0,"abnormalDetail":0,"posx":0.0, "posy":0.0,"fbNeed":"N"}), 1)
    else:
        posY = -tmp_pubmsg.pos_x + 85
        posX = -tmp_pubmsg.pos_y + 70
        #gps_posX = (h11*posX + h12*posY + h13) / (h31*posX + h32*posY + h33)
        #gps_posY = (h21*posX + h22*posY + h23) / (h31*posX + h32*posY + h33)
        #print(gps_posX, gps_posY)
        if tmp_pubmsg.type == 1:
            client.publish('/mams/ph/etri/event', json.dumps({"eventid": tmp_pubmsg.report_id, "status":total_status, "agentid":tmp_pubmsg.agent_id, "abnormalType":tmp_pubmsg.type,"abnormalDetail":tmp_pubmsg.detail,"posx":posX, "posy":posY,"fbNeed":"Y"}), 1)
        else:
            client.publish('/mams/ph/etri/event', json.dumps({"eventid": tmp_pubmsg.report_id, "status":total_status, "agentid":tmp_pubmsg.agent_id, "abnormalType":tmp_pubmsg.type,"abnormalDetail":tmp_pubmsg.detail,"posx":posX, "posy":posY,"fbNeed":"N"}), 1)

def abnormals_callback_agent(msg):
    print("Received an agent abnormals!")
    total_status = 0
    tmp_pubmsg = msg.abnormals[0]
    for lst in msg.abnormals:
        if (lst.status != 0) and (lst.status > total_status) and (not isthere(ringbf_warning.get(), lst)):
            total_status = lst.status
            tmp_pubmsg = lst
            ringbf_warning.append(lst)
    if total_status != 0:
        posX = -tmp_pubmsg.pos_x
        posY = -tmp_pubmsg.pos_y
        client.publish('/mams/ph/etri/event', json.dumps({"eventid": tmp_pubmsg.report_id, "status":total_status, "agentid":tmp_pubmsg.agent_id, "abnormalType":tmp_pubmsg.type,"abnormalDetail":tmp_pubmsg.detail,"posx":tmp_pubmsg.pos_x, "posy":tmp_pubmsg.pos_y,"fbNeed":"N"}), 1)

# configure and connect to MQTT broker
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_publish = on_publish
client.connect('220.81.76.111', 1883)
#client.connect('localhost', 1883)
# configure and connect to MQTT broker
client.loop_start()

def main():
    mqtt_bridge()
    close_mqtt()

if __name__ == '__main__':
    main()

