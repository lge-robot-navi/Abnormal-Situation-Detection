# -*- coding: utf-8 -*- 
"""
@file       subscribe_img.py
@date       Aug 26, 2020
@author     Jiho Chang(changjh@etri.re.kr)
@brief      This file include ROS node for subscribing map images
 
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
import struct
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)

def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))


def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))

def create_figure(header):
    print("create figure with ",header)
    fig = plt.figure( 1 )
    ax = fig.add_subplot( 111 )
    ax.axis('off')
    im = ax.imshow(np.zeros((header[1],header[2],header[3])), cmap="gray") # Blank starting image
    fig.tight_layout()
    fig.show()
    im.axes.figure.canvas.draw()
    return im

first_msg = 0
im = 0
def on_message(client, userdata, msg):
    global first_msg, im
    header = struct.unpack('IHHH', msg.payload[:10])
    #print("msg header:", msg.payload[:10])
    if first_msg == 0:
        im = create_figure(header)
        first_msg = 1
    #print("refresh image")
    array = np.frombuffer(msg.payload[10:], np.uint8)
    arrayt = array.reshape(header[1],header[2],header[3])
    img = Image.fromarray(arrayt.astype('uint8'))
    #print("img : ",img.size, img.mode)
    im.set_data(img)
    im.axes.figure.canvas.draw()

# 새로운 클라이언트 생성
client = mqtt.Client()
# 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_subscribe(topic 구독),
# on_message(발행된 메세지가 들어왔을 때)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_subscribe = on_subscribe
client.on_message = on_message
# address : localhost, port: 1883 에 연결
client.connect('52.79.114.42', 1883)
# common topic 으로 메세지 발행
client.subscribe('/mams/ph/etri/map/height_probability', 1)
client.loop_forever()
