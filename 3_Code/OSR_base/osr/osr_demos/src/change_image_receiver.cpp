/*
 * image_receiver.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

#include <thread>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// Boost
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>

#include "osr_comm/comm.h"
#include "osr_comm/packets.h"
#include "osr_comm/parsers.h"

#include "osr_msgs/PoseImage.h"

int main(int argc, char **argv) {


	ros::init(argc, argv, "change_image_receiver");
	ros::NodeHandle nh("~");
	int period;
	int port;
	nh.param<int>("period", period, 10);
	nh.param<int>("udp_port", port, 8888);

	std::string pub_pose_image_topic;
	nh.param < std::string > ("pose_image_topic", pub_pose_image_topic, "/pose_image");
	ros::Publisher pub_pose_image = nh.advertise< osr_msgs::PoseImage >(pub_pose_image_topic, 1, true);
	ros::Publisher pub_test_image = nh.advertise< sensor_msgs::Image >("/test_change_image", 1, true);

	boost::asio::io_service io_srv;
	osr::udp_receiver<osr::image_packet> receiver(io_srv, static_cast<unsigned int>(port));
	boost::thread receive_thread(boost::bind(&osr::udp_receiver<osr::image_packet>::receive_loop, &receiver));

	ros::Rate rate(period);
	int seq = 0;
    while(ros::ok()){
    	rate.reset();
    	osr::image_packet packet;
    	if(!receiver.get_new_message(packet)){
    		rate.sleep();
    		continue;
    	}
    	int id;
    	std::string agent_id;
    	float pos_x, pos_y, rot_z;
    	cv::Mat received_image;
    	osr::packet_to_image(packet, id, agent_id, pos_x, pos_y, rot_z, received_image);
    	osr_msgs::PoseImage pose_image;
    	pose_image.pos_x = pos_x;
    	pose_image.pos_y = pos_y;
    	pose_image.rot_z = rot_z;
		cv_bridge::CvImagePtr image_bridge(new cv_bridge::CvImage());
		image_bridge->header.stamp = ros::Time::now();
		image_bridge->encoding = sensor_msgs::image_encodings::TYPE_8UC3;
		image_bridge->image = received_image;
    	image_bridge->toImageMsg(pose_image.image);

		// map to publish
		if (pub_test_image.getNumSubscribers() > 0) {
			pub_test_image.publish(pose_image.image);
		}
		if (pub_pose_image.getNumSubscribers() > 0) {
			pub_pose_image.publish(pose_image);
		}

		rate.sleep();
    }

}
