/*
 * image_receiver.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

#include <thread>

// ROS
#include <ros/ros.h>

// Boost
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>

#include "osr_comm/comm.h"
#include "osr_comm/packets.h"
#include "osr_comm/parsers.h"
#include "osr_msgs/Abnormals.h"

int main(int argc, char **argv) {


	ros::init(argc, argv, "abnormals_receiver");
	ros::NodeHandle nh("~");
	int period;
	int port;
	std::string pub_abnormals_topic;
	nh.param<int>("period", period, 10);
	nh.param<int>("udp_port", port, 8888);
	nh.param < std::string > ("pub_abnormals_topic", pub_abnormals_topic, "/osr/agent_abnormals");

	ros::Publisher pub_received_abnormals = nh.advertise< osr_msgs::Abnormals >(pub_abnormals_topic, 1, true);

	boost::asio::io_service io_srv;
	osr::udp_receiver<osr::abnormals_packet> receiver(io_srv, static_cast<unsigned int>(port));
	boost::thread receive_thread(boost::bind(&osr::udp_receiver<osr::abnormals_packet>::receive_loop, &receiver));

	ros::Rate rate(period);
	int seq = 0;
    while(ros::ok()){
    	rate.reset();
    	osr::abnormals_packet packet;
    	int id;
    	std::string agent_id;
    	if(!receiver.get_new_message(packet)){
    		rate.sleep();
    		continue;
    	}
    	std::vector<osr_msgs::Abnormal> received_abnormals;
    	osr::packet_to_abnormals(packet, id, agent_id, received_abnormals);
		// map to publish
		if (pub_received_abnormals.getNumSubscribers() > 0) {
			osr_msgs::Abnormals abs_msg;
			abs_msg.header.frame_id;
			abs_msg.header.stamp = ros::Time::now();
			abs_msg.header.seq = seq++;
			abs_msg.abnormals = received_abnormals;
			pub_received_abnormals.publish(abs_msg);
		}
		rate.sleep();
    }
}
