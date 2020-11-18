/*
 * image_receiver.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

#include <thread>
#include <memory>

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


	ros::init(argc, argv, "abnormals_receiver_multiport");
	ros::NodeHandle nh("~");
	int period;
	int port;
	int local_id;
	int data_type;
	std::vector<int> robot_ids;
	nh.param<int>("period", period, 10);
	nh.param<int>("local_id", local_id, 8888);
	nh.param<int>("data_type", data_type, 8888);
	nh.param("robot_ids", robot_ids, std::vector<int>());

	std::vector<unsigned int> ports = osr::get_ports(local_id, data_type, robot_ids);

	std::string pub_abnormals_topic;
	nh.param < std::string > ("pub_abnormals_topic", pub_abnormals_topic, "/osr/agent_abnormals");

	ros::Publisher pub_received_abnormals = nh.advertise< osr_msgs::Abnormals >(pub_abnormals_topic, 1, true);

	boost::asio::io_service io_srv;
	std::vector<std::shared_ptr<osr::udp_receiver<osr::abnormals_packet>>> receivers;
	boost::thread_group tgroup;
	for (std::vector<unsigned int>::iterator it = ports.begin(); it != ports.end(); ++it){
		receivers.push_back(std::make_shared<osr::udp_receiver<osr::abnormals_packet>>(io_srv, *it));
		tgroup.add_thread(new boost::thread(boost::bind(&osr::udp_receiver<osr::abnormals_packet>::receive_loop, receivers.back())));
	}

	ros::Rate rate(period);
	int seq = 0;
    while(ros::ok()){
    	rate.reset();
    	osr::abnormals_packet packet;

        std::vector<osr::abnormals_packet> packets;
        for (auto receiver_ptr : receivers){
        	osr::abnormals_packet packet;
			if(!receiver_ptr->get_new_message(packet)){
				continue;
			}
			packets.push_back(packet);
        }
        if(packets.empty()){
    		rate.sleep();
        	continue;
        }

    	std::vector<osr_msgs::Abnormal> received_abnormals;
    	for(std::vector<osr::abnormals_packet>::iterator it = packets.begin(); it != packets.end(); ++it){
			int id;
			std::string agent_id;
			osr::packet_to_abnormals(*it, id, agent_id, received_abnormals);
    	}

    	// map to publish
		if (pub_received_abnormals.getNumSubscribers() > 0) {
			osr_msgs::Abnormals abs_msg;
			abs_msg.header.stamp = ros::Time::now();
			abs_msg.header.seq = seq++;
			abs_msg.header.frame_id;
			abs_msg.abnormals = received_abnormals;
			pub_received_abnormals.publish(abs_msg);
		}
		rate.sleep();
    }
}
