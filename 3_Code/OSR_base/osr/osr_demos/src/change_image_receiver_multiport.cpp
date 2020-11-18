/*
 * image_receiver.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

#include <thread>
#include <memory>
#include <queue>

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


	ros::init(argc, argv, "change_image_receiver_multiport");
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

	std::string pub_pose_image_topic;
	nh.param < std::string > ("pose_image_topic", pub_pose_image_topic, "/pose_image");
	ros::Publisher pub_pose_image = nh.advertise< osr_msgs::PoseImage >(pub_pose_image_topic, 1, true);
	ros::Publisher pub_test_image = nh.advertise< sensor_msgs::Image >("/test_change_image", 1, true);


	std::vector<unsigned int> ports = osr::get_ports(local_id, data_type, robot_ids);

	boost::asio::io_service io_srv;
	std::vector<std::shared_ptr<osr::udp_receiver<osr::image_packet>>> receivers;
	boost::thread_group tgroup;
	for (std::vector<unsigned int>::iterator it = ports.begin(); it != ports.end(); ++it){
		receivers.push_back(std::make_shared<osr::udp_receiver<osr::image_packet>>(io_srv, *it));
		tgroup.add_thread(new boost::thread(boost::bind(&osr::udp_receiver<osr::image_packet>::receive_loop, receivers.back())));
	}

	std::queue<osr_msgs::PoseImage> pose_images;
	int max_queue_size = 10;
	ros::Rate rate(period);
	int seq = 0;
    while(ros::ok()){
    	rate.reset();
        std::vector<osr::image_packet> packets;
        for (auto receiver_ptr : receivers){
        	osr::image_packet packet;
			if(!receiver_ptr->get_new_message(packet)){
				continue;
			}
			packets.push_back(packet);
        }
    	for(std::vector<osr::image_packet>::iterator it = packets.begin(); it != packets.end(); ++it){
			int id;
			std::string agent_id;
			float pos_x, pos_y, rot_z;
			cv::Mat received_image;
	    	osr::packet_to_image(*it, id, agent_id, pos_x, pos_y, rot_z, received_image);
	    	if(pose_images.size() >= max_queue_size){
	    		pose_images.pop();
	    	}
	    	osr_msgs::PoseImage pose_image;
	    	pose_image.pos_x = pos_x;
	    	pose_image.pos_y = pos_y;
	    	pose_image.rot_z = rot_z;
			cv_bridge::CvImagePtr image_bridge(new cv_bridge::CvImage());
			image_bridge->header.stamp = ros::Time::now();
			image_bridge->encoding = sensor_msgs::image_encodings::TYPE_8UC3;
			image_bridge->image = received_image;
	    	image_bridge->toImageMsg(pose_image.image);
	    	pose_images.push(pose_image);
    	}

    	if(!pose_images.empty()){
    		osr_msgs::PoseImage out_msg = pose_images.front();
			if (pub_test_image.getNumSubscribers() > 0) {
				pub_test_image.publish(out_msg.image);
			}
    		if (pub_pose_image.getNumSubscribers() > 0) {
    			pub_pose_image.publish(out_msg);
    		}
			pose_images.pop();
    	}
		rate.sleep();
    }

}
