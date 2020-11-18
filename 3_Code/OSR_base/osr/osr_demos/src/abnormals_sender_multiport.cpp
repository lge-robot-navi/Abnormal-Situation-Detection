/*
 * image_sender.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
// Boost
#include <boost/asio.hpp>

// OSR
#include "osr_comm/comm.h"
#include "osr_comm/packets.h"
#include "osr_comm/parsers.h"
#include "osr_msgs/Abnormal.h"

class MessageSubscriber{
public:
	MessageSubscriber(ros::NodeHandle _nh, std::string _abnormal_pedestrian_topic ="/abnormal_pedestrian_topic", std::string _abnormal_sound_topic ="/abnormal_sound_topic")
	: abnormal_pedestrian_report_id(0), abnormal_sound_report_id(0) {
		sub_abnormal_pedestrian = _nh.subscribe(_abnormal_pedestrian_topic, 1, &MessageSubscriber::subscribe_abnormal_pedestrian_cb, this);
		sub_abnormal_sound = _nh.subscribe(_abnormal_sound_topic, 1, &MessageSubscriber::subscribe_abnormal_sound_cb, this);
	};
	~MessageSubscriber(){};
private:
	ros::Subscriber sub_abnormal_pedestrian;
	ros::Subscriber sub_abnormal_sound;
	std::vector<osr_msgs::Abnormal> pedestrian_abnormals;
	std::vector<osr_msgs::Abnormal> sound_abnormals;
	int abnormal_pedestrian_report_id;
	int abnormal_sound_report_id;

public:
	void subscribe_abnormal_pedestrian_cb(const osr_msgs::Abnormal::ConstPtr& _msg){
		if(abnormal_pedestrian_report_id != _msg->report_id){
			pedestrian_abnormals.push_back(*_msg);
			abnormal_pedestrian_report_id = _msg->report_id;

		}
	}
	void subscribe_abnormal_sound_cb(const osr_msgs::Abnormal::ConstPtr& _msg){
		if(abnormal_sound_report_id != _msg->report_id){
			sound_abnormals.push_back(*_msg);
			abnormal_sound_report_id = _msg->report_id;
		}

	}
	bool get_abnormals(std::vector<osr_msgs::Abnormal>& _abnormals){
		if(pedestrian_abnormals.empty() && sound_abnormals.empty()){
			return false;
		}
		else{
			_abnormals.insert(_abnormals.end(), pedestrian_abnormals.begin(), pedestrian_abnormals.end());
			_abnormals.insert(_abnormals.end(), sound_abnormals.begin(), sound_abnormals.end());
			pedestrian_abnormals.clear();
			sound_abnormals.clear();
			return true;
		}
	}
};
int main(int argc, char **argv) {
	// Node Generation
	ros::init(argc, argv, "abnormals_sender_multiport");

	ros::NodeHandle nh("~");

	float period;
	nh.param<float>("period", period, 10);
	std::string robot_frame_id;
	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
	std::string sub_abnormal_pedestrian_topic, sub_abnormal_sound_topic;
	nh.param<std::string>("abnormal_pedestrian_topic", sub_abnormal_pedestrian_topic, "/osr/abnormal_pedestrian");
	nh.param<std::string>("abnormal_sound_topic", sub_abnormal_sound_topic, "/osr/abnormal_sound");
	std::string ip_addr;
	nh.param<std::string>("udp_ip_address", ip_addr, "127.0.0.1");
	int local_id;
	int data_type;
	int robot_id;
	nh.param<int>("local_id", local_id, 8888);
	nh.param<int>("data_type", data_type, 8888);
	nh.param<int>("robot_id", robot_id, 1);
	unsigned int port = osr::get_port(local_id, data_type, robot_id);

	ros::Rate rate(period);
	MessageSubscriber msg_sub(nh, sub_abnormal_pedestrian_topic, sub_abnormal_sound_topic);
	boost::asio::io_service io_srv;
	osr::udp_sender<osr::abnormals_packet> udp_sender(io_srv, ip_addr, port);

	int seq = 0;
	while (ros::ok()){
		rate.reset();
		ros::spinOnce();
		std::vector<osr_msgs::Abnormal> abnormals;
		if(!msg_sub.get_abnormals(abnormals)){
			rate.sleep();
			continue;
		}
		try{
			osr::abnormals_packet abs_pkt;
			osr::abnormals_to_packet(robot_id, robot_frame_id, abnormals, abs_pkt);
			size_t byte_transferred = udp_sender.send(abs_pkt);
			ROS_INFO_STREAM("agent : [ "<< byte_transferred << " ] bytes sent to server.");
			abs_pkt.clear();
			seq++;
		}
		catch (const std::exception& ex) {
	        ROS_ERROR("agent : %s", ex.what());
		}

		rate.sleep();
	}

	return 0;
}
