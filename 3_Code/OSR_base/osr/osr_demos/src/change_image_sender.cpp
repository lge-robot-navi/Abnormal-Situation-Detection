/*
 * image_sender.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

// ROS
#include <ros/ros.h>
#include <tf/tf.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
// Boost
#include <boost/asio.hpp>

// OSR
#include "osr_comm/comm.h"
#include "osr_comm/packets.h"
#include "osr_comm/parsers.h"


class MessageSubscriber{
public:
	MessageSubscriber(ros::NodeHandle _nh, std::string _image_topic ="/image", std::string _pose_topic ="/pose"){
		sub_image = _nh.subscribe(_image_topic, 1, &MessageSubscriber::subscribe_image_cb, this);
		sub_pose = _nh.subscribe(_pose_topic, 1, &MessageSubscriber::subscribe_pose_cb, this);
		pose_on = false;
		image_on = false;
	};
	~MessageSubscriber(){};
private:
	ros::Subscriber sub_image, sub_pose;
	cv::Mat image;
	nav_msgs::Odometry pose;
	bool pose_on, image_on;

public:
	void subscribe_image_cb(const sensor_msgs::Image::ConstPtr& _msg){
		image_on = true;
		cv_bridge::CvImagePtr img_ptr;
		try
		{
			img_ptr = cv_bridge::toCvCopy(_msg);
			img_ptr->image.convertTo(image, CV_8UC3);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception @ callbackDepth(): %s", e.what());
			return;
		}
	}
	void subscribe_pose_cb(const nav_msgs::Odometry::ConstPtr& _msg){
		pose_on = true;
		pose = *_msg;
	}
	cv::Mat get_image(){return image;}
	nav_msgs::Odometry get_pose(){return pose;}
	bool is_all_on(){
		if (image_on && pose_on){
			return true;
		}
		else{
			return false;
		}
	}
};
int main(int argc, char **argv) {
	// Node Generation
	ros::init(argc, argv, "change_image_sender");

	ros::NodeHandle nh("~");
	float period;
	int robot_id;
	std::string robot_frame_id;
	std::string sub_image_topic, sub_pose_topic;
	std::string ip_addr;
	int port;
	int image_height, image_width;
	nh.param<float>("period", period, 10);
	nh.param<int>("robot_id", robot_id, 1);
	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
	nh.param<std::string>("image_topic", sub_image_topic, "/camera/color/image_raw");
	nh.param<std::string>("pose_topic", sub_pose_topic, "/robot_odom");
	nh.param<std::string>("udp_ip_address", ip_addr, "127.0.0.1");
	nh.param<int>("udp_port", port, 8888);
	nh.param<int>("image_height", image_height, 112);
	nh.param<int>("image_width", image_width, 112);


	ros::Rate rate(period);
	MessageSubscriber msg_sub(nh, sub_image_topic, sub_pose_topic);
	boost::asio::io_service io_srv;
	osr::udp_sender<osr::image_packet> udp_sender(io_srv, ip_addr, static_cast<unsigned int>(port));

//	pub_image_topic = "/pub_image";
//	ros::Publisher pub_image = nh.advertise<sensor_msgs::Image>(pub_image_topic, 1);

	while (ros::ok()){
		rate.reset();
		ros::spinOnce();
		if(!msg_sub.is_all_on()){
			ROS_ERROR("[ERROR] data is not gathered yet.");
			rate.sleep();
			continue;
		}

		cv::Mat img = msg_sub.get_image();
		nav_msgs::Odometry pose = msg_sub.get_pose();
	    tf::Quaternion q(
			pose.pose.pose.orientation.x,
			pose.pose.pose.orientation.y,
			pose.pose.pose.orientation.z,
			pose.pose.pose.orientation.w);
	    tf::Matrix3x3 m(q);
	    double roll, pitch, yaw;
	    m.getRPY(roll, pitch, yaw);
		float pos_x = pose.pose.pose.position.x;
		float pos_y = pose.pose.pose.position.y;
		float rot_z = yaw;


		cv::resize(img, img, cv::Size(image_width, image_height));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

		try{
			osr::image_packet packet;
			osr::image_to_packet(robot_id, robot_frame_id, pos_x, pos_y, rot_z, img, packet);
			size_t byte_transferred = udp_sender.send(packet);
			ROS_INFO_STREAM("agent : [ "<< byte_transferred << " ] bytes sent to server.");
			packet.clear();
		}
		catch (const std::exception& ex) {
	        ROS_ERROR("agent : %s", ex.what());
		}

//		// map to publish
//		if (pub_image.getNumSubscribers() > 0) {
//			cv_bridge::CvImagePtr image_bridge(new cv_bridge::CvImage());
//			image_bridge->header.frame_id;
//			image_bridge->header.stamp;
//			image_bridge->encoding = sensor_msgs::image_encodings::TYPE_8UC3;
//			image_bridge->image = get_img;
//			pub_image.publish(image_bridge->toImageMsg());
//		}
		rate.sleep();
	}

	return 0;
}
