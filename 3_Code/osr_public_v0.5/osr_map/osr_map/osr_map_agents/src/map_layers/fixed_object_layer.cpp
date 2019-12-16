/**
 * @file		fixed_object_layer.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in fixed_object_layer class.
 * @remark
 * @warning
 *
 * Copyright (C) 2019  <Kiin Na>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */



#include "osr_map_agents/map_layers/fixed_object_layer.h"


osr_map::fixed_object_layer::fixed_object_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id)
	: nh_(_nh), period_(_period), robot_id_(_robot_id), robot_frame_id_(_robot_frame_id), map_frame_id_(_map_frame_id)
{
	curr_time_ = prev_time_ = ros::Time::now();
	is_in_tracks_ = is_in_pose_ = false;
};
void osr_map::fixed_object_layer::set_comm(std::string _pose_topic, std::string _layer_topic, std::string _object_topic)
{
  	pub_layer_ = nh_.advertise<grid_map_msgs::GridMap>(_layer_topic, 1, true);
	sub_object_ = nh_.subscribe(_object_topic, 1, &osr_map::fixed_object_layer::subscribe_object_callback, this);
	sub_pose_ = nh_.subscribe(_pose_topic, 1, &osr_map::fixed_object_layer::subscribe_pose_callback, this);
}
void osr_map::fixed_object_layer::set_map_param(float _length_x, float _length_y, float _resolution, bool _multi_project_on)
{
	length_x_ = _length_x;
	length_y_ = _length_y;
	resolution_ = _resolution;
	multi_project_on_ = _multi_project_on;
}
void osr_map::fixed_object_layer::reset()
{
	curr_time_ = ros::Time::now();
	float time_diff = (float) ((curr_time_ - prev_time_).toNSec() * (1e-6));
	prev_time_ = curr_time_;
}
void osr_map::fixed_object_layer::run()
{
	period_.reset();
	reset();
	ros::spinOnce();

	if(!is_in_tracks_ || !is_in_pose_){
		ROS_ERROR("object layer : NO TOPIC INPUT FOR TRACKS AND POSE.");
		period_.sleep();
		return;
	}
//	if(objects_.tracks.size() == 0)
//		return;

	tf::Quaternion q(odom_.pose.pose.orientation.x, odom_.pose.pose.orientation.y, odom_.pose.pose.orientation.z, odom_.pose.pose.orientation.w);
	tf::Matrix3x3 m(q);
	double roll, pitch, yaw;
	m.getRPY(roll, pitch, yaw);



	Eigen::Matrix4f l2g_pos, l2g_vel;
    l2g_pos << std::cos(yaw), -std::sin(yaw), 0, odom_.pose.pose.position.x,
			   std::sin(yaw),  std::cos(yaw), 0, odom_.pose.pose.position.y,
			   0, 0, 1, 0,
               0, 0, 0, 1;

    l2g_vel << std::cos(yaw), -std::sin(yaw), 0, odom_.twist.twist.linear.x,
			   std::sin(yaw),  std::cos(yaw), 0, odom_.twist.twist.linear.y,
			   0, 0, 1, 0,
			   0, 0, 0, 1;

	// map merge
	grid_map::GridMap layer;
	layer.setTimestamp(curr_time_.toNSec());
	layer.setGeometry(grid_map::Length(length_x_, length_y_), resolution_, grid_map::Position(odom_.pose.pose.position.x, odom_.pose.pose.position.y));
	layer.setFrameId(map_frame_id_);
	// multi-layered map
	// Update
	layer.add("object_update", robot_id_); // it should get range

	// Person
	layer.add("person_flow_x", 0);
	layer.add("person_flow_y", 0);
	layer.add("person_number", 0);
	layer.add("person_posture", 0);

	// Car
	layer.add("car_flow_x", 0);
	layer.add("car_flow_y", 0);
	layer.add("car_number", 0);


	// Make update layer // with four points in image
	//

	// Projection to Map Layer
	// Point to Pixel Projection (3X3 Conv w/ Elevation Mask later)
	for(std::vector<osr_msgs::Track>::iterator it = objects_.tracks.begin(); it != objects_.tracks.end(); it++)
	{
		// track : local to global position and velocity
		Eigen::Vector4f lpos(it->pose.position.x, it->pose.position.y, 0, 1);
	    Eigen::Vector4f lvel(it->twist.linear.x, it->twist.linear.y, 0, 1);
		Eigen::Vector4f gpos = l2g_pos * lpos;
		Eigen::Vector4f gvel = l2g_vel * lvel;
		grid_map::Position mpos(gpos(0), gpos(1));

		if(layer.isInside(mpos))
		{
			grid_map::Index index;
			layer.getIndex(mpos, index);
			// person
			if(static_cast<int>(it->type) == 1){
				if(layer.isValid(index, "person_number") && multi_project_on_){
					layer.at("person_posture", index) = static_cast<int>(it->posture);
					layer.at("person_flow_x", index) = (layer.at("person_flow_x", index)*layer.at("person_number", index) + gvel(0))/(layer.at("person_number", index) + 1);
					layer.at("person_flow_y", index) = (layer.at("person_flow_y", index)*layer.at("person_number", index) + gvel(1))/(layer.at("person_number", index) + 1);
					layer.at("person_number", index) += 1;
				}
				else{
					layer.at("person_posture", index) = static_cast<int>(it->posture);
					layer.at("person_number", index) = 1;
					layer.at("person_flow_x", index) = gvel(0);
					layer.at("person_flow_y", index) = gvel(1);
				}
			}
			// car
			else if(static_cast<int>(it->type) == 2){

				if(layer.isValid(index, "car_number") && multi_project_on_){
					layer.at("car_number", index) += 1;
					layer.at("car_flow_x", index) = (layer.at("car_flow_x", index)*layer.at("car_number", index) + gvel(0))/(layer.at("car_number", index) + 1);
					layer.at("car_flow_y", index) = (layer.at("car_flow_y", index)*layer.at("car_number", index) + gvel(1))/(layer.at("car_number", index) + 1);
				}
				else{
					layer.at("car_number", index) = 1;
					layer.at("car_flow_x", index) = gvel(0);
					layer.at("car_flow_y", index) = gvel(1);
				}
			}
			else{
				ROS_ERROR("object layer : NO DEFINED TRACK TYPE.");
			}
		}
	}


	// map to publish
	if(pub_layer_.getNumSubscribers() > 0)
	{
		grid_map_msgs::GridMap layer_msg;
		grid_map::GridMapRosConverter::toMessage(layer, layer_msg);
		pub_layer_.publish(layer_msg);
	}
	period_.sleep();

}
void osr_map::fixed_object_layer::subscribe_object_callback(const osr_msgs::Tracks::ConstPtr& _msg) {
	is_in_tracks_ = true;
	objects_ = *_msg;
}
void osr_map::fixed_object_layer::subscribe_pose_callback(const nav_msgs::Odometry::ConstPtr& _msg) {
	is_in_pose_ = true;
	odom_ = *_msg;
}
