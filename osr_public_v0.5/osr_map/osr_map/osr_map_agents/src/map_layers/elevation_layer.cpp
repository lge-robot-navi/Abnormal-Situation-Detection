/**
 * @file		elevation_layer.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in elevation_layer class.
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


#include "osr_map_agents/map_layers/elevation_layer.h"


osr_map::elevation_layer::elevation_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id)
	: nh_(_nh), period_(_period), robot_id_(_robot_id), robot_frame_id_(_robot_frame_id), map_frame_id_(_map_frame_id)
{
	curr_time_ = prev_time_ = ros::Time::now();
	pointcloud_ptr_ = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	is_in_pointcloud_ = false;
	is_in_pose_ = false;
};
void osr_map::elevation_layer::set_comm(std::string _layer_topic, std::string _pose_topic, std::string _pointcloud_topic)
{
  	pub_layer_ = nh_.advertise<grid_map_msgs::GridMap>(_layer_topic, 1, true);
	sub_pose_ = nh_.subscribe(_pose_topic, 1, &osr_map::elevation_layer::subscribe_pose_callback, this);
	sub_pointcloud_ = nh_.subscribe(_pointcloud_topic, 1, &osr_map::elevation_layer::subscribe_pointcloud_callback, this);
}
void osr_map::elevation_layer::set_map_param(float _length_x, float _length_y, float _resolution)
{
	length_x_ = _length_x;
	length_y_ = _length_y;
	resolution_ = _resolution;
}
void osr_map::elevation_layer::set_range_param(float _min_range, float _max_range){
	min_range_ = _min_range;
	max_range_ = _max_range;
}
void osr_map::elevation_layer::set_z_param(float _offset_z, float _min_z, float _max_z, float _thickness)
{
	offset_z_ = _offset_z;
	min_z_ = _min_z;
	max_z_ = _max_z;
	thickness_ = _thickness;
}
void osr_map::elevation_layer::reset()
{
	curr_time_ = ros::Time::now();
	float time_diff = (float) ((curr_time_ - prev_time_).toNSec() * (1e-6));
	prev_time_ = curr_time_;
}
void osr_map::elevation_layer::run()
{
	period_.reset();
	reset();
	ros::spinOnce();

	if(!is_in_pointcloud_ || !is_in_pose_){
		ROS_ERROR("elevation layer : NO TOPIC INPUT FOR POINTCLOUD and POSE.");
		period_.sleep();
		return;
	}
	if(pointcloud_ptr_->size() == 0)
	{
		ROS_ERROR("elevation layer : NO INPUT POINTCLOUD.");
		period_.sleep();
		return;
	}
    pcl::PointCloud< pcl::PointXYZ >::Ptr temp_pointcloud_ptr = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	for(pcl::PointCloud<pcl::PointXYZ>::iterator it = pointcloud_ptr_->begin(); it != pointcloud_ptr_->end(); ++it)
	{
		float range = std::sqrt(it->x * it->x + it->y * it->y);
		if(range < min_range_ || range > max_range_){
			continue;
		}
		temp_pointcloud_ptr->push_back(*it);
	}
	// map merge
	grid_map::GridMap layer;
	layer.setTimestamp(curr_time_.toNSec());
	layer.setGeometry(grid_map::Length(length_x_, length_y_), resolution_, grid_map::Position(pose_.pose.pose.position.x, pose_.pose.pose.position.y));
	layer.setFrameId(map_frame_id_);
	layer.add("elevation");
	layer.add("elevation_update", 0);

	pcl::transformPointCloud(*(temp_pointcloud_ptr), *(pointcloud_ptr_), Eigen::Vector3d(pose_.pose.pose.position.x, pose_.pose.pose.position.y, pose_.pose.pose.position.z)
							, Eigen::Quaterniond(pose_.pose.pose.orientation.w, pose_.pose.pose.orientation.x, pose_.pose.pose.orientation.y, pose_.pose.pose.orientation.z));

    //ray-tracing?
	for(pcl::PointCloud<pcl::PointXYZ>::iterator it = pointcloud_ptr_->begin(); it != pointcloud_ptr_->end(); ++it)
	{
		grid_map::Position point(it->x, it->y);
		float z = it->z + offset_z_;
		z = std::max(z, min_z_);
		z = std::min(z, max_z_);
		if(layer.isInside(point)){
			grid_map::Index index;
			layer.getIndex(point, index);
			if(layer.isValid(index, "elevation")){
				layer.atPosition("elevation", point) = std::max(layer.atPosition("elevation", point), z);
			}
			else{
				layer.atPosition("elevation", point) = z;
				layer.atPosition("elevation_update", point) = robot_id_;
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
void osr_map::elevation_layer::subscribe_pose_callback(const nav_msgs::Odometry::ConstPtr& _msg){
	is_in_pose_ = true;
	pose_ = *_msg;
}
void osr_map::elevation_layer::subscribe_pointcloud_callback(const sensor_msgs::PointCloud2::ConstPtr& _msg)
{
	is_in_pointcloud_ = true;
	pcl::fromROSMsg(*_msg, *(pointcloud_ptr_));
}
