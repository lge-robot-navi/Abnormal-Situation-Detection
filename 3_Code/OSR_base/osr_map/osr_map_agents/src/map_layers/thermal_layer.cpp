/**
 * @file		thermal_layer.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in thermal_layer class.
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


#include "osr_map_agents/map_layers/thermal_layer.h"


osr_map::thermal_layer::thermal_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id)
	: nh_(_nh), period_(_period), robot_id_(_robot_id), robot_frame_id_(_robot_frame_id), map_frame_id_(_map_frame_id)
{
	curr_time_ = prev_time_ = ros::Time::now();
	pointcloud_ptr_ = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	is_in_thermal_to_lidar_info_ = false;
	is_in_pointcloud_ = false;
	is_in_thermal_ = false;
	is_in_pose_ = false;
};
void osr_map::thermal_layer::set_comm(std::string _layer_topic, std::string _pose_topic, std::string _thermal_image_topic, std::string _pointcloud_topic, std::string _thermal_to_lidar_info_topic)
{
  	pub_layer_ = nh_.advertise<grid_map_msgs::GridMap>(_layer_topic, 1, true);
	sub_pose_ = nh_.subscribe(_pose_topic, 1, &osr_map::thermal_layer::subscribe_pose_callback, this);
	sub_thermal_ = nh_.subscribe(_thermal_image_topic, 1, &osr_map::thermal_layer::subscribe_thermal_callback, this);
	sub_pointcloud_ = nh_.subscribe(_pointcloud_topic, 1, &osr_map::thermal_layer::subscribe_pointcloud_callback, this);
	sub_thermal_to_lidar_info_ = nh_.subscribe(_thermal_to_lidar_info_topic, 1, &osr_map::thermal_layer::subscribe_thermal_to_lidar_info_callback, this);
}
void osr_map::thermal_layer::set_map_param(float _length_x, float _length_y, float _resolution)
{
	length_x_ = _length_x;
	length_y_ = _length_y;
	resolution_ = _resolution;
}

void osr_map::thermal_layer::set_thermal_param(float _min_thermal, float _max_thermal)
{
	min_thermal_ = _min_thermal;
	max_thermal_ = _max_thermal;
}
void osr_map::thermal_layer::set_range_param(float _min_range, float _max_range)
{
	min_range_ = _min_range;
	max_range_ = _max_range;
}
void osr_map::thermal_layer::reset()
{
	curr_time_ = ros::Time::now();
	float time_diff = (float) ((curr_time_ - prev_time_).toNSec() * (1e-6));
	prev_time_ = curr_time_;
}
void osr_map::thermal_layer::run()
{
	period_.reset();
	reset();
	ros::spinOnce();

	if(!is_in_thermal_to_lidar_info_ || !is_in_pointcloud_ || !is_in_thermal_ || !is_in_pose_){
		ROS_ERROR("thermal layer : NO TOPIC INPUT FOR CAMERA INFO, POINTCLOUD, THERMAL, and POSE.");
		period_.sleep();
		return;
	}
	if(pointcloud_ptr_->size() == 0)
	{
		ROS_ERROR("thermal layer : NO INPUT POINTCLOUD.");
		period_.sleep();
		return;
	}


	pcl::PointCloud<pcl::PointXYZ>::Ptr thermal_points_ptr = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_points_ptr = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr indices_ptr = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

	tf::Matrix3x3 rot;
	rot.setRotation(tf::Quaternion(0.514381255168, -0.51859133471, 0.483379901823, -0.482513028227));
	tf::Vector3 trans(0.00858674282799, 0.103442391639, 0.113050548439);

	Eigen::Matrix4f transform;
	transform << rot[0][0], rot[0][1], rot[0][2], trans[0],
				 rot[1][0], rot[1][1], rot[1][2], trans[1],
				 rot[2][0], rot[2][1], rot[2][2], trans[2],
				 0, 0, 0, 1;

	pcl::transformPointCloud(*(pointcloud_ptr_), *(thermal_points_ptr), transform.inverse());
	int wsize = 5;
	int half_wsize = 2;
	int count = 0;
	Eigen::MatrixXi ids_mtx = Eigen::MatrixXi::Constant(thermal_image_.rows, thermal_image_.cols, -1);
	for(int idx = 0; idx < thermal_points_ptr->size(); ++idx)
	{
		float range = std::sqrt(pointcloud_ptr_->at(idx).x * pointcloud_ptr_->at(idx).x + pointcloud_ptr_->at(idx).y * pointcloud_ptr_->at(idx).y);
		if(range < min_range_ || range > max_range_){
			continue;
		}
		pcl::PointXYZ index;
		Eigen::Vector4f xyz1(thermal_points_ptr->at(idx).x, thermal_points_ptr->at(idx).y, thermal_points_ptr->at(idx).z, 1);
		Eigen::VectorXf uvw = thermal_to_lidar_ * xyz1;
		index.x = uvw(0)/uvw(2);
		index.y = uvw(1)/uvw(2);
		if(half_wsize > index.x || index.x >= thermal_image_.cols - half_wsize || half_wsize > index.y || index.y >= thermal_image_.rows - half_wsize){
			continue;
		}
		lidar_points_ptr->push_back(pointcloud_ptr_->at(idx));
		indices_ptr->push_back(index);
		ids_mtx(index.y, index.x) = count++;

	}
	// map merge
	grid_map::GridMap layer;
	layer.setTimestamp(curr_time_.toNSec());
	layer.setGeometry(grid_map::Length(length_x_, length_y_), resolution_, grid_map::Position(pose_.pose.pose.position.x, pose_.pose.pose.position.y));
	layer.setFrameId(map_frame_id_);
	layer.add("thermal");
	layer.add("thermal_update", 0);

	pcl::transformPointCloud(*(lidar_points_ptr), *(lidar_points_ptr), Eigen::Vector3d(pose_.pose.pose.position.x, pose_.pose.pose.position.y, pose_.pose.pose.position.z)
							, Eigen::Quaterniond(pose_.pose.pose.orientation.w, pose_.pose.pose.orientation.x, pose_.pose.pose.orientation.y, pose_.pose.pose.orientation.z));

	for(int idx = 0; idx < lidar_points_ptr->size(); ++idx)
	{
		grid_map::Position point(lidar_points_ptr->at(idx).x, lidar_points_ptr->at(idx).y);
		int row = indices_ptr->at(idx).y;
		int col = indices_ptr->at(idx).x;
		float max_temp = static_cast<float>(thermal_image_.at<uint16_t>(indices_ptr->at(idx).y, indices_ptr->at(idx).x));
		float x = lidar_points_ptr->at(idx).x;
		float y = lidar_points_ptr->at(idx).y;
		for(int i = row - half_wsize; i <= row + half_wsize; ++i){
			for(int j = col - half_wsize; j <= row + half_wsize; ++j){
				if(static_cast<float>(thermal_image_.at<uint16_t>(i, j)) > max_temp){
					if(ids_mtx(i, j) < 0){continue;}
					float dist = std::sqrt(std::pow(lidar_points_ptr->at(ids_mtx(i, j)).x - lidar_points_ptr->at(idx).x, 2.0) + std::pow(lidar_points_ptr->at(ids_mtx(i, j)).y - lidar_points_ptr->at(idx).y, 2.0));
					if(dist > 1.0){continue;}
					max_temp = static_cast<float>(thermal_image_.at<uint16_t>(i, j));
				}
			}
		}
		// denormalization
		float denorm_temp = min_thermal_ + max_temp * (max_thermal_ - min_thermal_) / 16383.0;
		denorm_temp = std::max(denorm_temp, min_thermal_);
		denorm_temp = std::min(denorm_temp, max_thermal_);
		if(layer.isInside(point)){
			grid_map::Index index;
			layer.getIndex(point, index);

			if(layer.isValid(index, "thermal")){
				layer.atPosition("thermal", point) = std::max(layer.atPosition("thermal", point), denorm_temp);
			}
			else{
				layer.atPosition("thermal", point) = denorm_temp;
				layer.atPosition("thermal_update", point) = robot_id_;
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
void osr_map::thermal_layer::subscribe_pose_callback(const nav_msgs::Odometry::ConstPtr& _msg)
{
	is_in_pose_ = true;
	pose_ = *_msg;
}
void osr_map::thermal_layer::subscribe_pointcloud_callback(const sensor_msgs::PointCloud2::ConstPtr& _msg)
{
	is_in_pointcloud_ = true;
	pcl::fromROSMsg(*_msg, *(pointcloud_ptr_));
}
void osr_map::thermal_layer::subscribe_thermal_callback(const sensor_msgs::Image::ConstPtr& _msg)
{
	is_in_thermal_ = true;
    cv_bridge::CvImagePtr thermal_ptr;
    try
    {
    	thermal_ptr = cv_bridge::toCvCopy(_msg);
    	thermal_ptr->image.convertTo(thermal_image_, CV_16UC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception @ callbackDepth(): %s", e.what());
        return;
    }
}
void osr_map::thermal_layer::subscribe_thermal_to_lidar_info_callback(const sensor_msgs::CameraInfo::ConstPtr& _msg)
{
	if(is_in_thermal_to_lidar_info_)
		return;

	is_in_thermal_to_lidar_info_ = true;
	thermal_to_lidar_ = Eigen::MatrixXf::Zero(3, 4);
//	thermal_to_lidar_(0, 0) = _msg->P[0];
//	thermal_to_lidar_(0, 1) = _msg->P[1];
//	thermal_to_lidar_(0, 2) = _msg->P[2];
//	thermal_to_lidar_(0, 3) = _msg->P[3];
//	thermal_to_lidar_(1, 0) = _msg->P[4];
//	thermal_to_lidar_(1, 1) = _msg->P[5];
//	thermal_to_lidar_(1, 2) = _msg->P[6];
//	thermal_to_lidar_(1, 3) = _msg->P[7];
//	thermal_to_lidar_(2, 2) = 1.0;
	thermal_to_lidar_ << _msg->P[0], _msg->P[1], _msg->P[2], _msg->P[3],
			_msg->P[4], _msg->P[5], _msg->P[6], _msg->P[7],
			_msg->P[8], _msg->P[9], _msg->P[10], _msg->P[11];


}

