/**
 * @file		fixed_elevation_layer.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in fixed_elevation_layer class.
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



#include "osr_map_agents/map_layers/fixed_elevation_layer.h"


osr_map::fixed_elevation_layer::fixed_elevation_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id)
	: nh_(_nh), period_(_period), robot_id_(_robot_id), robot_frame_id_(_robot_frame_id), map_frame_id_(_map_frame_id)
{
	curr_time_ = prev_time_ = ros::Time::now();
};
void osr_map::fixed_elevation_layer::set_comm(std::string _layer_topic, std::string _pose_topic, std::string _depth_image_topic)
{
  	pub_layer_ = nh_.advertise<grid_map_msgs::GridMap>(_layer_topic, 1, true);
	sub_pose_ = nh_.subscribe(_pose_topic, 1, &osr_map::fixed_elevation_layer::subscribe_pose_callback, this);
//	sub_pointcloud_ = nh_.subscribe(_pointcloud_topic, 1, &osr_map::fixed_elevation_layer::subscribe_pointcloud_callback, this);
	sub_depth_image_ = nh_.subscribe(_depth_image_topic, 1, &osr_map::fixed_elevation_layer::subscribe_depth_image_callback, this);
}
void osr_map::fixed_elevation_layer::set_map_param(float _length_x, float _length_y, float _resolution)
{
	length_x_ = _length_x;
	length_y_ = _length_y;
	resolution_ = _resolution;
}

void osr_map::fixed_elevation_layer::set_z_param(float _offset_z, float _min_z, float _max_z, float _thickness)
{
	offset_z_ = _offset_z;
	min_z_ = _min_z;
	max_z_ = _max_z;
	thickness_ = _thickness;
}
void osr_map::fixed_elevation_layer::set_distance_param(float _min_dist, float _max_dist)
{
	min_dist_ = _min_dist;
	max_dist_ = _max_dist;
}
void osr_map::fixed_elevation_layer::reset()
{
	curr_time_ = ros::Time::now();
	float time_diff = (float) ((curr_time_ - prev_time_).toNSec() * (1e-6));
	prev_time_ = curr_time_;
}
// read? set?
void osr_map::fixed_elevation_layer::set_camera_param(float _fx, float _fy, float _cx, float _cy, float _camera_x, float _camera_y, float _camera_z, float _rx, float _ry, float _rz)
{
	// Read File

    // Apply the parameters
    cv::Matx33d K(_fx, 0, _cx, 0, _fy, _cy, 0, 0, 1);
    K_inv_ = K.inv();
    cv::Matx33d Rc = cx::getRz(_rz) * cx::getRy(_ry) * cx::getRx(_rx);
    R_ = Rc.t();
    camera_x_ = _camera_x;
    camera_y_ = _camera_y;
    camera_z_ = _camera_z;
}
void osr_map::fixed_elevation_layer::run()
{
	period_.reset();
	reset();
	ros::spinOnce();     // depth_image to pointcloud

	int filter_size = 50;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_ptr;
	pointcloud_ptr = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < depth_image_.rows - filter_size; i = i + filter_size){
        for (int j = 0; j < depth_image_.cols - filter_size; j = j + filter_size){
        	int mi = i;
        	int mj = j;
        	float min_depth = depth_image_.at<float>(i, j);
        	for(int wi = 0; wi < filter_size; wi++){
        		for(int wj = 0; wj < filter_size; wj++){
        			if(min_depth > depth_image_.at<float>(wi, wj)){
        				depth_image_.at<float>(mi, mj) = 0;
        				min_depth = depth_image_.at<float>(wi, wj);
        			}
        			else{
        				depth_image_.at<float>(wi, wj) = 0;
        			}
        		}
        	}
        }
	}

	for (int y = 0; y < depth_image_.rows; y++)
     {
         float* depth_row = depth_image_.ptr<float>(y);
         for (int x = 0; x < depth_image_.cols; x++)
         {
             const float& d = depth_row[x];
             if (d > min_dist_ && d < max_dist_)
             {
                 cv::Point3d pc = d * K_inv_ * cv::Point3d(x, y, 1);
                 cv::Point3d pw = R_.t() * pc;
                 pointcloud_ptr->push_back(pcl::PointXYZ(pw.z + camera_y_, -pw.x -camera_x_, -pw.y + camera_z_));

             }
         }
     }
     pcl::VoxelGrid<pcl::PointXYZ> vxfilter;
     vxfilter.setInputCloud (pointcloud_ptr);
     vxfilter.setLeafSize (0.3f, 0.3f, 0.3f);
     vxfilter.filter (*pointcloud_ptr);

 	pcl::PointCloud<pcl::PointXYZ>::Ptr global_pointcloud_ptr;
 	global_pointcloud_ptr = pcl::PointCloud< pcl::PointXYZ >::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  	 pcl::transformPointCloud(*(pointcloud_ptr), *(global_pointcloud_ptr),
    			 Eigen::Vector3d(pose_.pose.pose.position.x, pose_.pose.pose.position.y, pose_.pose.pose.position.z),
    			 Eigen::Quaterniond(pose_.pose.pose.orientation.w, pose_.pose.pose.orientation.x, pose_.pose.pose.orientation.y, pose_.pose.pose.orientation.z));

 	// map merge
 	grid_map::GridMap layer;
 	layer.setTimestamp(curr_time_.toNSec());
 	layer.setGeometry(grid_map::Length(length_x_, length_y_), resolution_, grid_map::Position(pose_.pose.pose.position.x, pose_.pose.pose.position.y));
 	layer.setFrameId(map_frame_id_);
 	layer.add("elevation");




	for(pcl::PointCloud<pcl::PointXYZ>::iterator it = global_pointcloud_ptr->begin(); it != global_pointcloud_ptr->end(); ++it)
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
void osr_map::fixed_elevation_layer::subscribe_pose_callback(const nav_msgs::Odometry::ConstPtr& _msg) {

	pose_ = *_msg;
}
void osr_map::fixed_elevation_layer::subscribe_depth_image_callback(const sensor_msgs::Image::ConstPtr& _msg)
{
//	depth_image_ = *_msg;//pcl::fromROSMsg(*_msg, *(pointcloud_ptr_));
    cv_bridge::CvImagePtr depth_ptr;
    try
    {
        depth_ptr = cv_bridge::toCvCopy(_msg);
        depth_ptr->image.convertTo(depth_image_, CV_32F, 0.001);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception @ callbackDepth(): %s", e.what());
        return;
    }

}
