/**
 * @file		map_manager.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes map_manager class for integrating the local layered maps from multiple agents to build the global layered map.
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

#ifndef MAP_MANAGER_H_
#define MAP_MANAGER_H_

#include <vector>
#include <iostream>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <tinyxml.h>
#include <regex>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


namespace osr_map{
/**
 *  @class 			map_manager
 *  @brief 			This class integrate the local layered maps from multiple agents to build the global layered map
 *  @since			Mar 22, 2019
 */
class map_manager{
public:
	map_manager(std::string _map_frame_id, std::string _world_frame_id);
	~map_manager(){};

	/**
	* @fn		update_map_with_max
	* @brief	update each layer of global layered map with maximum value among global layered map and local layered map
	* @param	_maps 		local layered maps
	* @param	_extend		if local layered map has value outside of global layered map,
	* 						the global map is resized to include all of the local map.
	* @return	success or fail
	*/
	bool update_map_with_max(std::map<std::string, grid_map::GridMap> _maps, bool _extend);
	bool update_map_with_max(std::vector<grid_map::GridMap> _maps, bool _extend);


	/**
	* @fn		overwrite_map
	* @brief	overwrite multiple local layered maps on the global layered map.
	* @param	_maps 		local layered maps
	* @param	_extend		if local layered map has value outside of global layered map,
	* 						the global map is resized to include all of the local map.
	* @return 	success or fail
	*/
	bool overwrite_map(std::map<std::string, grid_map::GridMap> _maps, bool _extend);
	bool overwrite_map(std::vector<grid_map::GridMap> _maps, bool _extend);
	bool overwrite_map(grid_map::GridMap _map);


	/**
	* @fn		project_map
	* @brief	directly project local layered map onto the global layered map.
	* @param	_lmap 	local layered map
	* @return 	success or fail
	*/
    bool project_map(const grid_map::GridMap& _lmap);


	/**
	* @fn		reset_update_layers
	* @brief	reset update layers to keep up-to-date data.
	* @param	_update_layers 		the list of update layers
	*/
	void reset_update_layers(std::vector<std::string> _update_layers);


	/**
	* @fn		save_map_bag
	* @brief	save the current global layered map to ROS bag format
	* @param	_path 		the file name and path to save the map
	* @param	_map_name	the map topic name of bag
	* @return 	success or fail
	*/
	bool save_map_bag(std::string _path, std::string _map_name);

	/**
	* @fn		load_map_bag
	* @brief	load the global layered map from ROS bag format
	* @param	_path 		the file name and path to load the map
	* @param	_map_name	the map topic name of bag
	* @return 	success or fail
	*/
	bool load_map_bag(std::string _path, std::string _map_name);

	/**
	* @fn		save_map
	* @brief	save the current global layered map to XML map meta format and image.
	* @param	_path 		the file name and path to save the map meta and images
	* @param	_map_name	the map name
	* @return 	success or fail
	*/
	bool save_map(std::string _path, std::string _map_name);


	/**
	* @fn		load_map
	* @brief	load the current global layered map from XML map meta format and image.
	* @param	_path 		the file name and path to load the map meta and images
	* @param	_map_name	the map name
	* @param 	_layeres 	the list of layer names
	* @return 	success or fail
	*/
	bool load_map(std::string _path, std::string _map_name, std::vector<std::string> _layers);


	/**
	* @fn		initialize_map
	* @brief	generate new global layered map
	* @param	_pos 		the position of the global layered map
	* @param	_len		the size of the global layered map
	* @param 	_resol 		the resolution of the global layered map
	* @param 	_layers 	the list of layer names
	*/
	void initialize_map(grid_map::Position _pos, grid_map::Length _len, float _resol, std::vector<std::string> _layers);


	/**
	* @fn		get_map
	* @brief	get the current global layered map
	* @return	map_	the current global layered map
	*/
	grid_map::GridMap get_map();

	/**
	* @fn		get_map
	* @brief	get the current global layered map
	* @param	_map	the empty map to get the current global layered map
	*/
	void get_map(grid_map::GridMap& _map);

private:
	std::string map_frame_id_, world_frame_id_;
	grid_map::GridMap map_;
};

}
#endif /* MAP_MANAGER_H_ */
