/**
 * @file		parsers.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes functions to convert between layered map (UDP message) and grid map (map library).
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

#ifndef PARSER_H_
#define PARSER_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include <grid_map_core/GridMap.hpp>
#include "packets.h"

namespace osr_map{
/**
* @fn		convert_light_map_from_grid_map
* @brief	convert grid map to layered map message
* @param	_id					agent id
* @param	_agent_id			agent name
* @param	_seq				sequence number
* @param	_grid_map			grid_map to convert to message
* @param	_min_max_table		minimum and maximum thresholds for layer to compress data
* @param	_map 				layered map message
*/
void convert_light_map_from_grid_map(const int _id, const std::string& _agent_id, const int _seq, const grid_map::GridMap& _grid_map, std::map<std::string, std::pair<float, float>>& _min_max_table, osr_map::light_layered_map& _map){

	_map.agent_id = _agent_id;
	_map.id = static_cast<uint8_t>(_id);
	_map.seq = static_cast<uint64_t>(_seq);
    _map.timestamp = _grid_map.getTimestamp();
    _map.frame_id = _grid_map.getFrameId();
	_map.length_x = _grid_map.getLength().x();
	_map.length_y = _grid_map.getLength().y();
	_map.resolution = _grid_map.getResolution();
	_map.position_x = _grid_map.getPosition().x();
	_map.position_y = _grid_map.getPosition().y();


	int num_row = _grid_map.getSize().x();
	int num_col = _grid_map.getSize().y();
	std::vector<std::string> layers = _grid_map.getLayers();
	for(const auto& layer : layers){
		float min_val = _min_max_table[layer].first;
		float max_val = _min_max_table[layer].second;

		osr_map::light_map_layer osr_map_layer;
		osr_map_layer.name = layer;
		osr_map_layer.min_val = _min_max_table[layer].first;
		osr_map_layer.max_val = _min_max_table[layer].second;
		osr_map_layer.data.resize(num_row * num_col);
		Eigen::MatrixXf data = _grid_map.get(layer);

		if(osr_map_layer.min_val < osr_map_layer.max_val){
			for(int row = 0; row < num_row; ++row){
				for(int col = 0; col < num_col; ++col){
					if(std::isnan(data(row, col))){
						osr_map_layer.data.at(num_row * row + col) = 0;
					}
					else{
						osr_map_layer.data.at(num_row * row + col) = 1 + static_cast<uint8_t>(std::round((data(row, col) - osr_map_layer.min_val) / (osr_map_layer.max_val - osr_map_layer.min_val) * 254.0));
					}
				}
			}
		}
		else{
			for(int row = 0; row < num_row; ++row){
				for(int col = 0; col < num_col; ++col){
					if(std::isnan(data(row, col))){
						osr_map_layer.data.at(num_row * row + col) = 0;
					}
					else{
						osr_map_layer.data.at(num_row * row + col) = static_cast<uint8_t>(data(row, col) + 1);
					}
				}
			}
		}
		_map.map_layers.push_back(osr_map_layer);
	}
}

/**
* @fn		convert_light_map_to_grid_map
* @brief	convert layered map message to grid map
* @param	_map				layered map message
* @param 	_id					agent id
* @param	_agent_id			agent name
* @param	_seq				sequence number
* @param	_grid_map			grid_map to convert to message
*/
void convert_light_map_to_grid_map(osr_map::light_layered_map& _map, int& _id, std::string& _agent_id, int& _seq, grid_map::GridMap& _grid_map)
{
	_agent_id = _map.agent_id;
	_id = static_cast<int>(_map.id);
	_seq = static_cast<int>(_map.seq);
	_grid_map.setTimestamp(_map.timestamp);
	_grid_map.setFrameId(_map.frame_id);
	_grid_map.setGeometry(grid_map::Length(_map.length_x, _map.length_y), _map.resolution, grid_map::Position(_map.position_x, _map.position_y));

	int num_row = _grid_map.getSize().x();
	int num_col = _grid_map.getSize().y();
	for(std::vector<osr_map::light_map_layer>::iterator it = _map.map_layers.begin(); it != _map.map_layers.end(); ++it)
	{
		Eigen::MatrixXf map_layer(num_row, num_col);
		if(it->min_val < it->max_val){
			for(int idx = 0; idx < it->data.size(); idx++){
				int row = (idx / num_col);
				int col = (idx % num_col);
				if(it->data.at(idx) == 0){
					map_layer(row, col) = NAN;
				}
				else{
					map_layer(row, col) = it->min_val + static_cast<float>(it->data.at(idx) - 1) * (it->max_val - it->min_val) / 254.0;
				}
			}
		}
		else{
			for(int idx = 0; idx < it->data.size(); idx++){
				int row = (idx / num_col);
				int col = (idx % num_col);
				if(it->data.at(idx) == 0){
					map_layer(row, col) = NAN;
				}
				else{
					map_layer(row, col) = static_cast<float>(it->data.at(idx) - 1);
				}
			}
		}

		_grid_map.add(it->name, map_layer);
	}
}
}
#endif /* CONVERTER_H_ */
