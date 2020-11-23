/**
 * @file		map_manager.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in map_manager class.
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



#include "osr_map_server/map_manager.h"


osr_map::map_manager::map_manager(std::string _map_frame_id, std::string _world_frame_id) :
	map_frame_id_(_map_frame_id), world_frame_id_(_world_frame_id)
{

}
bool osr_map::map_manager::update_map_with_max(std::map<std::string, grid_map::GridMap> _maps, bool _extend)
{
	for(std::map<std::string, grid_map::GridMap>::iterator it_map = _maps.begin(); it_map != _maps.end(); it_map++)
	{
		if (_extend) map_.extendToInclude(it_map->second);

		std::vector<std::string> dst_layers = map_.getLayers();
		std::vector<std::string> src_layers = it_map->second.getLayers();
		for (const auto& layer : src_layers) {
			if (std::find(dst_layers.begin(), dst_layers.end(), layer) == dst_layers.end()) {
				map_.add(layer);
			}
		}
		for (grid_map::GridMapIterator it_grid(it_map->second); !it_grid.isPastEnd(); ++it_grid) {
			grid_map::Position position;
			it_map->second.getPosition(*it_grid, position);
			grid_map::Index index;
			if (!map_.isInside(position)) continue;

			map_.getIndex(position, index);
			for (const auto& layer : src_layers) {
				if (!it_map->second.isValid(*it_grid)) continue;
				if(map_.isValid(index, layer)){ // if global map has value at the cell.
					map_.at(layer, index) = (abs(it_map->second.at(layer, *it_grid)) > abs(map_.at(layer, index))) ? it_map->second.at(layer, *it_grid) : map_.at(layer, index);
				}
				else{ // if not.
					map_.at(layer, index) = it_map->second.at(layer, *it_grid);
				}
			}
		}
	}
}
bool osr_map::map_manager::update_map_with_max(std::vector<grid_map::GridMap> _maps, bool _extend)
{
	for(std::vector<grid_map::GridMap>::iterator it_map = _maps.begin(); it_map != _maps.end(); it_map++)
	{
		if (_extend) map_.extendToInclude(*it_map);


		std::vector<std::string> dst_layers = map_.getLayers();
		std::vector<std::string> src_layers = it_map->getLayers();
		for (const auto& layer : src_layers) {
			if (std::find(dst_layers.begin(), dst_layers.end(), layer) == dst_layers.end()) {
				map_.add(layer);
			}
		}
		for (grid_map::GridMapIterator it_grid(*it_map); !it_grid.isPastEnd(); ++it_grid) {
			grid_map::Position position;
			it_map->getPosition(*it_grid, position);
			grid_map::Index index;
			if (!map_.isInside(position)) continue;

			map_.getIndex(position, index);
			for (const auto& layer : src_layers) {
				if (!it_map->isValid(*it_grid, layer)) continue;
				if(map_.isValid(index, layer)){ // if global map has value at the cell.
					map_.at(layer, index) = (abs(it_map->at(layer, *it_grid)) > abs(map_.at(layer, index))) ? it_map->at(layer, *it_grid) : map_.at(layer, index);
				}
				else{ // if not.
					map_.at(layer, index) = it_map->at(layer, *it_grid);
				}
			}
		}
	}
	return true;
}
void osr_map::map_manager::reset_update_layers(std::vector<std::string> _update_layers){
	for(const auto& layer : _update_layers){
		map_.clear(layer);
	}
}
bool osr_map::map_manager::project_map(const grid_map::GridMap& _lmap)
{
	std::vector<std::string> layers = map_.getLayers();
    std::vector<std::string> llayers = _lmap.getLayers();

    // data_layers
    for (grid_map::GridMapIterator it(_lmap); !it.isPastEnd(); ++it)
    {
        grid_map::Position pos;
        _lmap.getPosition(*it, pos);
        if(!map_.isInside(pos)){continue;} // fixed size of map
        grid_map::Index idx;
        map_.getIndex(pos, idx);
        for (const auto& layer : llayers){
            if(!_lmap.isValid(*it, layer)){continue;}
			std::size_t found = layer.find("update");
			if (found!=std::string::npos){ // found - update layer
				if (_lmap.at(layer, *it) == 0) {continue;}
			}
			else{ // not found - information layer

				map_.at(layer, idx) = _lmap.at(layer, *it);
			}
//            map_.at(layer, idx) = (abs(map_.at(layer, idx)) > abs(_lmap.at(layer, *it))) ? map_.at(layer, idx) : _lmap.at(layer, *it);
        }
    }
    return true;
}

bool osr_map::map_manager::project_map(const grid_map::GridMap& _lmap, const std::map<std::string, std::vector<std::string>>& _layers_map)
{
	std::vector<std::string> layers = map_.getLayers();
    std::vector<std::string> llayers = _lmap.getLayers();

    // data_layers
    for (grid_map::GridMapIterator it(_lmap); !it.isPastEnd(); ++it)
    {
        grid_map::Position pos;
        _lmap.getPosition(*it, pos);
        if(!map_.isInside(pos)){continue;} // fixed size of map
        grid_map::Index idx;
        map_.getIndex(pos, idx);

        for (auto uit = _layers_map.begin(); uit != _layers_map.end(); uit++){
            auto lit = std::find(llayers.begin(), llayers.end(), uit->first);
            if (lit == llayers.end()) {continue;}
            if(!_lmap.isValid(*it, uit->first)){continue;}
			if(_lmap.at(uit->first, *it) == 0) {continue;} // not update
			else{ // update
				map_.at(uit->first, idx) = _lmap.at(uit->first, *it); // update_layer update
				for(auto iit = uit->second.begin(); iit != uit->second.end(); iit++){ // information_layer update
					if(!_lmap.isValid(*it, *iit)){continue;}
					map_.at(*iit, idx) = _lmap.at(*iit, *it);
				}
			}
        }
    }
    return true;
}
bool osr_map::map_manager::overwrite_map(std::map<std::string, grid_map::GridMap> _maps, bool _extend)
{
	for(std::map<std::string, grid_map::GridMap>::iterator it = _maps.begin(); it != _maps.end(); it++)
	{
		if(!map_.addDataFrom(it->second, _extend, true, true)){
			ROS_ERROR("Failed to copy map" );
			return false;
		}
	}
	return true;
}
bool osr_map::map_manager::overwrite_map(std::vector<grid_map::GridMap> _maps, bool _extend)
{
	for(std::vector<grid_map::GridMap>::iterator it = _maps.begin(); it != _maps.end(); it++)
	{
		if(!project_map(*it)){
			ROS_ERROR("Failed to copy map" );
			return false;
		}
	}
	return true;
}
bool osr_map::map_manager::overwrite_map(grid_map::GridMap _map, std::map<std::string, std::vector<std::string>> _layers_map)
{

	if(!project_map(_map, _layers_map)){
		ROS_ERROR("Failed to copy map" );
		return false;
	}
	return true;
}
bool osr_map::map_manager::overwrite_map(grid_map::GridMap _map)
{

	if(!project_map(_map)){
		ROS_ERROR("Failed to copy map" );
		return false;
	}
	return true;
}

bool osr_map::map_manager::save_map_bag(std::string _path, std::string _map_name)
{
	std::string map_path = _path + "/" + _map_name + ".bag";
	if(grid_map::GridMapRosConverter::saveToBag(map_, map_path, _map_name))
	{
		ROS_INFO("map_manager : THE MAP IS SAVED.");
		return true;
	}
	else{
		return false;
	}
}

bool osr_map::map_manager::load_map_bag(std::string _path, std::string _map_name)
{
	std::string map_path = _path + "/" + _map_name + ".bag";
	std::ifstream map_file(map_path);
	if(map_file){
		if(grid_map::GridMapRosConverter::loadFromBag(map_path, _map_name, map_)) {
			ROS_INFO("map_manager: THE SAVED MAP IS LOADED.");
			return true;
		}
		else{
			ROS_ERROR("map_manager: FAILED TO LOAD MAP FROM BAG.");
			return false;
		}
	}
	else{
		ROS_ERROR("map_manager: FAILED TO LOAD MAP FROM BAG.");
		return false;
	}
}

bool osr_map::map_manager::load_map(std::string _path, std::string _map_name, std::vector<std::string> _layers)
{
	map_.clearAll();
	map_.setFrameId(map_frame_id_);

	std::string meta_path = _path + "/" + _map_name + "/" + "map_meta.xml";
	double pos_x, pos_y, length_x, length_y, resol;
	std::string layer_sentence;
	TiXmlDocument doc(meta_path);
	if(doc.LoadFile())
	{
		TiXmlHandle hDoc(&doc);
		TiXmlElement* pElem;
		TiXmlHandle hRoot(0);

		// block: name
		{
			pElem=hDoc.FirstChildElement().Element();
			hRoot=TiXmlHandle(pElem);
		}
		// block: position
		{
			pElem=hRoot.FirstChild("position").Element();
			if (pElem)
			{
				pElem->QueryDoubleAttribute("x", &pos_x);
				pElem->QueryDoubleAttribute("y", &pos_y);
			}
			else
			{
				ROS_ERROR("map_manager: FAILED TO LOAD MAP META.");
				return false;
			}
		}
		// block: length
		{
			pElem=hRoot.FirstChild("length").Element();
			if (pElem)
			{
				pElem->QueryDoubleAttribute("x", &length_x);
				pElem->QueryDoubleAttribute("y", &length_y);
			}
			else
			{
				ROS_ERROR("map_manager: FAILED TO LOAD MAP META.");
				return false;
			}
		}
		// block: resolution
		{
			pElem=hRoot.FirstChild("resolution").Element();
			if (pElem)
			{
				pElem->QueryDoubleAttribute("resolution", &resol);
			}
			else
			{
				ROS_ERROR("map_manager: FAILED TO LOAD MAP META.");
				return false;
			}
		}
		// block: layers
		std::vector<std::string> layers;
		{
			pElem=hRoot.FirstChild("layers").Element();
			if (pElem)
			{
				pElem->QueryStringAttribute("layers", &layer_sentence);
				std::regex reg(" ");
				std::sregex_token_iterator iter(layer_sentence.begin(), layer_sentence.end(), reg, -1);
				std::sregex_token_iterator end;
				layers = std::vector<std::string>(iter, end);
			}
			else
			{
				ROS_ERROR("map_manager: FAILED TO LOAD MAP META.");
				return false;
			}
		}
		ROS_INFO_STREAM("map_manager: THE MAP META IS LOADED. : (position_x, position_y, length_x, length_y) - (" << pos_x << ", "<< pos_y << ", " << length_x << ", " << length_y<<")");

		map_.setGeometry(grid_map::Length(length_x, length_y), resol, grid_map::Position(pos_x, pos_y));

		// image
		for(int idx = 0; idx < layers.size(); idx++)
		{
			std::string image_file_path = _path + "/" + _map_name + "/" + layers.at(idx) + ".png";
			cv::Mat image = cv::imread(image_file_path, CV_LOAD_IMAGE_UNCHANGED);
			if(image.data == NULL){
				ROS_ERROR_STREAM("FAILED TO LOAD " + layers.at(idx) + " IMAGE.");
			}
			grid_map::GridMapCvConverter::addLayerFromImage<float, 1>(image, layers.at(idx), map_, 0.0, 1.0);
		}
		ROS_INFO( "THE MAP META IS LOADED FROM IMAGES.");
		return true;
	}
	else
	{
		ROS_ERROR("map_manager: FAILED TO LOAD MAP META.");
		return false;
	}

}
bool osr_map::map_manager::save_map(std::string _path, std::string _map_name)
{
	// map meta
	std::string meta_path = _path + "/" + _map_name + "/" + "map_meta.xml";
	TiXmlDocument doc;
	TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );
	TiXmlElement * root = new TiXmlElement( "osr_map" );
	doc.LinkEndChild( root );
	TiXmlComment * comment = new TiXmlComment();
	comment->SetValue(" osr_map meta data " );
	root->LinkEndChild( comment );

	TiXmlElement * elem_position = new TiXmlElement( "position" );
	root->LinkEndChild( elem_position );
	elem_position->SetDoubleAttribute("x", map_.getPosition().x());
	elem_position->SetDoubleAttribute("y", map_.getPosition().y());

	TiXmlElement * elem_length = new TiXmlElement( "length" );
	root->LinkEndChild( elem_length );
	elem_length->SetDoubleAttribute("x", map_.getLength().x());
	elem_length->SetDoubleAttribute("y", map_.getLength().y());

	TiXmlElement * elem_resolution = new TiXmlElement( "resolution" );
	root->LinkEndChild( elem_resolution );
	elem_resolution->SetDoubleAttribute("resolution", map_.getResolution());


	TiXmlElement * elem_layers = new TiXmlElement( "layers" );
	root->LinkEndChild( elem_layers );
	std::string layer_sentence;
	std::vector<std::string> layers = map_.getLayers();
	// image
	for(int idx = 0; idx < layers.size(); idx++)
	{
		layer_sentence += layers.at(idx);
		if (idx < layers.size() - 1)
			layer_sentence += " ";
		std::string path = _path + "/" + _map_name + "/" + layers.at(idx) +".png";
		cv::Mat image = cv::Mat::zeros(map_.getSize()(0), map_.getSize()(1), CV_8UC1);
		if(grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map_, layers.at(idx), CV_8UC1, 0.0, 5.0, image))
		{
			ROS_INFO_STREAM( "save image " + layers.at(idx));
			cv::imwrite(path, image);
		}
		else{
			ROS_ERROR("Failed to convert images to save" );
			return false;
		}
	}
	elem_layers->SetAttribute("layers", layer_sentence);
	doc.SaveFile( meta_path );
	return true;
}

void osr_map::map_manager::initialize_map(grid_map::Position _pos, grid_map::Length _len, float _resol, std::vector<std::string> _layers)
{
	ROS_INFO("map_manager: NEW MAP SHOULD BE GENERATED.");
	map_.clearAll();
	map_.setFrameId(map_frame_id_);
	map_.setGeometry(_len, _resol, _pos);


	for (const auto& layer : _layers) {
		map_.add(layer);
	}
	ROS_INFO("map_manager: NEW MAP HAS BEEN GENERATED.");
}
grid_map::GridMap osr_map::map_manager::get_map()
{
	return map_;
}
void osr_map::map_manager::get_map(grid_map::GridMap& _map)
{
	_map = map_;
}
