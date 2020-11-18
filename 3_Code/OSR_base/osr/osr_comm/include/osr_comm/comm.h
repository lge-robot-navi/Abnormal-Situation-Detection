/*
 * comm.h
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

#ifndef OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_COMM_H_
#define OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_COMM_H_


#include <list>
#include <thread>
// boost
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/serialization/vector.hpp>
#include "packets.h"
const int MAX_STORAGE_SIZE = 5;
const int MAX_BUFFER_SIZE = 60000;
using boost::asio::ip::udp;
using boost::asio::ip::address;




namespace osr{
unsigned int get_port(int _local_id, int _data_type, int _robot_id){
	return static_cast<unsigned int>(10000 + 1000 * _local_id + 100 * _data_type + _robot_id);
}
std::vector<unsigned int> get_ports(int _local_id, int _data_type, std::vector<int> _robot_ids)
{
	std::vector<unsigned int> ports;
	for(int p = 0; p < _robot_ids.size(); ++p){
		ports.push_back(get_port(_local_id, _data_type, _robot_ids[p]));
	}
	return ports;
}

/**
 *  @class 			udp_receiver
 *  @brief 			This class is for receiving synchronized UDP message
 *  @since			Mar 22, 2019
 */
//<template packet_class>
template <typename T>
class udp_receiver {
public:
	udp_receiver(boost::asio::io_service& _io_service, const unsigned int& _port, const int _storage_size = MAX_STORAGE_SIZE) :
		socket_(_io_service, udp::endpoint(udp::v4(), _port)), storage_size_(_storage_size), is_receiving_(false), is_taking_(false){};
	/**
	* @fn		receive_loop
	* @brief	periodically run receive function.
	*/
	void receive_loop(){
		while(1){
			float byte_transferred = receive();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	/**
	* @fn		receive
	* @brief	receive compressed layered_map
	* @return 	bytes_transferred	the size of transferred message
	*/
	size_t receive(){
		try {
			boost::asio::streambuf::mutable_buffers_type mutable_buffer = buffer_.prepare(MAX_BUFFER_SIZE);
			size_t bytes_transferred = socket_.receive(mutable_buffer);
			boost::system::error_code error;
			if (error && error != boost::asio::error::message_size)
				throw boost::system::system_error(error);
			buffer_.commit(bytes_transferred);
			std::istream is(&buffer_);
			boost::archive::binary_iarchive ia(is);
			T msg;
			ia >> msg;
			if(is_taking_){
				msg.clear();
				return 0;
			}
			is_receiving_ = true;
			if(messages_.size() > storage_size_){
				messages_.front().clear();
				messages_.pop_front();
			}
			messages_.push_back(msg);
			is_receiving_ = false;
			msg.clear();
			return bytes_transferred;
		} catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
	}
	/**
	* @fn		get_all_messages
	* @brief	get layered maps
	* @return 	success or fail
	*/

	bool get_all_messages(std::vector<T>& _msgs)
	{
		if(messages_.size() == 0) return false;
		is_taking_ = true;
		while(1){
			if(!is_receiving_){
				_msgs.resize(messages_.size());
				std::copy(messages_.begin(), messages_.end(), _msgs.begin());
//				for(std::list<T>::iterator it = messages_.begin(); it != messages_.end(); ++it){it->clear();}
				messages_.erase(messages_.begin(), messages_.end());
				is_taking_ = false;
				break;
			}
		}
		return true;
	}
	/**
	* @fn		get_new_message
	* @brief	get up-to-date layered maps
	* @return 	msg		layered map
	*/
	bool get_new_message(T& _msg)
	{
		if(messages_.size() == 0) return false;
		_msg = messages_.back();
		messages_.clear();
		return true;
	}
private:
	udp::socket socket_;
	std::list<T> messages_;
	boost::asio::streambuf buffer_;
	bool is_receiving_;
	bool is_taking_;
	int storage_size_;
};



/**
 *  @class 			udp_sender
 *  @brief 			This class is for sending synchronized UDP message
 *  @since			Mar 22, 2019
 */
template <typename T>
class udp_sender {
public:
	udp_sender(boost::asio::io_service& _io_service, const std::string& _ip_address, const unsigned int& _port) :
		socket_(_io_service), receiver_endpoint_(address::from_string(_ip_address), _port){
			socket_.open(udp::v4());
	};
	~udp_sender(){
		socket_.close();
	};
	/**
	* @fn		send
	* @brief	send layered maps
	* @return 	bytes_transferred	the size of transferred message
	*/
	size_t send(const T & _msg){

		try{
			boost::asio::streambuf buffer;
			std::ostream os(&buffer);
			boost::archive::binary_oarchive ar(os);
			ar << _msg;
			size_t bytes_transferred = socket_.send_to(buffer.data(), receiver_endpoint_);
			return bytes_transferred;
		}
		catch (std::exception& e){
			std::cerr << e.what() << std::endl;
		}
	}
private:
	udp::socket socket_;
	udp::endpoint receiver_endpoint_;
};
}



#endif /* OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_COMM_H_ */
