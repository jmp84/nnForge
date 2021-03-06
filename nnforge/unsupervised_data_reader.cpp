/*
 *  Copyright 2011-2013 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "unsupervised_data_reader.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>

#include <vector>
#include <cmath>

namespace nnforge
{
	unsupervised_data_reader::unsupervised_data_reader()
	{
	}

	unsupervised_data_reader::~unsupervised_data_reader()
	{
	}

	size_t unsupervised_data_reader::get_input_neuron_elem_size() const
	{
		return neuron_data_type::get_input_size(get_input_type());
	}

	std::vector<feature_map_data_stat> unsupervised_data_reader::get_feature_map_input_data_stat_list()
	{
		std::vector<feature_map_data_stat> res;

		neuron_data_type::input_type type_code = get_input_type();

		if (type_code != neuron_data_type::type_float)
			throw neural_network_exception(((boost::format("Unable to stat data reader with input data type %1%") % type_code).str()));

		reset();

		unsigned int entry_count = get_entry_count();
		if (entry_count == 0)
			throw neural_network_exception("Unable to stat data reader with no entries");

		layer_configuration_specific input_configuration = get_input_configuration();
		res.resize(input_configuration.feature_map_count);

		for(std::vector<feature_map_data_stat>::iterator it = res.begin(); it != res.end(); ++it)
		{
			it->min = std::numeric_limits<float>::max();
			it->max = -std::numeric_limits<float>::max();
			it->average = 0.0F;
			it->std_dev = 0.0F;
		}

		std::vector<float> input_data(input_configuration.get_neuron_count());
		unsigned int neuron_count_per_feature_map = input_configuration.get_neuron_count_per_feature_map();

		while(read(&(*input_data.begin())))
		{
			std::vector<float>::const_iterator data_it = input_data.begin();
			for(std::vector<feature_map_data_stat>::iterator fm_it = res.begin(); fm_it != res.end(); ++fm_it)
			{
				float current_sum = 0.0F;
				for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
				{
					float val = *data_it;
					fm_it->min = std::min(fm_it->min, val);
					fm_it->max = std::max(fm_it->max, val);
					current_sum += val;
					++data_it;
				}
				fm_it->average += current_sum;
			}
		}

		float mult = 1.0F / ((float)entry_count * (float)neuron_count_per_feature_map);
		for(std::vector<feature_map_data_stat>::iterator it = res.begin(); it != res.end(); ++it)
			it->average *= mult;

		reset();

		while(read(&(*input_data.begin())))
		{
			std::vector<float>::const_iterator data_it = input_data.begin();
			for(std::vector<feature_map_data_stat>::iterator fm_it = res.begin(); fm_it != res.end(); ++fm_it)
			{
				float current_sum = 0.0F;
				float average = fm_it->average;
				for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
				{
					float val = *data_it;
					float diff = val - average;
					current_sum += diff * diff;
					++data_it;
				}
				fm_it->std_dev += current_sum;
			}
		}

		for(std::vector<feature_map_data_stat>::iterator it = res.begin(); it != res.end(); ++it)
			it->std_dev = sqrtf(it->std_dev * mult);

		return res;
	}

	void unsupervised_data_reader::next_epoch()
	{
		reset();
	}
}
