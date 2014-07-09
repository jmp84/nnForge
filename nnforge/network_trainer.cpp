/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "network_trainer.h"

#include <vector>

#include "neural_network_exception.h"

namespace nnforge
{
	network_trainer::network_trainer(network_schema_smart_ptr schema)
		: schema(schema)
		, epoch_count(50)
		, learning_rate_decay_tail_epoch_count(0)
		, learning_rate_decay_rate(0.5F)
		, learning_rate(0.02F)
	{
	}

	network_trainer::~network_trainer()
	{
	}

	void network_trainer::train(
		supervised_data_reader& reader,
		network_data_peeker& peeker,
		network_data_pusher& progress_pusher,
		network_data_pusher& pusher)
	{
		unsigned int reader_epoch_id = 0;
		current_error = previous_error = std::numeric_limits<float>::max();
		initialize_train(reader);
		unsigned int max_batch_size = get_max_batch_size();

		if (max_batch_size == 0)
			throw neural_network_exception("The trainer is unable to train even a single network");

		std::vector<training_task_state> task_list;

		while(true)
		{
			while (task_list.size() < max_batch_size)
			{
				network_data_peek_entry entry_peeked = peeker.peek(schema);
				if (entry_peeked.data == 0)
					break;

				training_task_state new_task;
				new_task.index_peeked = entry_peeked.index;
				new_task.data = entry_peeked.data;
				new_task.initial_epoch = entry_peeked.start_epoch;

				if (is_last_epoch(new_task))
				{
					std::cout << "Warning: Task is allocated which is already complete. Index " << new_task.index_peeked << ", Initial epoch " << new_task.initial_epoch << std::endl;
					continue;
				}

				task_list.push_back(new_task);

				if (new_task.initial_epoch > reader_epoch_id)
				{
					if (task_list.size() > 1)
						std::cout << "Warning: scrolling through reader requested (and done) while task_list is not empty. Index " << new_task.index_peeked << ", Initial epoch " << new_task.initial_epoch << std::endl;

					for(int i = reader_epoch_id; i < new_task.initial_epoch; ++i)
						reader.next_epoch();
					reader_epoch_id += (new_task.initial_epoch - reader_epoch_id);
				}
				else if (new_task.initial_epoch < reader_epoch_id)
					std::cout << "Warning: negative scrolling through reader requested. Index " << new_task.index_peeked << ", Initial epoch " << new_task.initial_epoch << std::endl;
			}

			if (task_list.size() == 0)
				break; // Nothing is left to be trained

			train_step(
				reader,
				task_list);

			for(int i = 0; i < task_list.size(); ++i)
				progress_pusher.push(task_list[i]);

			previous_error = current_error;
			current_error = task_list.back().history.back()->get_error();
			// The last task corresponds to testing.
			// It needs to be deleted before continuing
			task_list.back().history.pop_back(); 

			for(int i = static_cast<int>(task_list.size()) - 1; i >= 0; --i)
			{
				if (is_broken(task_list[i]))
				{
					std::cout << "# " << task_list[i].index_peeked << " - broken weights while training, discarding it." << std::endl;
					task_list.erase(task_list.begin() + i);
					continue;
				}

				if (is_last_epoch(task_list[i]))
				{
					pusher.push(task_list[i]);
					task_list.erase(task_list.begin() + i);
				}
			}

			reader.next_epoch();
			++reader_epoch_id;
		}
	}

	bool network_trainer::is_last_epoch(const training_task_state& state) const
	{
		return (state.get_current_epoch() >= epoch_count);
	}

	bool network_trainer::is_broken(const training_task_state& state) const
	{
		float error = state.history.back()->get_error();
		bool sanity_check = (error < 1.0e+10F) && (-error > -1.0E+10F) && !(-error < -1.0E+10F);
		return !sanity_check;
	}

	float network_trainer::get_global_learning_rate(unsigned int epoch) const
	{
		float tail_degradation_factor = 1.0F;
		{
			int first_iteration_with_decay = std::max(static_cast<int>(epoch_count) - static_cast<int>(learning_rate_decay_tail_epoch_count), 1);
			int tail_degradation_epoch = static_cast<int>(epoch) - first_iteration_with_decay + 1;
			if (tail_degradation_epoch > 0)
				tail_degradation_factor = powf(learning_rate_decay_rate, static_cast<float>(tail_degradation_epoch));
		}

		float head_degradation_factor = 1.0F;
		{
			int head_rise_epoch = static_cast<int>(learning_rate_rise_head_epoch_count) - static_cast<int>(epoch);
			if (head_rise_epoch > 0)
				head_degradation_factor = powf(learning_rate_rise_rate, static_cast<float>(head_rise_epoch));
		}

		return tail_degradation_factor * head_degradation_factor * learning_rate;
	}
}
