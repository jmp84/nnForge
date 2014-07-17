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

#pragma once

#include "factory_generator.h"
#include "config_options.h"
#include "network_data_pusher.h"
#include "testing_complete_result_set_visualizer.h"
#include "network_output_type.h"
#include "supervised_data_reader.h"
#include "data_writer.h"
#include "unsupervised_data_reader.h"
#include "output_neuron_value_set.h"
#include "output_neuron_class_set.h"
#include "layer_data_configuration.h"
#include "data_transformer.h"
#include "data_transformer_util.h"
#include "normalize_data_transformer.h"
#include "error_function.h"
#include "network_trainer.h"
#include "stream_duplicator.h"

#include <boost/filesystem.hpp>

namespace nnforge
{
	class neural_network_toolset
	{
	public:
		neural_network_toolset(factory_generator_smart_ptr factory);

		virtual ~neural_network_toolset();

		// Returns true if action is specified
		bool parse(int argc, char* argv[]);

		std::string get_action() const;

		void do_action();

	protected:
		virtual std::vector<string_option> get_string_options();

		virtual std::vector<bool_option> get_bool_options();

		virtual std::vector<float_option> get_float_options();

		virtual std::vector<int_option> get_int_options();

		virtual void prepare_training_data() = 0;

		virtual void prepare_testing_data();

		virtual void prepare_validating_data();

		virtual network_schema_smart_ptr get_schema() const = 0;

		virtual std::map<unsigned int, float> get_dropout_rate_map() const;

		virtual boost::filesystem::path get_input_data_folder() const;

		virtual boost::filesystem::path get_working_data_folder() const;

		virtual std::string get_class_name_by_id(unsigned int class_id) const;

		virtual network_tester_smart_ptr get_tester();

		virtual network_analyzer_smart_ptr get_analyzer();

		virtual std::vector<network_data_pusher_smart_ptr> get_validators_for_training(network_schema_smart_ptr schema);

		virtual network_output_type::output_type get_network_output_type() const;

		virtual bool is_training_with_validation() const;

		virtual testing_complete_result_set_visualizer_smart_ptr get_validating_visualizer() const;

		virtual void run_test_with_unsupervised_data(std::vector<output_neuron_value_set_smart_ptr>& predicted_neuron_value_set_list);

		virtual unsigned int get_testing_sample_count() const;

		virtual unsigned int get_validating_sample_count() const;

		virtual std::vector<data_transformer_smart_ptr> get_input_data_transformer_list_for_training() const;

		virtual std::vector<data_transformer_smart_ptr> get_output_data_transformer_list_for_training() const;

		virtual std::vector<data_transformer_smart_ptr> get_input_data_transformer_list_for_validating() const;

		virtual std::vector<data_transformer_smart_ptr> get_output_data_transformer_list_for_validating() const;

		virtual std::vector<data_transformer_smart_ptr> get_input_data_transformer_list_for_testing() const;

		virtual std::vector<data_transformer_smart_ptr> get_output_data_transformer_list_for_testing() const;

		virtual boost::filesystem::path get_ann_subfolder_name() const;

		virtual const_error_function_smart_ptr get_error_function() const;

		virtual std::pair<unsupervised_data_reader_smart_ptr, unsigned int> get_data_reader_and_sample_count_for_snapshots() const;

		virtual std::vector<std::vector<std::pair<unsigned int, unsigned int> > > get_samples_for_snapshot(
			network_data_smart_ptr data,
			unsupervised_data_reader_smart_ptr reader,
			unsigned int sample_count);

		virtual bool is_rgb_input() const;

		virtual supervised_data_reader_smart_ptr get_original_training_data_reader(const boost::filesystem::path& path) const;

		virtual data_writer_smart_ptr get_randomized_training_data_writer(
			supervised_data_reader& reader,
			const boost::filesystem::path& path) const;

		virtual supervised_data_reader_smart_ptr get_initial_data_reader_for_training() const;

		virtual supervised_data_reader_smart_ptr get_initial_data_reader_for_validating() const;

		virtual supervised_data_reader_smart_ptr get_initial_data_reader_for_testing_supervised() const;

		virtual unsupervised_data_reader_smart_ptr get_initial_data_reader_for_testing_unsupervised() const;

		virtual unsigned int get_classifier_visualizer_top_n() const;

		virtual std::vector<unsigned int> get_snapshot_data_dimension_list(unsigned int original_dimension_count) const;

	protected:
		static const char * training_data_filename;
		static const char * training_randomized_data_filename;
		static const char * validating_data_filename;
		static const char * testing_data_filename;
		static const char * testing_unsupervised_data_filename;
		static const char * schema_filename;
		static const char * normalizer_input_filename;
		static const char * normalizer_output_filename;
		static const char * snapshot_subfolder_name;
		static const char * snapshot_data_subfolder_name;
		static const char * ann_snapshot_subfolder_name;
		static const char * snapshot_invalid_subfolder_name;
		static const char * ann_subfolder_name;
		static const char * ann_resume_subfolder_name;
		static const char * trained_ann_index_extractor_pattern;
		static const char * logfile_name;

		network_tester_factory_smart_ptr tester_factory;
		network_updater_factory_smart_ptr updater_factory;
		hessian_calculator_factory_smart_ptr hessian_factory;
		network_analyzer_factory_smart_ptr analyzer_factory;

		std::string action;
		std::string snapshot_extension;
		std::string snapshot_extension_video;
		unsigned int ann_count;
		unsigned int training_epoch_count;
		unsigned int snapshot_count;
		float learning_rate;
		unsigned int learning_rate_decay_tail_epoch_count;
		float learning_rate_decay_rate;
		unsigned int learning_rate_rise_head_epoch_count;
		float learning_rate_rise_rate;
		float max_mu;
		bool per_layer_mu;
		float mu_increase_factor;
		unsigned int batch_offset;
		unsigned int snapshot_video_fps;
		int test_validate_ann_index;
		unsigned int snapshot_ann_index;
		std::string snapshot_data_set;
		unsigned int profile_updater_entry_count;
		unsigned int profile_hessian_entry_count;
		std::string training_algo;
		bool dump_resume;
		bool load_resume;
		unsigned int epoch_count_in_training_set;
		float weight_decay;
		// Contains the boundary for the uniform weights 
		// i.e. generate weights in (-initialize_uniform_weights,initialize_uniform_weights)
		std::string initialize_weights;
		float initial_weights_value;
		// user-defined initial seed for the random generator.
		// Experiments should be replicable if the same seed is used.
		unsigned long initial_weights_seed;
		// Apply decay to learning rate if validating NLL
		// for current epoch is worse than previous epoch
		bool learning_rate_decay_sgd_nll;
		unsigned int snapshot_scale;
		unsigned int batch_size;
		float momentum;

	protected:
		std::vector<output_neuron_value_set_smart_ptr> run_batch(
			supervised_data_reader& reader,
			output_neuron_value_set_smart_ptr actual_neuron_value_set);

		std::vector<output_neuron_value_set_smart_ptr> run_batch(unsupervised_data_reader& reader, unsigned int sample_count);

		void randomize_data();

		void create();

		void generate_input_normalizer();

		void generate_output_normalizer();

		unsigned int get_starting_index_for_batch_training();

		void validate(bool is_validate);

		void snapshot();

		void snapshot_data();

		void snapshot_invalid();

		void ann_snapshot();

		void save_ann_snapshot(
			const std::string& name,
			const network_data& data,
			const std::vector<layer_data_configuration_list>& layer_data_configuration_list_list);

		network_data_smart_ptr load_ann_data(unsigned int ann_id);

		void train();

		void profile_updater();

		void profile_hessian();

		normalize_data_transformer_smart_ptr get_input_data_normalize_transformer() const;

		normalize_data_transformer_smart_ptr get_output_data_normalize_transformer() const;

		normalize_data_transformer_smart_ptr get_reverse_output_data_normalize_transformer() const;

		supervised_data_reader_smart_ptr get_data_reader_for_training() const;

		std::pair<supervised_data_reader_smart_ptr, unsigned int> get_data_reader_for_validating_and_sample_count() const;

		std::pair<supervised_data_reader_smart_ptr, unsigned int> get_data_reader_for_testing_supervised_and_sample_count() const;

		std::pair<unsupervised_data_reader_smart_ptr, unsigned int> get_data_reader_for_testing_unsupervised_and_sample_count() const;

		std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> run_analyzer_for_single_neuron(
			network_analyzer& analyzer,
			unsigned int layer_id,
			unsigned int feature_map_id,
			const std::vector<unsigned int>& location_list,
			unsigned int feature_map_count) const;

		network_trainer_smart_ptr get_network_trainer(network_schema_smart_ptr schema) const;

		void dump_settings();

	private:
		factory_generator_smart_ptr factory;

		boost::filesystem::path input_data_folder;
		boost::filesystem::path working_data_folder;

		nnforge_shared_ptr<stream_duplicator> out_to_log_duplicator_smart_ptr;

	private:
		neural_network_toolset();
		neural_network_toolset(const neural_network_toolset&);
	};

	// Helper function that returns whether learning_rate 
	// must be updated if NLL gets worse (true) or rely on
	// default learning rate (false). 
	bool use_learning_rate_decay_sgd_nll(enum FuzzyBool b=UNSET);
}
