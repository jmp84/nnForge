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

#include "data_writer.h"
#include "supervised_sparse_data_stream_schema.h"
#include "layer_configuration_specific.h"
#include "neuron_data_type.h"
#include "nn_types.h"

#include <vector>
#include <ostream>

namespace nnforge {

/**
 * Training data writer. Writes data in sparse format. The corresponding
 * reader is supervised_sparse_data_stream_reader. Very similar to
 * supervised_data_stream_writer.
 */
class supervised_sparse_data_stream_writer : public data_writer {
public:
  // The constructor modifies output_stream to throw exceptions in case of failure
  // The stream should be created with std::ios_base::binary flag
  /**
   * Constructor. Writes metadata.
   * @param output_stream The training data stream.
   * @param input_configuration The input config (feature maps, etc.).
   * @param output_configuration The output config.
   * @param type_code The neuron type. Should be float here.
   */
  supervised_sparse_data_stream_writer(
      nnforge_shared_ptr<std::ostream> output_stream,
      const layer_configuration_specific& input_configuration,
      const layer_configuration_specific& output_configuration,
      neuron_data_type::input_type type_code = neuron_data_type::type_unknown);

  virtual ~supervised_sparse_data_stream_writer();

  /**
   * Writes one training instance to the stream.
   * @param type_code The neuron type.
   * @param input_neurons The input neurons to be written to the stream.
   * @param output_neurons The output neurons to be written to the stream.
   */
  void write(
      neuron_data_type::input_type type_code,
      const void * input_neurons,
      const float * output_neurons);

  /**
   * Writes one training instance to the stream. The input neuron type is float.
   * @param input_neurons The input neurons to be written to the stream.
   * @param output_neurons The output neurons to be written to the stream.
   */
  void write(
      const float * input_neurons,
      const float * output_neurons);

  /**
   * Writes one training instance to the stream. The input neuron type is
   * unsigned char.
   * @param input_neurons The input neurons to be written to the stream.
   * @param output_neurons The output neurons to be written to the stream.
   */
  void write(
      const unsigned char * input_neurons,
      const float * output_neurons);

  /**
   * Writes one training instance in raw format. This is used to write
   * randomized data.
   * @param all_entry_data The training data point to be written to the stream.
   * @param data_length The training data point size. Should be the number of
   * bytes I think.
   */
  virtual void raw_write(
      const void * all_entry_data,
      size_t data_length);

private:
  /** the stream where to write training data */
  nnforge_shared_ptr<std::ostream> out_stream;
  /** the input neuron count (14 * 16000 in the nnjm paper) */
  unsigned int input_neuron_count;
  /** the input sparse neuron count (14 in the nnjm paper) */
  unsigned int input_sparse_neuron_count;
  /** the output neuron count (32000 in the nnjm paper) */
  unsigned int output_neuron_count;
  /** the output sparse neuron count (1 in the nnjm paper) */
  unsigned int output_sparse_neuron_count;

  /** records the position where the input neuron type needs to be written.
   * only useful for unknown input neuron type */
  std::ostream::pos_type type_code_pos;
  /** input neuron type */
  neuron_data_type::input_type type_code;
  /** size of input neuron type (4 for float for example) */
  size_t input_elem_size;

  /** records the position where the number of training data points needs
   * to be written. that number can only be known after reading all data */
  std::ostream::pos_type entry_count_pos;
  /** number of training data points */
  unsigned int entry_count;

private:
  supervised_sparse_data_stream_writer(
      const supervised_sparse_data_stream_writer&);
  supervised_sparse_data_stream_writer& operator =(
      const supervised_sparse_data_stream_writer&);
};

typedef nnforge_shared_ptr<supervised_sparse_data_stream_writer>
    supervised_sparse_data_stream_writer_smart_ptr;
}
