/*
 * supervised_sparse_data_stream_reader.h
 *
 *  Created on: 16 Jun 2014
 *      Author: jmp84
 */

#pragma once

#include "supervised_data_reader.h"

namespace nnforge {

/**
 * Training data reader. Training data is in sparse format. When reading,
 * dense format is restored to fit with the rest of the library. Very similar
 * to supervised_data_stream_reader
 */
class supervised_sparse_data_stream_reader : public supervised_data_reader {

public:
  supervised_sparse_data_stream_reader(
      nnforge_shared_ptr<std::istream> input_stream);

  virtual ~supervised_sparse_data_stream_reader();

  /**
   * Goes to the position of the first training data point in the training data
   * stream.
   */
  virtual void reset();

  /**
   * Reads one training data point in the training data stream. Converts to
   * dense format.
   * @param input_neurons Input neurons read from the training data stream.
   * Dense format.
   * @param output_neurons Output neurons read from the training data stream.
   * Dense format.
   * @return True if one training data point was read, false if no more training
   * data point.
   */
  virtual bool read(
        void * input_neurons,
        float * output_neurons);

  /**
   * Reads one training data point in raw format. Used to write randomized
   * training data.
   * @param all_elems Resulting data point
   * @return True if one training data point was read, false if no more training
   * data point.
   */
  virtual bool raw_read(std::vector<unsigned char>& all_elems);

  virtual layer_configuration_specific get_input_configuration() const {
    return input_configuration;
  }

  virtual layer_configuration_specific get_output_configuration() const {
    return output_configuration;
  }

  virtual neuron_data_type::input_type get_input_type() const {
    return type_code;
  }

  virtual unsigned int get_entry_count() const {
    return entry_count;
  }

  /**
   * Rewinds to the position of a specific training instance. Used to write
   * randomized data.
   * @param entry_id The training instance id.
   */
  virtual void rewind(unsigned int entry_id);

private:
  /**
   * Whether more training instances can be read.
   * @return Whether more training instances can be read.
   */
  bool entry_available();

  /**
   * Converts input neurons in sparse format into input neurons in dense format.
   * The float type for input neurons in sparse format is currently hard coded.
   * @param input_sparse_neurons
   * @param input_neurons
   */
  void sparse_input_to_nonsparse_input(float* input_sparse_neurons,
                                       void* input_neurons);

  /**
   * Converts output neurons in sparse format into output neurons in dense
   * format.
   * @param output_sparse_neurons
   * @param output_neurons
   */
  void sparse_output_to_nonsparse_output(float* output_sparse_neurons,
                                         float* output_neurons);

  /**
   * Rounding method. Very simple method because the input represents a word
   * id so it's always very close to an integer.
   * @param wordIdFloat The word id.
   * @return The word id converted to int.
   */
  unsigned int round_float_to_unsigned_int(const float wordIdFloat);

  /** the training data stream */
  nnforge_shared_ptr<std::istream> in_stream;
  /** number of input neurons (14 * 16000 in the nnjm paper) */
  unsigned int input_neuron_count;
  /** number of input neurons in sparse format (14 in the nnjm paper) */
  unsigned int input_sparse_neuron_count;
  /** number of output neurons (1 * 32000 in the nnjm paper) */
  unsigned int output_neuron_count;
  /** number of input neurons in sparse format (1 in the nnjm paper) */
  unsigned int output_sparse_neuron_count;
  /** input configuration: feature maps, etc. */
  layer_configuration_specific input_configuration;
  layer_configuration_specific output_configuration;
  /** should be float */
  neuron_data_type::input_type type_code;
  /** number of training instances */
  unsigned int entry_count;
  /** number of training instances read so far */
  unsigned int entry_read_count;
  /** position of the first training instance in the stream */
  std::istream::pos_type reset_pos;

  typedef nnforge_shared_ptr<supervised_sparse_data_stream_reader>
      supervised_sparse_data_stream_reader_smart_ptr;
};

} // namespace nnforge
