/*
 * supervised_sparse_data_stream_reader.cpp
 *
 *  Created on: 16 Jun 2014
 *      Author: jmp84
 */

#include "neural_network_exception.h"
#include "supervised_sparse_data_stream_reader.h"
#include "supervised_sparse_data_stream_schema.h"

#include <boost/uuid/uuid_io.hpp>

namespace nnforge {

supervised_sparse_data_stream_reader::supervised_sparse_data_stream_reader(
    nnforge_shared_ptr<std::istream> input_stream) :
        in_stream(input_stream),
        entry_read_count(0) {
  in_stream->exceptions(std::ostream::eofbit |
                        std::ostream::failbit |
                        std::ostream::badbit);

  boost::uuids::uuid guid_read;
  in_stream->read(reinterpret_cast<char*>(guid_read.data),
                  sizeof(guid_read.data));
  if (guid_read !=
      supervised_sparse_data_stream_schema::supervised_sparse_data_stream_guid)
    throw neural_network_exception(
        (boost::format(
            "Unknown supervised data GUID encountered in "
            "input stream: %1%") % guid_read).str());

  input_configuration.read(*in_stream);
  output_configuration.read(*in_stream);

  input_neuron_count = input_configuration.get_neuron_count();
  // hard coded
  input_sparse_neuron_count = input_configuration.dimension_sizes[0];
  output_neuron_count = output_configuration.get_neuron_count();
  // hard coded
  output_sparse_neuron_count = 1;

  unsigned int type_code_read;
  in_stream->read(reinterpret_cast<char*>(&type_code_read),
                  sizeof(type_code_read));
  type_code = static_cast<neuron_data_type::input_type>(type_code_read);

  in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

  reset_pos = in_stream->tellg();
}

supervised_sparse_data_stream_reader::~supervised_sparse_data_stream_reader() {
}

void supervised_sparse_data_stream_reader::reset() {
  in_stream->seekg(reset_pos);

  entry_read_count = 0;
}

bool supervised_sparse_data_stream_reader::read(void * input_neurons,
                                                float * output_neurons) {
  if (!entry_available()) {
    return false;
  }

  // hard coded float* input, should be if else depending on input type
  float* input_sparse_neurons = new float[input_sparse_neuron_count];
  float* output_sparse_neurons = new float[output_sparse_neuron_count];

  if (input_neurons) {
    in_stream->read(reinterpret_cast<char*>(input_sparse_neurons),
                    get_input_neuron_elem_size() * input_sparse_neuron_count);
    sparse_input_to_nonsparse_input(input_sparse_neurons, input_neurons);
  } else {
    in_stream->seekg(get_input_neuron_elem_size() * input_sparse_neuron_count,
                     std::ios_base::cur);
  }

  if (output_neurons) {
    in_stream->read(reinterpret_cast<char*>(output_sparse_neurons),
                    sizeof(float) * output_sparse_neuron_count);
    sparse_output_to_nonsparse_output(output_sparse_neurons, output_neurons);
  } else {
    in_stream->seekg(sizeof(float) * output_sparse_neuron_count,
                     std::ios_base::cur);
  }

  entry_read_count++;

  return true;
}

bool supervised_sparse_data_stream_reader::raw_read(
    std::vector<unsigned char>& all_elems) {
  if (!entry_available()) {
    return false;
  }

  size_t bytes_to_read =
      get_input_neuron_elem_size() * input_sparse_neuron_count +
      sizeof(float) * output_sparse_neuron_count;
  all_elems.resize(bytes_to_read);
  in_stream->read(
      reinterpret_cast<char*>(&(*all_elems.begin())), bytes_to_read);

  return true;
}

bool supervised_sparse_data_stream_reader::entry_available() {
  return (entry_read_count < entry_count);
}

void supervised_sparse_data_stream_reader::rewind(unsigned int entry_id) {
  in_stream->seekg(
      reset_pos +
      (std::istream::off_type)entry_id *
      (std::istream::off_type)(
          (get_input_neuron_elem_size() * input_sparse_neuron_count) +
          (sizeof(float) * output_sparse_neuron_count)),
      std::ios::beg);

    entry_read_count = entry_id;
}

// hard coded float* input, should be if else depending on input type
void supervised_sparse_data_stream_reader::sparse_input_to_nonsparse_input(
    float* input_sparse_neurons, void* input_neurons) {
  // hard coded float* cast
  float* input_neurons_float = static_cast<float*>(input_neurons);
  // this is needed, otherwise input_neurons is not initialized correctly
  memset(input_neurons_float, 0, sizeof(float) * input_neuron_count);
  for (unsigned int i = 0; i < input_sparse_neuron_count; ++i) {
    float wordIdFloat = input_sparse_neurons[i];
    unsigned int wordId = round_float_to_unsigned_int(wordIdFloat);
    // in the nnjm paper, the dense representation is a vector of 14 * 16000.
    // for this library, it's 16000 * 14.
    input_neurons_float[input_sparse_neuron_count * wordId + i] = 1;
  }
}

void supervised_sparse_data_stream_reader::sparse_output_to_nonsparse_output(
    float* output_sparse_neurons, float* output_neurons) {
  // this is needed, otherwise output_neurons is not initialized correctly
  memset(output_neurons, 0, sizeof(float) * output_neuron_count);
  float wordIdFloat = *output_sparse_neurons;
  unsigned int wordId = round_float_to_unsigned_int(wordIdFloat);
  output_neurons[wordId] = 1;
}

unsigned int supervised_sparse_data_stream_reader::round_float_to_unsigned_int(
    const float wordIdFloat) {
  return static_cast<unsigned int>(wordIdFloat + 0.5);
}

} // namespace nnforge
