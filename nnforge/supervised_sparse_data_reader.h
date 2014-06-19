/*
 * supervised_sparse_data_reader.h
 *
 *  Created on: 16 Jun 2014
 *      Author: jmp84
 */

#pragma once

namespace nnforge {

class supervised_sparse_data_reader : public supervised_data_reader {

public:
  supervised_sparse_data_reader(nnforge_shared_ptr<std::istream> input_stream);

  virtual ~supervised_sparse_data_reader();

  virtual void reset();

  virtual bool read(
        void * input_neurons,
        float * output_neurons);

  virtual bool raw_read(std::vector<unsigned char>& all_elems);

  virtual layer_configuration_specific get_input_configuration() const;

  virtual layer_configuration_specific get_output_configuration() const;

  virtual neuron_data_type::input_type get_input_type() const;

  virtual unsigned int get_entry_count() const;

  virtual void rewind(unsigned int entry_id);

};

}
