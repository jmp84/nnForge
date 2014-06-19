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

#include <boost/uuid/uuid.hpp>

namespace nnforge {

/**
 * boost uuid that identifies training data written and read with
 * supervised_sparse_data_stream_writer and supervised_sparse_data_stream_reader
 */
class supervised_sparse_data_stream_schema {

public:
  static const boost::uuids::uuid supervised_sparse_data_stream_guid;

private:
  supervised_sparse_data_stream_schema();
  supervised_sparse_data_stream_schema(
      const supervised_sparse_data_stream_schema&);
  supervised_sparse_data_stream_schema& operator =(
      const supervised_sparse_data_stream_schema&);
};

} // namespace nnforge
