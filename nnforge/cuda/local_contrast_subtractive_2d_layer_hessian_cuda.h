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

#include "layer_hessian_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class local_contrast_subtractive_2d_layer_hessian_cuda : public layer_hessian_cuda
		{
		public:
			local_contrast_subtractive_2d_layer_hessian_cuda();

			virtual ~local_contrast_subtractive_2d_layer_hessian_cuda();

			virtual void enqueue_test(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

		protected:
			virtual bool is_in_place_backprop() const;

			virtual void hessian_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

		private:
			int affected_feature_map_count;
			int unaffected_feature_map_count;
			std::vector<int> half_window_sizes;
			float central_mult;
		};
	}
}
