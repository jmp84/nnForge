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

#include "softmax_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../neural_network_exception.h"
#include "neural_network_cuda_exception.h"

#include "util_cuda.h"

/**
 * computes max using the reduce algorithm
 */
__global__ void max_kernel(
		const float * __restrict input,
		float * __restrict output,
		unsigned int size) {

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	// global to shared memory
	// if out of bounds, use identity elt (here it's a max over exps,
	// so zero is fine)
	extern __shared__ float sdata[];
	sdata[tid] = id < size ? input[id] : 0.0F;
	__syncthreads();

	// reduction algorithm
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = max(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();
	}

	// store the block max into the output
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

/**
 * computes exp(x_i - m) over a range
 * m is max(x_i)
 */
__global__ void exponential_minus_max_kernel(
		const float * __restrict input,
		float * __restrict output,
		const float* max_exp,
		unsigned int size) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	// check out of bounds
	if (id >= size) {
		return;
	}
	output[id] = expf(input[id] - *max_exp);
}

/**
 * computes sum using the reduce algorithm
 */
__global__ void sum_kernel(
		const float * __restrict input,
		float * __restrict output,
		unsigned int size) {

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	// global to shared memory
	// if out of bounds, use identity elt for sum, i.e. 0
	extern __shared__ float sdata[];
	sdata[tid] = id < size ? input[id] : 0.0F;
	__syncthreads();

	// reduction algorithm
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// store the block sum into the output
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

/**
 * divides all elements in a range by a normalizing constant
 */
__global__ void normalize_kernel(
		float * __restrict output,
		float* sum,
		unsigned int size) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	// check out of bounds
	if (id >= size) {
		return;
	}

	output[id] /= (*sum);
}

__global__ void dot_product_kernel(
		const float * __restrict input1,
		const float * __restrict input2,
		float* __restrict output,
		unsigned int size) {

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	// global to shared memory, elt by elt multiplication
	// if out of bounds, use identity elt
	extern __shared__ float sdata[];
	sdata[tid] = (id < size) ? (input1[id] * input2[id]) : 0.0F;
	__syncthreads();

	// reduction algorithm
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// store the block dot product into the output
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

__global__ void elementwise_mult_minus_sum_kernel(
		const float* __restrict input,
		float* __restrict output,
		const float* sum,
		unsigned int size) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	// check out of bounds
	if (id >= size) {
		return;
	}

	output[id] = input[id] * (output[id] - *sum);
}

__global__ void softmax_deriviative_upd_kernel(
	float * __restrict errors,
	const float * __restrict output_neurons,
	int feature_map_count,
	int neuron_count_per_feature_map,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int entry_id = blockIdx.y * blockDim.y + threadIdx.y;
	if ((neuron_id < neuron_count_per_feature_map) && (entry_id < entry_count))
	{
		int initial_offset = entry_id * feature_map_count * neuron_count_per_feature_map + neuron_id;
		float sum = 0.0F;
		const float * current_output_neurons = output_neurons + initial_offset;
		const float * current_output_errors = errors + initial_offset;
		for(int i = 0; i < feature_map_count; ++i)
		{
			sum += __load_nc(current_output_neurons) * __load_nc(current_output_errors);
			current_output_neurons += neuron_count_per_feature_map;
			current_output_errors += neuron_count_per_feature_map;
		}

		current_output_neurons = output_neurons + initial_offset;
		float * current_errors = errors + initial_offset;
		for(int i = 0; i < feature_map_count; ++i)
		{
			*current_errors = __load_nc(current_output_neurons) * (__load_nc(current_errors) - sum);
			current_output_neurons += neuron_count_per_feature_map;
			current_errors += neuron_count_per_feature_map;
		}
	}
}

namespace nnforge
{
	namespace cuda
	{
		softmax_layer_updater_cuda::softmax_layer_updater_cuda()
		{
		}

		softmax_layer_updater_cuda::~softmax_layer_updater_cuda()
		{
		}

		// from http://graphics.stanford.edu/~seander/bithacks.html
		unsigned int next_power_of_two(unsigned int const v) {
			if (sizeof(unsigned int) != 4) {
				throw neural_network_exception(
						"Size of unsigned int is not 4, needed for bit hacks");
			}
			if (v > 1) {
				float f = (float)v;
				unsigned int const t = 1U << ((*(unsigned int *)&f >> 23) - 0x7f);
				return t << (t < v);
			} else {
				return 1;
			}
			// should not read here
			return 0;
		}

		/**
		 * computes softmax
		 * softmax(x_1, ..., x_n) = (x_1/sum(x_i), ..., x_n/sum(x_i)
		 * to avoid underflow/overflow, we use the log sum exp trick
		 * (http://math.stackexchange.com/questions/648514/preventing-underflow-log-sum-exp-trick)
		 * softmax is parallelized for speed (in nnjm paper, layer size is 32000)
		 */
		void softmax_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			// max number of thread on K20 gpu
			const unsigned int NUM_THREADS = 1024;
			unsigned int numThreads(NUM_THREADS);
			unsigned int layer_size =
					input_elem_count_per_feature_map *
					input_configuration_specific.feature_map_count;
			unsigned int remainder = layer_size % NUM_THREADS;
			unsigned int quotient = layer_size / NUM_THREADS;
			unsigned int numBlocks = remainder == 0 ? quotient : quotient + 1;
			unsigned int size_shared_memory = sizeof(float) * NUM_THREADS;
			unsigned int size_global_memory = sizeof(float) * layer_size;
			unsigned int numBlocksPower2 = next_power_of_two(numBlocks);
			unsigned int size_intermediate_shared_memory =
					sizeof(float) * numBlocksPower2;
			if (offset_input_entry_id > 0)
				throw neural_network_exception("softmax_layer_updater_cuda is not able to run using offset");


			float* max_intermediate;
			float* max_global;
			float* exp_minus_max;
			float* sum_exps_intermediate;
			float* sum_exps;
			cuda_safe_call(cudaMalloc(
					(void **) &max_intermediate,
					sizeof(float) * numBlocks));
			cuda_safe_call(cudaMalloc((void **) &max_global, sizeof(float)));
			cuda_safe_call(cudaMalloc((void **) &exp_minus_max, size_global_memory));
			cuda_safe_call(cudaMalloc(
					(void **) &sum_exps_intermediate,
					sizeof(float) * numBlocks));
			cuda_safe_call(cudaMalloc((void **) &sum_exps, sizeof(float)));

			// compute max
			// only 2 calls, assume size <= than 1024 x 1024
			// first: max per block
			max_kernel<<<numBlocks, numThreads, size_shared_memory>>>(
					*input_neurons_buffer,
					max_intermediate,
					layer_size);
			// second: global max
			max_kernel<<<1, numBlocksPower2, size_intermediate_shared_memory>>>(
					max_intermediate,
					max_global,
					numBlocks);

			// compute exp subtracting the max
			exponential_minus_max_kernel<<<numBlocks, numThreads>>>(
					*input_neurons_buffer,
					*output_neurons_buffer,
					max_global,
					layer_size);

			// compute the sum
			// only 2 calls, assume size <= 1024 x 1024
			// first: sum per block
			sum_kernel<<<numBlocks, numThreads, size_shared_memory>>>(
					*output_neurons_buffer,
					sum_exps_intermediate,
					layer_size);
			// second: global sum
			sum_kernel<<<1, numBlocksPower2, size_intermediate_shared_memory>>>(
					sum_exps_intermediate,
					sum_exps,
					numBlocks);

			// finally normalize
			normalize_kernel<<<numBlocks, numThreads>>> (
					*output_neurons_buffer,
					sum_exps,
					layer_size);

			// clean up
			cuda_safe_call(cudaFree(max_intermediate));
			cuda_safe_call(cudaFree(max_global));
			cuda_safe_call(cudaFree(exp_minus_max));
			cuda_safe_call(cudaFree(sum_exps_intermediate));
			cuda_safe_call(cudaFree(sum_exps));
		}

		void softmax_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				entry_count,
				1);

			// max number of thread on K20 gpu
			const unsigned int NUM_THREADS = 1024;
			unsigned int numThreads(NUM_THREADS);
			unsigned int layer_size =
					input_elem_count_per_feature_map *
					input_configuration_specific.feature_map_count;
			unsigned int remainder = layer_size % NUM_THREADS;
			unsigned int quotient = layer_size / NUM_THREADS;
			unsigned int numBlocks = remainder == 0 ? quotient : quotient + 1;
			unsigned int size_shared_memory = sizeof(float) * NUM_THREADS;
			unsigned int size_global_memory = sizeof(float) * layer_size;
			unsigned int numBlocksPower2 = next_power_of_two(numBlocks);
			unsigned int size_intermediate_shared_memory =
					sizeof(float) * numBlocksPower2;

			float* intermediate_dot_product;
			float* final_dot_product;
			cudaMalloc((void**) &intermediate_dot_product, sizeof(float) * numBlocks);
			cudaMalloc((void**) &final_dot_product, sizeof(float));

			// compute dot product
			dot_product_kernel<<<numBlocks, numThreads, size_shared_memory>>>(
					*output_errors_buffer,
					*output_neurons_buffer,
					intermediate_dot_product,
					layer_size);
			sum_kernel<<<1, numBlocksPower2, size_intermediate_shared_memory>>>(
					intermediate_dot_product,
					final_dot_product,
					numBlocks);

			// compute elementwise multiplication, dot product
			// subtracted from second operand
			elementwise_mult_minus_sum_kernel<<<numBlocks, numThreads>>>(
					*output_neurons_buffer,
					*output_errors_buffer,
					final_dot_product,
					layer_size);

			// clean up
			cuda_safe_call(cudaFree(intermediate_dot_product));
			cuda_safe_call(cudaFree(final_dot_product));
		}

		bool softmax_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}
	}
}
