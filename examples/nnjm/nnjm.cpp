/**
 * TODO header
 */

#include <iostream>
#include <stdio.h>

#define BOOST_LOG_DYN_LINK
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#ifdef NNFORGE_CUDA_BACKEND_ENABLED
#include <nnforge/cuda/cuda.h>
#else
#include <nnforge/plain/plain.h>
#endif
#include <examples/nnjm/nnjm_toolset.h>

/**
 * Initializes logging.
 * We use trivial logging.
 * Only messages with severity greater or equal to info are logged.
 */
void initLog()
{
	boost::log::core::get()->set_filter
			(
					boost::log::trivial::severity >= boost::log::trivial::info
			);
}

/**
 * Reimplementation of the nnjm paper (BBN paper at ACL 2014 on
 * neural network joint model for machine translation, to appear)
 */
int main(int argc, char* argv[])
{
	initLog();
	try
	{
		#ifdef NNFORGE_CUDA_BACKEND_ENABLED
		nnforge::cuda::cuda::init();
		#else
		nnforge::plain::plain::init();
		#endif

		#ifdef NNFORGE_CUDA_BACKEND_ENABLED
		nnjm::NnjmToolset ts(
				nnforge::factory_generator_smart_ptr(
						new nnforge::cuda::factory_generator_cuda()));
		#else
		nnjm::NnjmToolset ts(
				nnforge::factory_generator_smart_ptr(
						new nnforge::plain::factory_generator_plain()));
		#endif

		if (ts.parse(argc, argv))
		{
			const std::string& action = ts.get_action();
			if (action == "create_vocab")
			{
				ts.initVocab();
				ts.storeVocab();
				return 0;
			}
			if (
					action == "prepare_training_data" ||
					action == "prepare_testing_data" ||
					action == "prepare_validating_data" ||
					action == "create")
			{
				//init vocabulary passing
				ts.loadVocab();
			}
			ts.do_action();
		}
	}
	catch (const std::exception& e)
	{
		std::cout << "Exception caught: " << e.what() << std::endl;
		return 1;
	}
	return 0;
}
