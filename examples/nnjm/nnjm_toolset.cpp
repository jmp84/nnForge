/**
 * TODO header
 */

#include <examples/nnjm/nnjm_toolset.h>

#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#define BOOST_LOG_DYN_LINK
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/tokenizer.hpp>

#include <examples/nnjm/vocab.h>
#include <nnforge/supervised_sparse_data_stream_reader.h>
#include <nnforge/supervised_sparse_data_stream_writer.h>

namespace nnjm
{

	NnjmToolset::NnjmToolset(nnforge::factory_generator_smart_ptr factory)
	: nnforge::neural_network_toolset(factory)
	{
	}

	NnjmToolset::~NnjmToolset()
	{
	}

	void NnjmToolset::storeVocab()
	{
		boost::filesystem::path sourceInputVocabPath =
				get_working_data_folder() / sourceInputVocabularyFileName_;
		boost::filesystem::path targetInputVocabPath =
				get_working_data_folder() / targetInputVocabularyFileName_;
		boost::filesystem::path targetOutputVocabPath =
				get_working_data_folder() / targetOutputVocabularyFileName_;
		boost::filesystem::ofstream sourceInputVocabFile(sourceInputVocabPath);
		boost::filesystem::ofstream targetInputVocabFile(targetInputVocabPath);
		boost::filesystem::ofstream targetOutputVocabFile(
				targetOutputVocabPath);
		vocab_->store(
				sourceInputVocabFile,
				targetInputVocabFile,
				targetOutputVocabFile);
	}

	bool NnjmToolset::loadVocab()
	{
		boost::filesystem::path sourceInputVocabPath =
				get_working_data_folder() / sourceInputVocabularyFileName_;
		boost::filesystem::path targetInputVocabPath =
				get_working_data_folder() / targetInputVocabularyFileName_;
		boost::filesystem::path targetOutputVocabPath =
				get_working_data_folder() / targetOutputVocabularyFileName_;
		boost::filesystem::ifstream sourceInputVocabFile(sourceInputVocabPath);
		boost::filesystem::ifstream targetInputVocabFile(targetInputVocabPath);
		boost::filesystem::ifstream targetOutputVocabFile(
				targetOutputVocabPath);
		if (
				!sourceInputVocabFile.is_open() ||
				!targetInputVocabFile.is_open() ||
				!targetOutputVocabFile.is_open())
		{
			BOOST_LOG_TRIVIAL(warning) << "Default vocabulary files not found. "
					"Creating...";
			this->initVocab();
			return false;
		}
		vocab_.reset(new Vocab(
				sourceInputVocabFile,
				targetInputVocabFile,
				targetOutputVocabFile,
				inputVocabSize_,
				outputVocabSize_));
		return true;
	}

	void NnjmToolset::initVocab()
	{
		boost::filesystem::path sourceTextPath =
				get_input_data_folder() / sourceTextFileName_;
		boost::filesystem::path targetTextPath =
				get_input_data_folder() / targetTextFileName_;
		// TODO handle bad ifstream, exceptions with getline doesn't work
		boost::filesystem::ifstream sourceTextFile(sourceTextPath);
		boost::filesystem::ifstream targetTextFile(targetTextPath);
		if (!vocab_)
		{
			vocab_.reset(new Vocab(
					sourceTextFile,
					targetTextFile,
					inputVocabSize_,
					outputVocabSize_));
		}
	}

	std::vector<nnforge::string_option> NnjmToolset::get_string_options()
	{
		std::vector<nnforge::string_option> res;
		res.push_back(
				nnforge::string_option(
						"source-input-vocab-filename",
						&sourceInputVocabularyFileName_,
						"sourceInput.vcb",
						"Source input vocabulary file name"
				));
		res.push_back(
				nnforge::string_option(
						"target-input-vocab-filename",
						&targetInputVocabularyFileName_,
						"targetInput.vcb",
						"Target input vocabulary file name"
				));
		res.push_back(
				nnforge::string_option(
						"target-output-vocab-filename",
						&targetOutputVocabularyFileName_,
						"targetOutput.vcb",
						"Target output vocabulary file name"
				));
		res.push_back(
				nnforge::string_option(
						"source-text",
						&sourceTextFileName_,
						"",
						"Source text (one sentence per line)"));
		res.push_back(
				nnforge::string_option(
						"source-text-test",
						&sourceTextTestFileName_,
						"",
						"Source text for testing (one sentence per line)"));
		res.push_back(
				nnforge::string_option(
						"source-text-validate",
						&sourceTextValidateFileName_,
						"",
						"Source text for validating (one sentence per line)"));
		res.push_back(
				nnforge::string_option(
						"target-text",
						&targetTextFileName_,
						"",
						"Target text (one sentence per line)"));
		res.push_back(
				nnforge::string_option(
						"target-text-test",
						&targetTextTestFileName_,
						"",
						"Target text for testing (one sentence per line)"));
		res.push_back(
				nnforge::string_option(
						"target-text-validate",
						&targetTextValidateFileName_,
						"",
						"Target text for validating (one sentence per line)"));
		res.push_back(
				nnforge::string_option(
						"alignment",
						&alignmentFileName_,
						"",
						"Word alignment (Berkeley format: one alignment per "
						"line, formatted as "
						"'src_pos-trg_pos<SPACE>src_pos-trg_pos etc.')"));
		res.push_back(
				nnforge::string_option(
						"alignment-test",
						&alignmentTestFileName_,
						"",
						"Word alignment for testing (Berkeley format: one "
						"alignment per line, formatted as "
						"'src_pos-trg_pos<SPACE>src_pos-trg_pos etc.')"));
		res.push_back(
				nnforge::string_option(
						"alignment-validate",
						&alignmentValidateFileName_,
						"",
						"Word alignment for validating (Berkeley format: one "
						"alignment per line, formatted as "
						"'src_pos-trg_pos<SPACE>src_pos-trg_pos etc.')"));
		res.push_back(
				nnforge::string_option(
						"output-training-file",
						&outputTrainingFileName_,
						"training.sdt",
						"Output training file name where training data will be "
						"stored in binary format and fed to neural network "
						"training."));
		res.push_back(
				nnforge::string_option(
						"validating-data-file",
						&validatingDataFileName_,
						"validating.sdt",
						"Output validation file name where part of the "
						"training data will be stored in binary format to be "
						"used for validation in neural network training."));
		res.push_back(
				nnforge::string_option(
						"output-testing-file",
						&outputTestingFileName_,
						"testing.udt",
						"Output testing file name where testing data will be "
						"stored in binary format and fed to a trained neural "
						"network."));
		res.push_back(
				nnforge::string_option(
						"output-testing-label-file",
						&outputTestingLabelFileName_,
						"testing_rec_ids.txt",
						"Output testing label file name containing target "
						"words."));
		res.push_back(
				nnforge::string_option(
						"output-prediction-file",
						&outputPredictionsFileName_,
						"output.csv",
						"Output predictions file name in testing."));
		res.push_back(
				nnforge::string_option(
						"error-function",
						&errorFunction_,
						"nll",
						"Error function (mse, negative log-likelihood, "
						"cross-entropy, etc.). "
						"Possible values are 'nll' (default, stands for "
						"negative log-likelihood), 'mse'."));
		return res;
	}

	std::vector<nnforge::int_option> NnjmToolset::get_int_options()
	{
		std::vector<nnforge::int_option> res;
		res.push_back(
				nnforge::int_option(
						"output-vocab-size",
						&outputVocabSize_,
						32000,
						"Output vocabulary size, defines size of the output "
						"layer."));
		res.push_back(
				nnforge::int_option(
						"input-vocab-size",
						&inputVocabSize_,
						16000,
						"Input vocabulary size, defines the size of the input "
						"layer together with --target-ngram-size=. No "
						"distinction is made between source and target. This "
						"may be extended for more flexibility."));
		res.push_back(
				nnforge::int_option(
						"target-ngram-size",
						&targetNgramSize_,
						4,
						"Target n-gram size in the nnjm model, defines the "
						"size of the input layer together with "
						"--input-vocab-size=."));
		res.push_back(
				nnforge::int_option(
						"source-window-size",
						&sourceWindowSize_,
						11,
						"Source window size in the nnjm model."));
		res.push_back(
				nnforge::int_option(
						"continuous-space-dimension",
						&continuousSpaceDimension_,
						192,
						"Continuous space dimension where to project input "
						"words (192 in the nnjm paper); defines the size of "
						"the first layer together with --target-ngram-size= "
						"and --source-window-size=."));
		res.push_back(
				nnforge::int_option(
						"hidden-layer-size",
						&hiddenLayerSize_,
						512,
						"Hidden layer size (512 in the nnjm paper)."));
		return res;
	}

	void NnjmToolset::prepare_training_data()
	{
		// setup input configuration
		nnforge::layer_configuration_specific inputConfiguration;
		// Each word (source or target) is represented as a vector of size the
		// input vocabulary size with all zeros except for a 1 at the index for
		// this word. Note that the vocab size is constrained to be the same
		// for source and target. Similarly, the linear transformation that
		// converts words to a continuous representation ("shared mapping
		// layer") is the same for source and target words. This may be a
		// limitation. The context is represented as a 1D vector with 14
		// elements (1 element per context word) with input vocab size feature
		// maps per element.
		inputConfiguration.feature_map_count = vocab_->getInputVocabSize();
		// The context consists of target history words and source words.
		// In the nnjm paper, this is 3 + 11 = 14 context words
		inputConfiguration.dimension_sizes.push_back(
				targetNgramSize_ - 1 + sourceWindowSize_);

		// The output is a 1 dimensional vector with
		// output vocabulary size feature maps (32000 in the nnjm paper).
		nnforge::layer_configuration_specific outputConfiguration;
		outputConfiguration.feature_map_count = vocab_->getOutputVocabSize();
		outputConfiguration.dimension_sizes.push_back(1);

		// configure the training data writer
		// TODO make a function
		boost::scoped_ptr<std::ofstream> trainingDataWriterTxt;
		boost::filesystem::path trainingFilePath =
				get_working_data_folder() / outputTrainingFileName_;
		BOOST_LOG_TRIVIAL(info) <<
				"Writing training data to " << trainingFilePath.string();
		nnforge_shared_ptr<std::ofstream> trainingFile(
				new boost::filesystem::ofstream(
						trainingFilePath,
						std::ios_base::out |
						std::ios_base::binary |
						std::ios_base::trunc));
		nnforge::supervised_sparse_data_stream_writer trainingDataWriter(
				trainingFile,
				inputConfiguration,
				outputConfiguration);
		boost::filesystem::path trainingFilePathTxt(trainingFilePath);
		trainingFilePathTxt.replace_extension(boost::filesystem::path(".txt"));
		trainingDataWriterTxt.reset(
				new boost::filesystem::ofstream(trainingFilePathTxt));

		// configure input data, output data to be written
		std::vector<float> inputData(targetNgramSize_ - 1);
		std::vector<float> outputData(1);
		unsigned int trainingEntryCountWritten = 0;
		std::vector<WordId> trainingNgram(targetNgramSize_ + sourceWindowSize_);

		// open files containing training data: source, target and alignment
		boost::filesystem::path sourceTextPath =
				get_input_data_folder() / sourceTextFileName_;
		boost::filesystem::path targetTextPath =
				get_input_data_folder() / targetTextFileName_;
		boost::filesystem::path alignmentPath =
				get_input_data_folder() / alignmentFileName_;
		boost::filesystem::ifstream sourceTextFile(sourceTextPath);
		boost::filesystem::ifstream targetTextFile(targetTextPath);
		boost::filesystem::ifstream alignmentFile(alignmentPath);

		// read the training data
		std::string sourceLine, targetLine, alignmentLine;
		while (
				std::getline(sourceTextFile, sourceLine) &&
				std::getline(targetTextFile, targetLine) &&
				std::getline(alignmentFile, alignmentLine))
		{
			BOOST_LOG_TRIVIAL(debug) << "processing training instance:" <<
					std::endl << "source sentence: " << sourceLine << std::endl
					<< "target sentence: " << targetLine << std::endl <<
					"alignment: " << alignmentLine;
			std::vector<std::string> sourceTokens, targetTokens;
			boost::split(sourceTokens, sourceLine, boost::is_any_of(" "));
			boost::split(targetTokens, targetLine, boost::is_any_of(" "));
			std::vector<std::vector<std::size_t> > target2SourceAlignment(
					targetTokens.size());
			getTarget2SourceAlignment(alignmentLine, &target2SourceAlignment);
			*trainingDataWriterTxt << targetLine << std::endl;

			// targetTokenIndex includes targetTokens.size() for the end of
			// sentence marker; the training data is assumed not to have
			// sentence markers
			for (
					std::size_t targetTokenIndex = 0;
					targetTokenIndex <= targetTokens.size();
					++targetTokenIndex)
			{
				prepareTrainingInstance(
						targetTokenIndex,
						sourceTokens,
						targetTokens,
						target2SourceAlignment,
						&trainingNgram);
				convertToInputSparseData(trainingNgram, &inputData);
				convertToOutputSparseData(trainingNgram, &outputData);

				trainingEntryCountWritten++;
				trainingDataWriter.write(
						&(*inputData.begin()),
						&(*outputData.begin()));
			}
		}
		BOOST_LOG_TRIVIAL(info) <<
				"Training entries written: " << trainingEntryCountWritten;
	}

	void NnjmToolset::prepare_validating_data()
	{
		// setup input configuration
		nnforge::layer_configuration_specific inputConfiguration;
		// Each word (source or target) is represented as a vector of size the
		// input vocabulary size with all zeros except for a 1 at the index for
		// this word. Note that the vocab size is constrained to be the same
		// for source and target. Similarly, the linear transformation that
		// converts words to a continuous representation ("shared mapping
		// layer") is the same for source and target words. This may be a
		// limitation. The context is represented as a 1D vector with 14
		// elements (1 element per context word) with input vocab size feature
		// maps per element.
		inputConfiguration.feature_map_count = vocab_->getInputVocabSize();
		// The context consists of target history words and source words.
		// In the nnjm paper, this is 3 + 11 = 14 context words
		inputConfiguration.dimension_sizes.push_back(
				targetNgramSize_ - 1 + sourceWindowSize_);

		// The output is a 1 dimensional vector with
		// output vocabulary size feature maps (32000 in the nnjm paper).
		nnforge::layer_configuration_specific outputConfiguration;
		outputConfiguration.feature_map_count = vocab_->getOutputVocabSize();
		outputConfiguration.dimension_sizes.push_back(1);

		// configure the validation data writer
		// TODO make a function
		nnforge::supervised_sparse_data_stream_writer_smart_ptr
		validatingDataWriter;
		boost::scoped_ptr<std::ofstream> validatingDataWriterTxt;
		boost::filesystem::path validatingFilePath =
				get_working_data_folder() / validatingDataFileName_;
		BOOST_LOG_TRIVIAL(info) <<
				"Writing validating data to " << validatingFilePath.string();
		nnforge_shared_ptr<std::ofstream> validatingFile(
				new boost::filesystem::ofstream(
						validatingFilePath,
						std::ios_base::out |
						std::ios_base::binary |
						std::ios_base::trunc));
		validatingDataWriter =
				nnforge::supervised_sparse_data_stream_writer_smart_ptr(
						new nnforge::supervised_sparse_data_stream_writer(
								validatingFile,
								inputConfiguration,
								outputConfiguration));
		boost::filesystem::path validatingFilePathTxt(validatingFilePath);
		validatingFilePathTxt.replace_extension(
				boost::filesystem::path(".txt"));
		validatingDataWriterTxt.reset(
				new boost::filesystem::ofstream(validatingFilePathTxt));

		// configure input data, output data to be written
		std::vector<float> inputData(targetNgramSize_ - 1);
		std::vector<float> outputData(1);
		unsigned int validatingEntryCountWritten = 0;
		std::vector<WordId> trainingNgram(targetNgramSize_ + sourceWindowSize_);

		// open files containing validating data: source, target and alignment
		boost::filesystem::path sourceTextPath =
				get_input_data_folder() / sourceTextValidateFileName_;
		boost::filesystem::path targetTextPath =
				get_input_data_folder() / targetTextValidateFileName_;
		boost::filesystem::path alignmentPath =
				get_input_data_folder() / alignmentValidateFileName_;
		boost::filesystem::ifstream sourceTextFile(sourceTextPath);
		boost::filesystem::ifstream targetTextFile(targetTextPath);
		boost::filesystem::ifstream alignmentFile(alignmentPath);

		// read the validating data
		std::string sourceLine, targetLine, alignmentLine;
		while (
				std::getline(sourceTextFile, sourceLine) &&
				std::getline(targetTextFile, targetLine) &&
				std::getline(alignmentFile, alignmentLine))
		{
			BOOST_LOG_TRIVIAL(debug) << "processing validating instance:" <<
					std::endl << "source sentence: " << sourceLine <<
					std::endl << "target sentence: " << targetLine <<
					std::endl << "alignment: " << alignmentLine;
			std::vector<std::string> sourceTokens, targetTokens;
			boost::split(sourceTokens, sourceLine, boost::is_any_of(" "));
			boost::split(targetTokens, targetLine, boost::is_any_of(" "));
			std::vector<std::vector<std::size_t> > target2SourceAlignment(
					targetTokens.size());
			getTarget2SourceAlignment(alignmentLine, &target2SourceAlignment);
			*validatingDataWriterTxt << targetLine << std::endl;
			// targetTokenIndex includes targetTokens.size() for the end of
			// sentence marker; the validating data is assumed not to have
			// sentence markers
			for (
					std::size_t targetTokenIndex = 0;
					targetTokenIndex <= targetTokens.size();
					++targetTokenIndex)
			{
				prepareTrainingInstance(
						targetTokenIndex,
						sourceTokens,
						targetTokens,
						target2SourceAlignment,
						&trainingNgram);
				convertToInputSparseData(trainingNgram, &inputData);
				convertToOutputSparseData(trainingNgram, &outputData);

				validatingEntryCountWritten++;
				validatingDataWriter->write(
						&(*inputData.begin()),
						&(*outputData.begin()));
			}
		}

		BOOST_LOG_TRIVIAL(info) <<
				"Validation entries written: " << validatingEntryCountWritten;
	}

	nnforge::network_schema_smart_ptr NnjmToolset::get_schema() const
	{
		nnforge::network_schema_smart_ptr schema(new nnforge::network_schema());

		// shared mapping layer: projects words to continuous space
		schema->add_layer(
				nnforge::const_layer_smart_ptr(
						new nnforge::convolution_layer(
								std::vector<unsigned int>(1, 1),
								vocab_->getInputVocabSize(),
								continuousSpaceDimension_)));

		// tanh layer
		schema->add_layer(
				nnforge::const_layer_smart_ptr(
						new nnforge::convolution_layer(
								std::vector<unsigned int>(
										1,
										targetNgramSize_ - 1 + sourceWindowSize_
										),
										continuousSpaceDimension_,
										hiddenLayerSize_)));
		schema->add_layer(
				nnforge::const_layer_smart_ptr(
						new nnforge::hyperbolic_tangent_layer()));

		// second tanh layer
		// second tanh layer is optional for speed
		schema->add_layer(
				nnforge::const_layer_smart_ptr(
						new nnforge::convolution_layer(
								std::vector<unsigned int>(1, 1),
								hiddenLayerSize_,
								hiddenLayerSize_)));
		schema->add_layer(
				nnforge::const_layer_smart_ptr(
						new nnforge::hyperbolic_tangent_layer()));

		// output layer
		schema->add_layer(
				nnforge::const_layer_smart_ptr(
						new nnforge::convolution_layer(
								std::vector<unsigned int>(1, 1),
								hiddenLayerSize_,
								vocab_->getOutputVocabSize())));
		schema->add_layer(
				nnforge::const_layer_smart_ptr(
						new nnforge::softmax_layer()));

		return schema;
	}

	nnforge::network_output_type::output_type
	NnjmToolset::get_network_output_type() const
	{
		return nnforge::network_output_type::type_classifier;
	}

	nnforge::const_error_function_smart_ptr
	NnjmToolset::get_error_function() const
	{
		if (errorFunction_ == "nll")
		{
			return nnforge::error_function_smart_ptr(
					new nnforge::negative_log_likelihood_error_function());
		}
		if (errorFunction_ == "mse")
		{
			return nnforge::error_function_smart_ptr(
					new nnforge::mse_error_function());
		}
		BOOST_LOG_TRIVIAL(fatal) <<
				"Unknown or unimplemented error function: " << errorFunction_ <<
				" (possible values are 'ce' and 'mse').";
		return nnforge::error_function_smart_ptr();
	}

	nnforge::supervised_data_reader_smart_ptr
	NnjmToolset::get_original_training_data_reader(
			const boost::filesystem::path& path) const
	{
		nnforge_shared_ptr<std::istream> in(
				new boost::filesystem::ifstream(
						path,
						std::ios_base::in |
						std::ios_base::binary));
		nnforge::supervised_data_reader_smart_ptr reader(
				new nnforge::supervised_sparse_data_stream_reader(in));
		return reader;
	}

	nnforge::data_writer_smart_ptr
	NnjmToolset::get_randomized_training_data_writer(
			nnforge::supervised_data_reader& reader,
			const boost::filesystem::path& path) const
	{
		nnforge_shared_ptr<std::ostream> out(
				new boost::filesystem::ofstream(
						path,
						std::ios_base::out |
						std::ios_base::binary |
						std::ios_base::trunc));
		nnforge::supervised_sparse_data_stream_reader& typed_reader =
				dynamic_cast<nnforge::supervised_sparse_data_stream_reader&>(
						reader);
		nnforge::data_writer_smart_ptr writer(
				new nnforge::supervised_sparse_data_stream_writer(
						out,
						typed_reader.get_input_configuration(),
						typed_reader.get_output_configuration(),
						typed_reader.get_input_type()));
		return writer;
	}

	nnforge::supervised_data_reader_smart_ptr
	NnjmToolset::get_initial_data_reader_for_training() const
	{
		nnforge_shared_ptr<std::istream> trainingDataStream(
				new boost::filesystem::ifstream(
						get_working_data_folder() /
						training_randomized_data_filename,
						std::ios_base::in | std::ios_base::binary));
		nnforge::supervised_data_reader_smart_ptr currentReader(
				new nnforge::supervised_sparse_data_stream_reader(
						trainingDataStream));
		return currentReader;
	}

	nnforge::supervised_data_reader_smart_ptr
	NnjmToolset::get_initial_data_reader_for_validating() const
	{
		nnforge_shared_ptr<std::istream> validating_data_stream(
				new boost::filesystem::ifstream(
						get_working_data_folder() / validating_data_filename,
						std::ios_base::in | std::ios_base::binary));
		nnforge::supervised_data_reader_smart_ptr current_reader(
				new nnforge::supervised_sparse_data_stream_reader(
						validating_data_stream));
		return current_reader;
	}

	const std::size_t NnjmToolset::getAffiliatedSourceWordIndex(
			const std::vector<std::vector<std::size_t> >&
			target2SourceAlignment,
			const std::size_t targetIndex,
			const std::size_t sourceTokensSize) const
	{
		// special case: targetIndex corresponds to the end of sentence
		if (targetIndex == target2SourceAlignment.size())
		{
			return sourceTokensSize;
		}
		std::size_t numberSourceWordsAlignedToTarget =
				target2SourceAlignment[targetIndex].size();
		// first case: one single source word aligned
		if (numberSourceWordsAlignedToTarget == 1)
		{
			return target2SourceAlignment[targetIndex][0];
		}
		// second case: several source words aligned
		// use the middle word rounding down
		if (numberSourceWordsAlignedToTarget > 1)
		{
			return target2SourceAlignment
					[targetIndex]
					[(numberSourceWordsAlignedToTarget - 1) / 2];
		}
		// third case: no source word aligned
		// in that case use the affiliation of the closest aligned word, starting
		// from the right
		bool foundAlignedWord = false;
		int neighborTargetIndex = boost::lexical_cast<int>(targetIndex) + 1;
		int jump = 2;
		bool jumpLeft = true;
		bool rightEnd = false, leftEnd = false;
		while (true)
		{
			if (neighborTargetIndex
					>= boost::lexical_cast<int>(target2SourceAlignment.size()))
			{
				rightEnd = true;
			}
			if (neighborTargetIndex <= 0)
			{
				leftEnd = true;
			}
			if (rightEnd && leftEnd)
			{
				break;
			}
			if (
					neighborTargetIndex >= 0 &&
					neighborTargetIndex < boost::lexical_cast<int>(
							target2SourceAlignment.size()) &&
					target2SourceAlignment[neighborTargetIndex].size() > 0)
			{
				foundAlignedWord = true;
				return getAffiliatedSourceWordIndex(
						target2SourceAlignment,
						neighborTargetIndex,
						sourceTokensSize);
			}
			if (jumpLeft)
			{
				neighborTargetIndex -= jump;
			}
			else
			{
				neighborTargetIndex += jump;
			}
			jumpLeft = !jumpLeft;
			++jump;
		}
		if (!foundAlignedWord)
		{
			BOOST_LOG_TRIVIAL(fatal)<< "No target word aligned";
		}
		return 0;
	}

	void NnjmToolset::getTarget2SourceAlignment(
			const std::string& alignmentLine,
			std::vector<std::vector<std::size_t> >*
			target2SourceAlignment) const
	{
		// target2SourceAlignment supposed to have correct size
		std::fill(
				target2SourceAlignment->begin(),
				target2SourceAlignment->end(),
				std::vector<std::size_t>());
		std::vector<std::string> links;
		boost::split(links, alignmentLine, boost::is_any_of(" "));
		BOOST_FOREACH(const std::string& link, links)
		{
			std::vector<std::string> linkPair;
			boost::split(linkPair, link, boost::is_any_of("-"));
			std::vector<int> linkInt(linkPair.size());
			std::transform(
					linkPair.begin(), linkPair.end(), linkInt.begin(),
					boost::lexical_cast<int, std::string>);
			if (linkInt.size() != 2)
			{
				BOOST_LOG_TRIVIAL(fatal)<< "Wrong format for link: " << link;
			}
			(*target2SourceAlignment)[linkInt[1]].push_back(linkInt[0]);
		}
	}

	void NnjmToolset::prepareTrainingInstance(
			const std::size_t targetTokenIndex,
			const std::vector<std::string>& sourceTokens,
			const std::vector<std::string>& targetTokens,
			const std::vector<std::vector<std::size_t> >&
			target2SourceAlignment,
			std::vector<WordId>* trainingNgram) const
	{
		// trainingNgram is assumed to have the correct size
		const int sourceWindowHalfSize = sourceWindowSize_ / 2;
		for (
				int targetNgramIndex = 0;
				targetNgramIndex < targetNgramSize_ - 1;
				++targetNgramIndex)
		{
			int targetHistoryIndex =
					boost::lexical_cast<int>(targetTokenIndex) -
					targetNgramIndex - 1;
			int inputTargetId;
			if (targetHistoryIndex >= 0)
			{
				inputTargetId =
						vocab_->getInputTargetId(
								targetTokens[targetHistoryIndex]);
			}
			else
			{
				inputTargetId = vocab_->getTargetStartSentenceId();
			}
			(*trainingNgram)[targetNgramIndex] = inputTargetId;
		}
		std::size_t affiliatedSourceWordIndex =
				getAffiliatedSourceWordIndex(
						target2SourceAlignment,
						targetTokenIndex,
						sourceTokens.size());
		for (
				int sourceWindowIndex = 0;
				sourceWindowIndex < sourceWindowSize_;
				++sourceWindowIndex)
		{
			int sourceTokenIndex =
					boost::lexical_cast<int>(affiliatedSourceWordIndex) -
					sourceWindowHalfSize +
					sourceWindowIndex;
			int inputSourceId;
			if (sourceTokenIndex < 0)
			{
				inputSourceId = vocab_->getSourceStartSentenceId();
			}
			else if (sourceTokenIndex >=
					boost::lexical_cast<int>(sourceTokens.size()))
			{
				inputSourceId = vocab_->getSourceEndSentenceId();
			}
			else
			{
				inputSourceId =
						vocab_->getInputSourceId(
								sourceTokens[sourceTokenIndex]);
			}
			(*trainingNgram)[targetNgramSize_ - 1 + sourceWindowIndex] =
					inputSourceId;
		}
		int outputTargetId;
		if (targetTokenIndex < targetTokens.size())
		{
			outputTargetId = vocab_->getOutputTargetId(
					targetTokens[targetTokenIndex]);
		}
		else
		{
			outputTargetId = vocab_->getTargetEndSentenceId();
		}
		(*trainingNgram)[targetNgramSize_ - 1 + sourceWindowSize_] =
				outputTargetId;
	}

	void NnjmToolset::convertToInputSparseData(
			const std::vector<WordId>& trainingNgram,
			std::vector<float>* inputSparseData) const
	{
		inputSparseData->assign(trainingNgram.begin(), trainingNgram.end() - 1);
	}

	void NnjmToolset::convertToOutputSparseData(
			const std::vector<WordId>& trainingNgram,
			std::vector<float>* outputSparseData) const
	{
		outputSparseData->assign(trainingNgram.end() - 1, trainingNgram.end());
	}

} //namespace nnjm
