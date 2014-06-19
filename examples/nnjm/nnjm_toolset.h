/**
 * TODO header
 */

#pragma once

#include <nnforge/nnforge.h>
#include <nnforge/neural_network_toolset.h>
#include <examples/nnjm/nnjm_types.h>

#include <examples/nnjm/vocab.h>

namespace nnjm {

class Vocab;

/**
 * Set of tools that are an interface between our application (nnjm) and the
 * neural network library.
 */
class NnjmToolset: public nnforge::neural_network_toolset {
public:
  /**
   * Constructor.
   * @param factory
   */
  NnjmToolset(nnforge::factory_generator_smart_ptr factory);

  /**
   * Destructor.
   */
  virtual ~NnjmToolset();

  /**
   * Initializes the vocabulary. TODO redesign this (shouldn't be public,
   * can only be called after this->parse(argc, argv) has been called, etc.)
   */
  void initVocab();

protected:
  /**
   * Additional string options for this application (nnjm).
   * These options are read by the neural_network_toolset::parse method.
   * @return The string options.
   */
  virtual std::vector<nnforge::string_option> get_string_options();

  /**
   * Additional integer options for this application (nnjm).
   * These options are read by the neural_network_toolset::parse method.
   * @return The integer options.
   */
  virtual std::vector<nnforge::int_option> get_int_options();

  /**
   * Additional float options for this application (nnjm).
   * These options are read by the neural_network_toolset::parse method.
   * @return The float options.
   */
  virtual std::vector<nnforge::float_option> get_float_options();

  /**
   * Prepares the training data.
   * Reads word aligned parallel text and output training data in binary format
   * to be read by the neural network trainer.
   */
  virtual void prepare_training_data();

  /**
   * Prepares the validation data.
   */
  virtual void prepare_validating_data();

  /**
   * Configures the neural network architecture.
   * @return
   */
  virtual nnforge::network_schema_smart_ptr get_schema() const;

  /**
   * Gets network output type: unkown, classifier, roc or regression
   * Here, we use the classifier type.
   * @return The network output type.
   */
  virtual nnforge::network_output_type::output_type
      get_network_output_type() const;

  /**
   * Whether validation data was used for training (measure error function on
   * validation data and adapt gradient descent learning weight if error goes
   * up)
   * @return True if training used validation data.
   */
  virtual bool is_training_with_validation() const;

  /**
   * Gets the error function, such as mse, cross-entropy, etc.
   * The default in parent class is mse. The default for this application is
   * cross-entropy.
   * @return The error function.
   */
  virtual nnforge::const_error_function_smart_ptr get_error_function() const;

  /**
   * Gets the training data reader. Used for randomizing training data. Here,
   * the supervised_sparse_data_stream_reader is used.
   * @param path The path to the training data.
   * @return The training data reader.
   */
  virtual nnforge::supervised_data_reader_smart_ptr
      get_original_training_data_reader(
          const boost::filesystem::path& path) const;

  /**
   * Gets the training data writer. Used for randomizing training data. Here,
   * the supervised_sparse_data_stream_writer is used.
   * @param reader The training data reader.
   * @param path The path to the randomized training data.
   * @return
   */
  virtual nnforge::data_writer_smart_ptr get_randomized_training_data_writer(
        nnforge::supervised_data_reader& reader,
        const boost::filesystem::path& path) const;

  /**
   * Gets the reader to read training data. The default in parent class is to
   * use a supervised_data_stream_reader. The default for this application is
   * to use a supervised_sparse_data_stream_reader. The difference with
   * get_original_training_data_reader is that the reader reads randomized data
   * whereas in get_original_training_data_reader, the reader reads the
   * original training data.
   * @return The training data reader.
   */
  virtual nnforge::supervised_data_reader_smart_ptr
      get_initial_data_reader_for_training() const;

  /**
   * Gets the validating data reader. The reader reads the validating data
   * specified by options or default. For this application, the reader
   * is a supervised_sparse_data_stream_reader .
   * @return The validating data reader.
   */
  virtual nnforge::supervised_data_reader_smart_ptr
      get_initial_data_reader_for_validating() const;

private:
  /**
   * Implements the affiliated word heuristic for nnjm.
   * @param target2sourceAlignment Target to source alignment.
   * @param targetIndex Target word index for which we want to find the
   * affiliated source word index.
   * @oaram sourceTokensSize The number of source words.
   * @return The affiliated source word index.
   */
  const std::size_t getAffiliatedSourceWordIndex(
      const std::vector<std::vector<std::size_t> >& target2sourceAlignment,
      const std::size_t targetIndex, const std::size_t sourceTokensSize) const;

  /**
   * Computes target to source alignment from an alignment in Berkeley format.
   * Berkeley format is "0-0 1-1, etc.".
   * @param alignmentLine The alignment in Berkeley format.
   * @param target2SourceAlignment The target to source alignment.
   */
  void getTarget2SourceAlignment(
      const std::string& alignmentLine,
      std::vector<std::vector<std::size_t> >* target2SourceAlignment) const;

  /**
   * Prepares a training instance.
   * A training instance is a context, i.e. the features, and a target word,
   * i.e. the label. The context consists of a history of target words as in a
   * usual n-gram language model and a sequence of source words.
   * @param targetTokenIndex The index of the target token.
   * @param sourceTokens The sequence of source tokens.
   * @param targetTokens The sequence of target tokens.
   * @param target2SourceAlignment The target to source alignment.
   * @param trainingNgram The resulting training n-gram that consists of the
   * context (source words and history target words) and the label (target word)
   * (in the nnjm paper, this is a vector of size 15)
   */
  void prepareTrainingInstance(
      const std::size_t targetTokenIndex,
      const std::vector<std::string>& sourceTokens,
      const std::vector<std::string>& targetTokens,
      const std::vector<std::vector<std::size_t> >& target2SourceAlignment,
      std::vector<WordId>* trainingNgram) const;

  /**
   * Converts a training n-gram to input data in sparse format.
   * The input data is later fed to the training writer.
   * @param trainingNgram The training n-gram that consists of the
   * context (source words and history target words) and the label (target word)
   * (in the nnjm paper, this is a vector of size 15)
   * @param inputSparseData The input data to be fed to the training writer (in
   * the nnjm paper, this is a vector of size 14, if it was not sparse, it would
   * be a vector of size 14 * 16000).
   */
  void convertToInputSparseData(const std::vector<WordId>& trainingNgram,
                                std::vector<float>* inputSparseData) const;

  /**
   * Converts a training n-gram to input data in sparse format.
   * The input data is later fed to the training writer.
   * @param trainingNgram The training n-gram that consists of the
   * context (source words and history target words) and the label (target word)
   * (in the nnjm paper, this is a vector of size 15)
   * @param outputData The output data to be fed to the training writer (in the
   * nnjm paper, this is a vector of size 1, if it was not sparse, it would be
   * a vector of size 32000).
   */
  void convertToOutputSparseData(const std::vector<WordId>& trainingNgram,
                                 std::vector<float>* outputSparseData) const;

  /** source text file name */
  std::string sourceTextFileName_;
  /** source text file name for testing */
  std::string sourceTextTestFileName_;
  /** source text file name for validating */
  std::string sourceTextValidateFileName_;
  /** target text file name */
  std::string targetTextFileName_;
  /** target text file name for testing */
  std::string targetTextTestFileName_;
  /** target text file name for validating */
  std::string targetTextValidateFileName_;
  /** alignment file name */
  std::string alignmentFileName_;
  /** alignment file name for testing */
  std::string alignmentTestFileName_;
  /** alignment file name for validating */
  std::string alignmentValidateFileName_;
  /** output training file name */
  std::string outputTrainingFileName_;
  /** output validating file name */
  std::string validatingDataFileName_;
  /** output testing file name */
  std::string outputTestingFileName_;
  /** output testing label file name; contains the target words */
  std::string outputTestingLabelFileName_;
  /** file where to write the output of testing */
  std::string outputPredictionsFileName_;
  /** error function (mse, crosse-entropy, etc.) */
  std::string errorFunction_;
  /** output vocabulary size (32000 in the nnjm paper) */
  int outputVocabSize_;
  /** input vocabulary size (16000 in the nnjm paper).
   * No distinction is made for source and target. This could be extended
   * for more flexibility. */
  int inputVocabSize_;
  /** target n-gram size (4 in the nnjm paper) */
  int targetNgramSize_;
  /** source window size (11 in the nnjm paper) */
  int sourceWindowSize_;
  /** vocabulary class to manage mapping between words and ids */
  nnforge_shared_ptr<Vocab> vocab_;
  /** continuous space dimension where to project input words
   * (192 in the nnjm paper) */
  int continuousSpaceDimension_;
  /** hidden layer size (512 in the nnjm paper) */
  int hiddenLayerSize_;
  /** portion of training data reserved for validation in training */
  float reservedForValidation_;
};

} // namespace nnjm
