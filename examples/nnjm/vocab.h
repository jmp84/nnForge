/*
 * vocab.h
 *
 *  Created on: 21 May 2014
 *      Author: jmp84
 */

#ifndef VOCAB_H_
#define VOCAB_H_

#include <examples/nnjm/nnjm_types.h>

namespace nnjm
{

	typedef nnjm_unordered_map<std::string, int> HashVocabularyType;
	typedef std::vector<std::string> VectorVocabularyType;


	/**
	 * Compare function used to sort key/value pairs by value in reverse order
	 * @param p1
	 * @param p2
	 * @return
	 */
	bool compareValue(
			const std::pair<std::string, int>& p1,
			const std::pair<std::string, int>& p2);

	/**
	 * Class to manage mapping between words and ids.
	 * TODO add serialization.
	 */
	class Vocab
	{
	public:
		/**
		 * Loads vocabulary from files
		 * @param inputSourceVocabFile The input source vocabulary file name
		 * @param inputTargetVocabFile The input target vocabulary file name
		 * @param outputTargetVocabFile The input target vocabulary file name
		 * @param inputVocabSize The size of the input vocabulary
		 * @param outputVocabSize The size of the output vocabulary
		 */
		Vocab(
				std::istream& inputSourceVocabFile,
				std::istream& inputTargetVocabFile,
				std::istream& outputTargetVocabFile,
				const int inputVocabSize,
				const int outputVocabSize
		);

		/**
		 * Constructor.
		 * Reads source and target text and builds the input source, input
		 * target and output target word-id mappings. Note that the input sizes
		 * are for regular words, the constructor will also add special words.
		 * @param sourceTextFile The source text file name.
		 * @param targetTextFile The target text file name.
		 * @param inputVocabSize The input source vocabulary size for regular
		 * words. No distinction between source and target. This may be extended
		 * for more flexibility.
		 * @param outputVocabSize The output vocabulary size for regular words.
		 */
		Vocab(
				std::istream& sourceTextFile,
				std::istream& targetTextFile,
				const int inputVocabSize,
				const int outputVocabSize);

		/**
		 * Returns the id corresponding to the source word or the special oov
		 * id.
		 * @param sourceWord The source word in string format.
		 * @return The id for the source word if it is in the vocabulary or the
		 * oov id.
		 */
		const int getInputSourceId(const std::string& sourceWord) const;

		/**
		 * Returns the id corresponding to the target word or the special oov
		 * id. The difference with getOutputTargetId is that the input
		 * vocabulary is considered.
		 * @param targetWord The target word in string format.
		 * @return The id for the target word if it is in the vocabulary or the
		 * oov id.
		 */
		const int getInputTargetId(const std::string& targetWord) const;

		/**
		 * Returns the id corresponding to the target word or the special oov
		 * id. The difference with getInputTargetId is that the output
		 * vocabulary is considered.
		 * @param targetWord The target word in string format.
		 * @return The id for the target word if it is in the vocabulary or the
		 * oov id.
		 */
		const int getOutputTargetId(const std::string& targetWord) const;

		/**
		 * Computes the input vocabulary size, including special tokens.
		 * @return The input vocabulary size, including special tokens.
		 */
		const int getInputVocabSize() const;

		/**
		 * Computes the output vocabulary size, including special tokens.
		 * @return The output vocabulary size, including special tokens.
		 */
		const int getOutputVocabSize() const;

		/**
		 * Gets the input source id for the sentence start.
		 * @return The input source id for the sentence start.
		 */
		const int getSourceStartSentenceId() const;

		/**
		 * Gets the input source id for the sentence end.
		 * @return The input source id for the sentence end.
		 */
		const int getSourceEndSentenceId() const;

		/**
		 * Gets the input target id for the sentence start.
		 * @return The input target id for the sentence start.
		 */
		const int getTargetStartSentenceId() const;

		/**
		 * Gets the output target id for the sentence end.
		 * @return The output target id for the sentence end.
		 */
		const int getTargetEndSentenceId() const;

		/**
		 * Stores vocabulary in three files
		 * @param inputSourceVocabFile The input source vocabulary file
		 * @param inputTargetVocabFile The input target vocabulary file
		 * @param outputTargetVocabFile The output target vocabulary file
		 */
		void store(
				std::ostream& inputSourceVocabFile,
				std::ostream& inputTargetVocabFile,
				std::ostream& outputTargetVocabFile);

	private:
		/**
		 * Helper for the constructor
		 * Reads vocabulary from file.
		 * @param vocabFile The vocabulary file name
		 * @param vocabSize The vocabulary size
		 * @param hv A hash (string-to-id)
		 * @param vv A vector (id-to-string)
		 */
		void loadVocab(
				std::istream& vocabFile,
				int const vocabSize,
				HashVocabularyType *hv,
				VectorVocabularyType *vv);

		/**
		 * Helper for the constructor.
		 * Reads a text and generates a vocabulary.
		 * @param textFile The text file input stream.
		 * @param vocabSize The vocabulary size.
		 * @param word2Id The resulting word to id map.
		 * @param id2Word The resulting id to word map.
		 */
		void makeVocab(
				std::istream& textFile, const int vocabSize,
				nnjm_unordered_map<std::string, int>* word2Id,
				std::vector<std::string>* id2Word);

		/** word to id map for input source words */
		nnjm_unordered_map<std::string, int> inputSourceWord2Id_;
		/** id to word map for input source words */
		std::vector<std::string> inputSourceId2Word_;
		/** word to id map for input target words */
		nnjm_unordered_map<std::string, int> inputTargetWord2Id_;
		/** id to word map for input target words */
		std::vector<std::string> inputTargetId2Word_;
		/** word to id map for output target words */
		nnjm_unordered_map<std::string, int> outputTargetWord2Id_;
		/** id to word map for output target words */
		std::vector<std::string> outputTargetId2Word_;
		/** special id for input oovs (no distinction between source and
		 * target) */
		int inputOovId_;
		/** special id for output oovs (different than inputOovId_ because the
		 * input vocab size and the output vocab size may be different) */
		int outputOovId_;
		/** special id for source start of sentence (belongs to input vocab) */
		int sourceStartSentenceId_;
		/** special id for source end of sentence (belongs to input vocab) */
		int sourceEndSentenceId_;
		/** special id for target start of sentence (belongs to input vocab;
		 * note that it doesn't belong to output vocab because a target start of
		 * sentence is never predicted) */
		int targetStartSentenceId_;
		/** special id for target end of sentence (belongs to target vocab; note
		 * that it doesn't belong to the input vocab because target end of
		 * sentence is never used as context) */
		int targetEndSentenceId_;
		/** input vocab size, including special tokens (oov, <src>, </src>,
		 * <trg>) */
		int inputVocabSize_;
		/** output vocab size, including special tokens (oov, </trg>) */
		int outputVocabSize_;
		/** input target vocab offset to get the correct input target ids (in
		 * the nnjm paper, source words are in the range 0-15999 and target
		 * words are in the range 16000-31999) */
		int inputTargetVocabOffset_;
	};

} // namespace

#endif /* VOCAB_H_ */
