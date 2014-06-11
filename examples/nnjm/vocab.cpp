/*
 * vocab.cpp
 *
 *  Created on: 21 May 2014
 *      Author: jmp84
 */

#include <examples/nnjm/vocab.h>

#include <fstream>

#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>

namespace nnjm {

bool compareValue(const std::pair<std::string, int>& p1,
                  const std::pair<std::string, int>& p2) {
  if (p1.second > p2.second) {
    return true;
  }
  // the rest of the function is to make it deterministic
  if (p1.second < p2.second) {
    return false;
  }
  if (p1.first.compare(p2.first) < 0) {
    return true;
  }
  return false;
}

Vocab::Vocab(std::istream& sourceTextFile,
             std::istream& targetTextFile,
             const int inputVocabSize,
             const int outputVocabSize) :
                 // '2 *' is because there's source and target
                 inputOovId_(2 * inputVocabSize),
                 sourceStartSentenceId_(2 * inputVocabSize + 1),
                 sourceEndSentenceId_(2 * inputVocabSize + 2),
                 targetStartSentenceId_(2 * inputVocabSize + 3),
                 outputOovId_(outputVocabSize),
                 targetEndSentenceId_(outputVocabSize + 1),
                 inputTargetVocabOffset_(inputVocabSize),
                 // there are 4 special words:
                 // <oov>, <src>, </src> and <trg>
                 inputVocabSize_(2 * inputVocabSize + 4),
                 // there are 2 special words:
                 // <oov>, </trg>
                 outputVocabSize_(outputVocabSize + 2) {
  makeVocab(sourceTextFile, inputVocabSize,
            &inputSourceWord2Id_, &inputSourceId2Word_);
  makeVocab(targetTextFile, inputVocabSize,
            &inputTargetWord2Id_, &inputTargetId2Word_);
  // reset target input stream
  targetTextFile.clear();
  targetTextFile.seekg(0, std::ios_base::beg);
  makeVocab(targetTextFile, outputVocabSize,
            &outputTargetWord2Id_, &outputTargetId2Word_);
}

const int Vocab::getInputSourceId(const std::string& sourceWord) const {
  nnjm_unordered_map<std::string, int>::const_iterator lookup =
      inputSourceWord2Id_.find(sourceWord);
  if (lookup == inputSourceWord2Id_.end()) {
    return inputOovId_;
  }
  return lookup->second;
}

const int Vocab::getInputTargetId(const std::string& targetWord) const {
  nnjm_unordered_map<std::string, int>::const_iterator lookup =
      inputTargetWord2Id_.find(targetWord);
  if (lookup == inputTargetWord2Id_.end()) {
    return inputOovId_;
  }
  return lookup->second + inputTargetVocabOffset_;
}

const int Vocab::getOutputTargetId(const std::string& targetWord) const {
  nnjm_unordered_map<std::string, int>::const_iterator lookup =
      outputTargetWord2Id_.find(targetWord);
  if (lookup == outputTargetWord2Id_.end()) {
    return outputOovId_;
  }
  return lookup->second;
}

const int Vocab::getInputVocabSize() const {
  return inputVocabSize_;
}

const int Vocab::getOutputVocabSize() const {
  return outputVocabSize_;
}

const int Vocab::getSourceStartSentenceId() const {
  return sourceStartSentenceId_;
}

const int Vocab::getSourceEndSentenceId() const {
  return sourceEndSentenceId_;
}

const int Vocab::getTargetStartSentenceId() const {
  return targetStartSentenceId_;
}

const int Vocab::getTargetEndSentenceId() const {
  return targetEndSentenceId_;
}

void Vocab::makeVocab(std::istream& textFile, const int vocabSize,
                      nnjm_unordered_map<std::string, int>* word2Id,
                      std::vector<std::string>* id2Word) {
  word2Id->clear();
  id2Word->clear();
  id2Word->resize(vocabSize);
  std::string line;
  boost::char_separator<char> space(" ");
  nnjm_unordered_map<std::string, int> frequency;
  while (std::getline(textFile, line)) {
    boost::tokenizer<boost::char_separator<char> > tokens(line, space);
    BOOST_FOREACH(const std::string& token, tokens) {
      frequency[token]++;
    }
  }
  std::vector<std::pair<std::string, int> > frequencyVec;
  frequencyVec.reserve(frequency.size());
  std::copy(frequency.begin(), frequency.end(),
            std::back_inserter(frequencyVec));
  std::sort(frequencyVec.begin(), frequencyVec.end(), compareValue);
  typedef std::pair<std::string, int> pairStringInt;
  int id = 0;
  BOOST_FOREACH(const pairStringInt& p, frequencyVec) {
    (*word2Id)[p.first] = id;
    (*id2Word)[id] = p.first;
    id++;
    if (id >= vocabSize) {
      break;
    }
  }
}

} // namespace nnjm
