/*
 * vocab_test.cpp
 *
 *  Created on: 26 May 2014
 *      Author: jmp84
 */

#define private public

#include <examples/nnjm/vocab.h>
#include <nnforge/nn_types.h>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

/**
 * Simple test for the compareValue function.
 */
BOOST_AUTO_TEST_CASE(compareValue) {
  std::pair<std::string, int> p1("the", 1000);
  std::pair<std::string, int> p2("seldom", 2);
  BOOST_CHECK_EQUAL(true, nnjm::compareValue(p1, p2));
}

struct VocabFixture {
  VocabFixture() {
    std::string sourceText = "45187 82073 15 22 28500 18 2575 31846 3 102 "
        "25017 133794 19 21379 5 566 957608 3532 5 26635 155153 725236 4\n"
        "63 134058 45187 82073 6702 193461 24 18185 134424 47438 27496 4";
    std::string targetText = "5023 8107 12 11 1547 14 205 55755 25 12 1226 22 "
        "11 36053 26 158559 16746 53 6119 9 3 16497 14412 115 10105 113 6 3 "
        "2904 514343 16497 5\n"
        "5023 8107 6098 3 514343 128 3 5880 4 47688 2904 9017 209 5";
    std::istringstream sourceTextStream(sourceText);
    std::istringstream targetTextStream(targetText);
    vocab_.reset(new nnjm::Vocab(sourceTextStream, targetTextStream, 4, 9));
  }

  nnforge_shared_ptr<nnjm::Vocab> vocab_;
};

BOOST_FIXTURE_TEST_CASE(Vocab, VocabFixture) {

  nnjm_unordered_map<std::string, int> expectedInputSourceWord2Id;
  std::vector<std::string> expectedInputSourceId2Word(4);
  nnjm_unordered_map<std::string, int> expectedInputTargetWord2Id;
  std::vector<std::string> expectedInputTargetId2Word(4);
  nnjm_unordered_map<std::string, int> expectedOutputTargetWord2Id;
  std::vector<std::string> expectedOutputTargetId2Word(9);

  expectedInputSourceWord2Id["45187"] = 0;
  expectedInputSourceWord2Id["82073"] = 1;
  expectedInputSourceWord2Id["4"] = 2;
  expectedInputSourceWord2Id["5"] = 3;

  expectedInputSourceId2Word[0] = "45187";
  expectedInputSourceId2Word[1] = "82073";
  expectedInputSourceId2Word[2] = "4";
  expectedInputSourceId2Word[3] = "5";

  expectedInputTargetWord2Id["3"] = 0;
  expectedInputTargetWord2Id["16497"] = 1;
  expectedInputTargetWord2Id["11"] = 2;
  expectedInputTargetWord2Id["2904"] = 3;

  expectedInputTargetId2Word[0] = "3";
  expectedInputTargetId2Word[1] = "16497";
  expectedInputTargetId2Word[2] = "11";
  expectedInputTargetId2Word[3] = "2904";

  expectedOutputTargetWord2Id["3"] = 0;
  expectedOutputTargetWord2Id["16497"] = 1;
  expectedOutputTargetWord2Id["11"] = 2;
  expectedOutputTargetWord2Id["2904"] = 3;
  expectedOutputTargetWord2Id["514343"] = 4;
  expectedOutputTargetWord2Id["5"] = 5;
  expectedOutputTargetWord2Id["12"] = 6;
  expectedOutputTargetWord2Id["8107"] = 7;
  expectedOutputTargetWord2Id["5023"] = 8;

  expectedOutputTargetId2Word[0] = "3";
  expectedOutputTargetId2Word[1] = "16497";
  expectedOutputTargetId2Word[2] = "11";
  expectedOutputTargetId2Word[3] = "2904";
  expectedOutputTargetId2Word[4] = "514343";
  expectedOutputTargetId2Word[5] = "5";
  expectedOutputTargetId2Word[6] = "12";
  expectedOutputTargetId2Word[7] = "8107";
  expectedOutputTargetId2Word[8] = "5023";

  BOOST_CHECK(expectedInputSourceWord2Id == vocab_->inputSourceWord2Id_);
  BOOST_CHECK(expectedInputSourceId2Word == vocab_->inputSourceId2Word_);
  BOOST_CHECK(expectedInputTargetWord2Id == vocab_->inputTargetWord2Id_);
  BOOST_CHECK(expectedInputTargetId2Word == vocab_->inputTargetId2Word_);
  BOOST_CHECK(expectedOutputTargetWord2Id == vocab_->outputTargetWord2Id_);
  BOOST_CHECK(expectedOutputTargetId2Word == vocab_->outputTargetId2Word_);
  BOOST_CHECK_EQUAL(4, vocab_->inputOovId_);
  BOOST_CHECK_EQUAL(9, vocab_->outputOovId_);
  BOOST_CHECK_EQUAL(5, vocab_->sourceStartSentenceId_);
  BOOST_CHECK_EQUAL(6, vocab_->sourceEndSentenceId_);
  BOOST_CHECK_EQUAL(7, vocab_->targetStartSentenceId_);
  BOOST_CHECK_EQUAL(10, vocab_->targetEndSentenceId_);
  BOOST_CHECK_EQUAL(8, vocab_->inputVocabSize_);
  BOOST_CHECK_EQUAL(11, vocab_->outputVocabSize_);
}

BOOST_FIXTURE_TEST_CASE(getInputSourceId, VocabFixture) {
  BOOST_CHECK_EQUAL(1, vocab_->getInputSourceId("82073"));
  // oov
  BOOST_CHECK_EQUAL(4, vocab_->getInputSourceId("1234567"));
}

BOOST_FIXTURE_TEST_CASE(getInputTargetId, VocabFixture) {
  BOOST_CHECK_EQUAL(0, vocab_->getInputTargetId("3"));
  // oov
  BOOST_CHECK_EQUAL(4, vocab_->getInputTargetId("1234567"));
}

BOOST_FIXTURE_TEST_CASE(getOutputTargetId, VocabFixture) {
  BOOST_CHECK_EQUAL(8, vocab_->getOutputTargetId("5023"));
  // oov
  BOOST_CHECK_EQUAL(9, vocab_->getOutputTargetId("1234567"));
}

BOOST_FIXTURE_TEST_CASE(getSourceStartSentenceId, VocabFixture) {
  BOOST_CHECK_EQUAL(5, vocab_->getSourceStartSentenceId());
}

BOOST_FIXTURE_TEST_CASE(getSourceEndSentenceId, VocabFixture) {
  BOOST_CHECK_EQUAL(6, vocab_->getSourceEndSentenceId());
}

BOOST_FIXTURE_TEST_CASE(getTargetStartSentenceId, VocabFixture) {
  BOOST_CHECK_EQUAL(7, vocab_->getTargetStartSentenceId());
}

BOOST_FIXTURE_TEST_CASE(getTargetEndSentenceId, VocabFixture) {
  BOOST_CHECK_EQUAL(10, vocab_->getTargetEndSentenceId());
}
