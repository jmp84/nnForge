/*
 * nnjm_toolset_test.cpp
 *
 *  Created on: 26 May 2014
 *      Author: jmp84
 */

#define private public

#ifdef NNFORGE_CUDA_BACKEND_ENABLED
#include <nnforge/cuda/cuda.h>
#else
#include <nnforge/plain/plain.h>
#endif
#include <examples/nnjm/nnjm_toolset.h>

#include <boost/algorithm/string.hpp>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

struct NnjmToolsetFixture {
  NnjmToolsetFixture() {
#ifdef NNFORGE_CUDA_BACKEND_ENABLED
    nnjmToolset_.reset(new nnjm::NnjmToolset(
        nnforge::factory_generator_smart_ptr(
            new nnforge::cuda::factory_generator_cuda())));
#else
    nnjmToolset_.reset(new nnjm::NnjmToolset(
        nnforge::factory_generator_smart_ptr(
            new nnforge::plain::factory_generator_plain())));
#endif
    std::string sourceText = "45187 82073 15 22 28500 18 2575 31846 3 102 "
        "25017 133794 19 21379 5 566 957608 3532 5 26635 155153 725236 4\n"
        "63 134058 45187 82073 6702 193461 24 18185 134424 47438 27496 4";
    std::string targetText = "5023 8107 12 11 1547 14 205 55755 25 12 1226 22 "
        "11 36053 26 158559 16746 53 6119 9 3 16497 14412 115 10105 113 6 3 "
        "2904 514343 16497 5\n"
        "5023 8107 6098 3 514343 128 3 5880 4 47688 2904 9017 209 5";
    std::istringstream sourceTextStream(sourceText);
    std::istringstream targetTextStream(targetText);
    nnjmToolset_->vocab_.reset(
        new nnjm::Vocab(sourceTextStream, targetTextStream, 4, 9));
    nnjmToolset_->outputVocabSize_ = 9;
    nnjmToolset_->inputVocabSize_ = 4;
    nnjmToolset_->targetNgramSize_ = 4;
    nnjmToolset_->sourceWindowSize_ = 7;
  }

  nnforge_shared_ptr<nnjm::NnjmToolset> nnjmToolset_;
};

BOOST_FIXTURE_TEST_CASE(getAffiliatedSourceWordIndexCase1, NnjmToolsetFixture) {
  std::string alignment = "0-0 1-1";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(2);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  BOOST_CHECK_EQUAL(
      0,
      nnjmToolset_->getAffiliatedSourceWordIndex(target2SourceAlignment, 0, 2));
}

BOOST_FIXTURE_TEST_CASE(getAffiliatedSourceWordIndexCase2, NnjmToolsetFixture) {
  std::string alignment = "0-1 1-1 2-1";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(2);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  BOOST_CHECK_EQUAL(
      1,
      nnjmToolset_->getAffiliatedSourceWordIndex(target2SourceAlignment, 1, 3));
}

BOOST_FIXTURE_TEST_CASE(getAffiliatedSourceWordIndexCase2RoundDown,
                        NnjmToolsetFixture) {
  std::string alignment = "0-1 1-1";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(2);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  BOOST_CHECK_EQUAL(
      0,
      nnjmToolset_->getAffiliatedSourceWordIndex(target2SourceAlignment, 1, 2));
}

BOOST_FIXTURE_TEST_CASE(getAffiliatedSourceWordIndexCase3Right,
                        NnjmToolsetFixture) {
  std::string alignment = "0-0 1-2";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(3);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  BOOST_CHECK_EQUAL(
      1,
      nnjmToolset_->getAffiliatedSourceWordIndex(target2SourceAlignment, 1, 2));
}

BOOST_FIXTURE_TEST_CASE(getAffiliatedSourceWordIndexCase3Left,
                        NnjmToolsetFixture) {
  std::string alignment = "0-0 1-3";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(4);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  BOOST_CHECK_EQUAL(
      0,
      nnjmToolset_->getAffiliatedSourceWordIndex(target2SourceAlignment, 1, 2));
}

BOOST_FIXTURE_TEST_CASE(getAffiliatedSourceWordIndexCase3RightJump,
                        NnjmToolsetFixture) {
  std::string alignment = "0-0 4-4";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(5);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  BOOST_CHECK_EQUAL(
      4,
      nnjmToolset_->getAffiliatedSourceWordIndex(target2SourceAlignment, 2, 5));
}

BOOST_FIXTURE_TEST_CASE(getAffiliatedSourceWordIndexEndSentence,
                        NnjmToolsetFixture) {
  std::string alignment = "0-0 1-1";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(2);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  BOOST_CHECK_EQUAL(
      2,
      nnjmToolset_->getAffiliatedSourceWordIndex(target2SourceAlignment, 2, 2));
}

BOOST_FIXTURE_TEST_CASE(getTarget2SourceAlignment, NnjmToolsetFixture) {
  std::string alignment = "0-0 0-1 2-1 2-2";
  std::vector<std::vector<std::size_t> > target2SourceAlignment(4);
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  std::vector<std::vector<std::size_t> > target2SourceAlignmentExpected(4);
  target2SourceAlignmentExpected[0].push_back(0);
  target2SourceAlignmentExpected[1].push_back(0);
  target2SourceAlignmentExpected[1].push_back(2);
  target2SourceAlignmentExpected[2].push_back(2);
  BOOST_CHECK(target2SourceAlignmentExpected == target2SourceAlignment);
}

BOOST_FIXTURE_TEST_CASE(prepareTrainingInstance, NnjmToolsetFixture) {
  std::string source = "45187 82073 15 22 28500 18 2575 31846 3 102 "
      "25017 133794 19 21379 5 566 957608 3532 5 26635 155153 725236 4";
  std::string target = "5023 8107 12 11 1547 14 205 55755 25 12 1226 22 "
      "11 36053 26 158559 16746 53 6119 9 3 16497 14412 115 10105 113 6 3 "
      "2904 514343 16497 5";
  std::string alignment = "0-0 1-1 2-2 3-2 3-3 4-4 5-5 6-6 7-7 9-8 10-9 11-10 "
      "15-11 10-13 16-13 16-14 11-15 16-15 11-16 12-17 13-18 14-19 15-21 16-21 "
      "16-22 16-23 19-24 20-28 16-29 17-29 21-29 21-30 22-31";
  std::vector<std::string> sourceTokens, targetTokens;
  boost::split(sourceTokens, source, boost::is_any_of(" "));
  boost::split(targetTokens, target, boost::is_any_of(" "));
  std::vector<std::vector<std::size_t> > target2SourceAlignment(
      targetTokens.size());
  nnjmToolset_->getTarget2SourceAlignment(alignment, &target2SourceAlignment);
  std::vector<nnjm::WordId> trainingNgram(
      nnjmToolset_->targetNgramSize_ + nnjmToolset_->sourceWindowSize_);
  nnjmToolset_->prepareTrainingInstance(
      0, sourceTokens, targetTokens, target2SourceAlignment, &trainingNgram);
  std::string expectedTrainingNgramString = "7 7 7 5 5 5 0 1 4 4 8";
  std::vector<std::string> expectedTrainingNgramVecString;
  boost::split(expectedTrainingNgramVecString,
               expectedTrainingNgramString,
               boost::is_any_of(" "));
  std::vector<nnjm::WordId> expectedTrainingNgram(
      expectedTrainingNgramVecString.size());
  std::transform(expectedTrainingNgramVecString.begin(),
                 expectedTrainingNgramVecString.end(),
                 expectedTrainingNgram.begin(),
                 boost::lexical_cast<int, std::string>);
  BOOST_CHECK(expectedTrainingNgram == trainingNgram);
}

BOOST_FIXTURE_TEST_CASE(convertToInputData, NnjmToolsetFixture) {
  std::string trainingNgramString = "7 7 7 5 5 5 0 1 4 4 8";
  std::vector<std::string> trainingNgramVecString;
  boost::split(trainingNgramVecString,
               trainingNgramString,
               boost::is_any_of(" "));
  std::vector<nnjm::WordId> trainingNgram(trainingNgramVecString.size());
  std::transform(trainingNgramVecString.begin(),
                 trainingNgramVecString.end(),
                 trainingNgram.begin(),
                 boost::lexical_cast<int, std::string>);
  std::vector<unsigned char> inputData(8 * 10, 0);
  nnjmToolset_->convertToInputData(trainingNgram, &inputData);
  std::vector<unsigned char> expectedInputData(8 * 10, 0);
  expectedInputData[7 * 10 + 0] = 1;
  expectedInputData[7 * 10 + 1] = 1;
  expectedInputData[7 * 10 + 2] = 1;
  expectedInputData[5 * 10 + 3] = 1;
  expectedInputData[5 * 10 + 4] = 1;
  expectedInputData[5 * 10 + 5] = 1;
  expectedInputData[0 * 10 + 6] = 1;
  expectedInputData[1 * 10 + 7] = 1;
  expectedInputData[4 * 10 + 8] = 1;
  expectedInputData[4 * 10 + 9] = 1;
  BOOST_CHECK(expectedInputData == inputData);
}

BOOST_FIXTURE_TEST_CASE(convertToOutputData, NnjmToolsetFixture) {
  std::string trainingNgramString = "7 7 7 5 5 5 0 1 4 4 8";
  std::vector<std::string> trainingNgramVecString;
  boost::split(trainingNgramVecString,
               trainingNgramString,
               boost::is_any_of(" "));
  std::vector<nnjm::WordId> trainingNgram(trainingNgramVecString.size());
  std::transform(trainingNgramVecString.begin(),
                 trainingNgramVecString.end(),
                 trainingNgram.begin(),
                 boost::lexical_cast<int, std::string>);
  std::vector<float> outputData(11, 0);
  nnjmToolset_->convertToOutputData(trainingNgram, &outputData);
  std::vector<float> expectedOutputData(11, 0);
  expectedOutputData[8] = 1;
  BOOST_CHECK(expectedOutputData == outputData);
}
