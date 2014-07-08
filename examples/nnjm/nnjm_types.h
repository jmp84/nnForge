/*
 * nnjm_types.h
 *
 *  Created on: 21 May 2014
 *      Author: jmp84
 */

#ifndef NNJM_TYPES_H_
#define NNJM_TYPES_H_

#include <unordered_map>

/**
 * Compiler dependent types, similar to what is done in nn_types.h
 */

#ifdef NNFORGE_CPP11COMPILER
#define nnjm_unordered_map std::unordered_map
#else
#define nnjm_unordered_map std::tr1::unordered_map
#endif

// TODO review int or unsigned int or size_t or etc.
namespace nnjm
{
	typedef int WordId;
}

#endif /* NNJM_TYPES_H_ */
