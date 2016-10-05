/*
 * BSPGlobalArray.hpp
 *
 *  Created on: 2014-8-21
 *      Author: junfeng
 */

#ifndef BSPGLOBALARRAY_HPP_
#define BSPGLOBALARRAY_HPP_

#include "BSPArrayPartition.hpp"

namespace BSP {

class GlobalArray: public BSP::ArrayPartition {
private:
	ArrayRegistration *_registration;
public:
	GlobalArray(ArrayRegistration &registration);
	virtual ~GlobalArray();
	ArrayRegistration *getRegistration() {return _registration;}
        void getLocalRange(uint64_t *lower, uint64_t *upper);
        void getRange(uint64_t procID, uint64_t *lower, uint64_t *upper);
        char *blockScatter(const unsigned int dimDivide, const unsigned int blockWidth,
                unsigned int *nBytesInBlock, uint64_t *blockSizeAlongDim,
                unsigned int *localBlockCount);
        void blockGather(const unsigned int dimDivide, const unsigned int blockWidth,
                const unsigned int nBytesInBlock, const uint64_t *blockSizeAlongDim,
                const unsigned int localBlockCount, const char *localBlockData);
        void permute(const unsigned int dimPermute, const unsigned int *P);
};

} /* namespace BSP */
#endif /* BSPGLOBALARRAY_HPP_ */
