/*
 * BSPIndexSetRegionTensor.cpp
 *
 *  Created on: 2014-8-20
 *      Author: junfeng
 */

#include "BSPIndexSetRegionTensor.hpp"
#include <cassert>
#include <iostream>

using namespace BSP;

IndexSetRegionTensor::IndexSetRegionTensor(const unsigned numberOfDimensions,
        LocalArray **lowerComponentAlongDim, LocalArray **upperComponentAlongDim,
        LocalArray **stepAlongDim) :
IndexSet(numberOfDimensions,
computeNumberOfIndices(numberOfDimensions,
lowerComponentAlongDim, upperComponentAlongDim, stepAlongDim)) {
    this->initConstantIterators();
    if (_begin == NULL || _end == NULL || _curr == NULL)
        throw ENotEnoughMemory();
}

IndexSetRegionTensor::IndexSetRegionTensor(unsigned int nDims, uint64_t *start, uint64_t *stop, int32_t *step):
        IndexSet(nDims, computeNumberOfIndices(nDims, start, stop, step)) {
    this->initConstantIterators();
    if (_begin == NULL || _end == NULL || _curr == NULL)
        throw ENotEnoughMemory();
}

IndexSetRegionTensor::~IndexSetRegionTensor() {
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        delete[] _lowerComponentAlongDim[iDim];
        delete[] _upperComponentAlongDim[iDim];
        delete[] _stepAlongDim[iDim];
    }
}

uint64_t IndexSetRegionTensor::computeNumberOfIndices(unsigned int nDims, uint64_t *start, uint64_t *stop, int32_t *step) {
    uint64_t result = 1;
    _numberOfRegions = 1;
    for (unsigned iDim = 0; iDim < nDims; ++ iDim) {
        _numberOfComponentsAlongDim[iDim] = 1;
        _lowerComponentAlongDim[iDim] = new uint64_t[1];
        _upperComponentAlongDim[iDim] = new uint64_t[1];
        _stepAlongDim[iDim] = new int32_t[1];
        if (!_lowerComponentAlongDim[iDim] || !_upperComponentAlongDim[iDim] || !_stepAlongDim[iDim]) {
            throw ENotEnoughMemory();
        }

        if (step[iDim] == 0)
            throw EInvalidRegionDescriptor(iDim, 0, start[iDim],stop[iDim],step[iDim]);

        int64_t dimSize = ((int64_t)stop[iDim] - (int64_t)start[iDim] - 1) / step[iDim] + 1;
        if (dimSize <= 0) 
            throw EInvalidRegionDescriptor(iDim, 0, start[iDim],stop[iDim],step[iDim]);
        result *= dimSize;

        _stepAlongDim[iDim][0] = step[iDim];
        _lowerComponentAlongDim[iDim][0] = start[iDim];
        _upperComponentAlongDim[iDim][0] = start[iDim] + (dimSize - 1) * step[iDim];
    }
    return result;
}

uint64_t IndexSetRegionTensor::computeNumberOfIndices(
        const unsigned numberOfDimensions, LocalArray **lowerComponentAlongDim,
        LocalArray **upperComponentAlongDim, LocalArray **stepAlongDim) {
    _numberOfRegions = 1;
    uint64_t combination[7];
    for (unsigned iDim = 0; iDim < numberOfDimensions; iDim++) {
        // set and check number of components along this dim
        _numberOfComponentsAlongDim[iDim] =
                lowerComponentAlongDim[iDim]->getElementCount(
                LocalArray::ALL_DIMS);
        if (0 == _numberOfComponentsAlongDim[iDim]
                || upperComponentAlongDim[iDim]->getElementCount(
                LocalArray::ALL_DIMS)
                != _numberOfComponentsAlongDim[iDim])
            throw EInvalidArgument();
        if (stepAlongDim) {
            if (stepAlongDim[iDim]->getElementCount(LocalArray::ALL_DIMS) 
                    != _numberOfComponentsAlongDim[iDim])
                throw EInvalidArgument();
        }

        // allocate auxilary arrays
        _lowerComponentAlongDim[iDim] =
                new uint64_t[_numberOfComponentsAlongDim[iDim]];
        _upperComponentAlongDim[iDim] =
                new uint64_t[_numberOfComponentsAlongDim[iDim]];
        _stepAlongDim[iDim] = new int32_t[_numberOfComponentsAlongDim[iDim]];

        // copy step along dim
        if (stepAlongDim) {
            if (stepAlongDim[iDim]->getElementType() == ArrayShape::INT32)
                memcpy(_stepAlongDim[iDim], stepAlongDim[iDim]->getData(), 
                        _numberOfComponentsAlongDim[iDim] * sizeof(int32_t));
            else {
                LocalArray properTyped(ArrayShape::INT32, *stepAlongDim[iDim]);
                memcpy(_stepAlongDim[iDim], stepAlongDim[iDim]->getData(), 
                        _numberOfComponentsAlongDim[iDim] * sizeof(int32_t));
            }
        } else {
            for (uint64_t i = 0; i < _numberOfComponentsAlongDim[iDim]; ++ i) {
                _stepAlongDim[iDim][i] = 1;
            }
        }

        // copy lower components
        if (_lowerComponentAlongDim[iDim] == NULL
                || _upperComponentAlongDim[iDim] == NULL)
            throw ENotEnoughMemory();
        if (lowerComponentAlongDim[iDim]->getElementType() == ArrayShape::UINT64)
            memcpy(_lowerComponentAlongDim[iDim],
                    lowerComponentAlongDim[iDim]->getData(),
                    _numberOfComponentsAlongDim[iDim] * sizeof (uint64_t));
        else {
            LocalArray properTyped(ArrayShape::UINT64, *lowerComponentAlongDim[iDim]);
            memcpy(_lowerComponentAlongDim[iDim],
                    properTyped.getData(),
                    _numberOfComponentsAlongDim[iDim] * sizeof (uint64_t));
        }

        // calculate and set upper components
        if (upperComponentAlongDim[iDim]->getElementType() == ArrayShape::UINT64)
            memcpy(_upperComponentAlongDim[iDim],
                    upperComponentAlongDim[iDim]->getData(),
                    _numberOfComponentsAlongDim[iDim] * sizeof (uint64_t));
        else {
            LocalArray properTyped(ArrayShape::UINT64, *upperComponentAlongDim[iDim]);
            memcpy(_upperComponentAlongDim[iDim],
                    properTyped.getData(),
                    _numberOfComponentsAlongDim[iDim] * sizeof (uint64_t));
        }
        for (uint64_t i = 0; i < _numberOfComponentsAlongDim[iDim]; ++ i) {
            int64_t sliceSize = ((int64_t)_upperComponentAlongDim[iDim][i] 
                    - (int64_t)_lowerComponentAlongDim[iDim][i] - 1) / _stepAlongDim[iDim][i] + 1;
            if (sliceSize <= 0) 
                throw EInvalidRegionDescriptor(iDim, i, 
                        _lowerComponentAlongDim[iDim][i],
                        _upperComponentAlongDim[iDim][i],
                        _stepAlongDim[iDim][i]);
            _upperComponentAlongDim[iDim][i] = _lowerComponentAlongDim[iDim][i] + (sliceSize - 1) * _stepAlongDim[iDim][i];
        }

        combination[iDim] = 0;
        _numberOfRegions *= _numberOfComponentsAlongDim[iDim];
    }

    uint64_t result = 0;
    // iterate through the combinations
    while (combination[0] < _numberOfComponentsAlongDim[0]) {
        // to compute the region size of each combination
        uint64_t currentRegionSize = 1;
        for (unsigned iDim = 0; iDim < numberOfDimensions; iDim++) {
            currentRegionSize *=
                    ((int64_t) _upperComponentAlongDim[iDim][combination[iDim]]
                    - (int64_t) _lowerComponentAlongDim[iDim][combination[iDim]]) / _stepAlongDim[iDim][combination[iDim]]
                    + 1;
        }
        // and add the region size to the result
        result += currentRegionSize;

        // next combination
        for (int iDim = numberOfDimensions - 1; iDim >= 0; iDim--) {
            combination[iDim]++;
            if (iDim == 0
                    || combination[iDim] < _numberOfComponentsAlongDim[iDim])
                break;
            combination[iDim] = 0;
        }
    }

    return result;
}

void IndexSetRegionTensor::initConstantIterators() {
    uint64_t initialComponentBound[14];
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        initialComponentBound[iDim] = _numberOfComponentsAlongDim[iDim];
        initialComponentBound[_numberOfDimensions + iDim] =
                (_upperComponentAlongDim[iDim][0] - _lowerComponentAlongDim[iDim][0]) / _stepAlongDim[iDim][0] + 1;
    }
    _begin = new Iterator(this, 2 * _numberOfDimensions, _numberOfDimensions,
            initialComponentBound, true);
    _end = new Iterator(this, 2 * _numberOfDimensions, _numberOfDimensions,
            initialComponentBound, false);
    _curr = new Iterator(*_begin);
}

void IndexSetRegionTensor::updateComponentBoundsOfIterator(Iterator *iterator) {
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        uint64_t componentRank = iterator->getComponentRank(iDim);
        uint64_t newBound =
                (_upperComponentAlongDim[iDim][componentRank]
                - _lowerComponentAlongDim[iDim][componentRank])
                / _stepAlongDim[iDim][componentRank] + 1;
        iterator->setComponentBound(_numberOfDimensions + iDim, newBound);
    }
}

void IndexSetRegionTensor::getIndex(Iterator &iterator) {
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        iterator._index[iDim] =
                _lowerComponentAlongDim[iDim][iterator.getComponentRank(iDim)]
                + iterator.getComponentRank(_numberOfDimensions + iDim)
                * _stepAlongDim[iDim][iterator.getComponentRank(iDim)];
    }
}


