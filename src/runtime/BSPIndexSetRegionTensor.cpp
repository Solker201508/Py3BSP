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

IndexSetRegionTensor::IndexSetRegionTensor(LocalArray& localArray) : IndexSet(localArray.getNumberOfDimensions(),
    computeNumberOfIndices(localArray)) {
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
        _lowerComponentAlongDim[iDim][0] = start[iDim];
        _upperComponentAlongDim[iDim][0] = stop[iDim] - 1;
        _stepAlongDim[iDim][0] = step[iDim];
        int64_t dimSize = ((int64_t)stop[iDim] - (int64_t)start[iDim] - 1) / step[iDim] + 1;
        assert(dimSize > 0);
        result *= dimSize;
    }
    return result;
}

uint64_t IndexSetRegionTensor::computeNumberOfIndices(LocalArray &localArray) {
    _numberOfRegions = 1;
    unsigned numberOfDimensions = localArray.getNumberOfDimensions();
    for (unsigned iDim = 0; iDim < numberOfDimensions; ++ iDim) {
        _numberOfComponentsAlongDim[iDim] = 1;
        _lowerComponentAlongDim[iDim] = new uint64_t[1];
        _upperComponentAlongDim[iDim] = new uint64_t[1];
        _stepAlongDim[iDim] = new int32_t[1];
        if (!_lowerComponentAlongDim[iDim] || !_upperComponentAlongDim[iDim] || !_stepAlongDim[iDim]) {
            throw ENotEnoughMemory();
        }
        _lowerComponentAlongDim[iDim][0] = 0;
        _upperComponentAlongDim[iDim][0] = localArray.getElementCount(iDim) - 1;
        _stepAlongDim[iDim][0] = 1;
    }
    return localArray.getElementCount(LocalArray::ALL_DIMS);
}

uint64_t IndexSetRegionTensor::computeNumberOfIndices(
        const unsigned numberOfDimensions, LocalArray **lowerComponentAlongDim,
        LocalArray **upperComponentAlongDim, LocalArray **stepAlongDim) {
    _numberOfRegions = 1;
    uint64_t combination[7];
    for (unsigned iDim = 0; iDim < numberOfDimensions; iDim++) {
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
        _lowerComponentAlongDim[iDim] =
                new uint64_t[_numberOfComponentsAlongDim[iDim]];
        _upperComponentAlongDim[iDim] =
                new uint64_t[_numberOfComponentsAlongDim[iDim]];
        _stepAlongDim[iDim] = new int32_t[_numberOfComponentsAlongDim[iDim]];
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
        combination[iDim] = 0;
        for (uint64_t iComponent = 0;
                iComponent < _numberOfComponentsAlongDim[iDim]; iComponent++) {
            if (_stepAlongDim[iDim][iComponent] == 0)
                throw EInvalidRegionDescriptor(iDim, iComponent,
                        _lowerComponentAlongDim[iDim][iComponent],
                        _upperComponentAlongDim[iDim][iComponent],
                        _stepAlongDim[iDim][iComponent]);
            if (_stepAlongDim[iDim][iComponent] > 0 && _upperComponentAlongDim[iDim][iComponent]
                    < _lowerComponentAlongDim[iDim][iComponent])
                throw EInvalidRegionDescriptor(iDim, iComponent,
                        _lowerComponentAlongDim[iDim][iComponent],
                        _upperComponentAlongDim[iDim][iComponent],
                        _stepAlongDim[iDim][iComponent]);
            if (_stepAlongDim[iDim][iComponent] < 0 && _upperComponentAlongDim[iDim][iComponent]
                    > _lowerComponentAlongDim[iDim][iComponent])
                throw EInvalidRegionDescriptor(iDim, iComponent,
                        _lowerComponentAlongDim[iDim][iComponent],
                        _upperComponentAlongDim[iDim][iComponent],
                        _stepAlongDim[iDim][iComponent]);
        }
        _numberOfRegions *= _numberOfComponentsAlongDim[iDim];
    }

    uint64_t result = 0;
    while (combination[0] < _numberOfComponentsAlongDim[0]) {
        uint64_t currentRegionSize = 1;
        for (unsigned iDim = 0; iDim < numberOfDimensions; iDim++) {
            currentRegionSize *=
                    (_upperComponentAlongDim[iDim][combination[iDim]]
                    - _lowerComponentAlongDim[iDim][combination[iDim]]) / _stepAlongDim[iDim][combination[iDim]]
                    + 1;
        }
        result += currentRegionSize;
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


