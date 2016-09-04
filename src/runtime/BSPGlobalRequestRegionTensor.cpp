/*
 * GlobalRequestRegionTensor.cpp
 *
 *  Created on: 2014-8-5
 *      Author: junfeng
 */

#include "BSPGlobalRequestRegionTensor.hpp"
#include <cassert>
#include <iostream>

using namespace BSP;

GlobalRequestRegionTensor::GlobalRequestRegionTensor(ArrayPartition &partition,
        IndexSetRegionTensor &indexSet, const std::string requestID) :
GlobalRequestRegion(partition) {
    setRequestID(requestID);
    _nRegions = 1;
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        _nComponentsAlongDim[iDim] = indexSet._numberOfComponentsAlongDim[iDim];
        _nRegions *= _nComponentsAlongDim[iDim];
    }
    allocateComponents();

    // allocate some auxiliary arrays
    uint64_t * nComponentsAtPositionAlongDim[7];
    uint64_t **lowerLocalOffsetAtPositionAlongDim[7];
    uint64_t **upperLocalOffsetAtPositionAlongDim[7];
    int32_t **localStepAlongDim[7];

    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        uint64_t nProcsAlongThisDim = partition.getProcCount(iDim);
        nComponentsAtPositionAlongDim[iDim] = new uint64_t[nProcsAlongThisDim];
        lowerLocalOffsetAtPositionAlongDim[iDim] =
                new uint64_t *[nProcsAlongThisDim];
        upperLocalOffsetAtPositionAlongDim[iDim] =
                new uint64_t *[nProcsAlongThisDim];
        localStepAlongDim[iDim] = new int32_t *[nProcsAlongThisDim];

        if (nComponentsAtPositionAlongDim[iDim] == NULL
                || lowerLocalOffsetAtPositionAlongDim[iDim] == NULL
                || upperLocalOffsetAtPositionAlongDim[iDim] == NULL
                || localStepAlongDim[iDim] == NULL)
            throw ENotEnoughMemory();
        for (uint64_t iProc = 0; iProc < nProcsAlongThisDim; iProc++) {
            nComponentsAtPositionAlongDim[iDim][iProc] = 0;
            lowerLocalOffsetAtPositionAlongDim[iDim][iProc] = NULL;
            upperLocalOffsetAtPositionAlongDim[iDim][iProc] = NULL;
            localStepAlongDim[iDim][iProc] = NULL;
        }
    }

    // iterate through the components to convert them into owner positions 
    // and offsets in owners
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        for (uint64_t iComponent = 0; iComponent < _nComponentsAlongDim[iDim];
                iComponent++) {
            if (indexSet._lowerComponentAlongDim[iDim][iComponent]
                    > indexSet._upperComponentAlongDim[iDim][iComponent])
                throw EInvalidArgument();
            _lowerOwnerPositionAlongDim[iDim][iComponent] =
                    partition.getPosition(iDim,
                    indexSet._lowerComponentAlongDim[iDim][iComponent],
                    getRequestID(), 0, NULL);
            _lowerOffsetInOwnerAlongDim[iDim][iComponent] =
                    indexSet._lowerComponentAlongDim[iDim][iComponent]
                    - getNode(iDim,
                    _lowerOwnerPositionAlongDim[iDim][iComponent]);
            _upperOwnerPositionAlongDim[iDim][iComponent] =
                    partition.getPosition(iDim,
                    indexSet._upperComponentAlongDim[iDim][iComponent],
                    getRequestID(), 0, NULL);
            _upperOffsetInOwnerAlongDim[iDim][iComponent] =
                    indexSet._upperComponentAlongDim[iDim][iComponent]
                    - getNode(iDim,
                    _upperOwnerPositionAlongDim[iDim][iComponent]);
            _stepAlongDim[iDim][iComponent] =
                    indexSet._stepAlongDim[iDim][iComponent];

            int32_t step = indexSet._stepAlongDim[iDim][iComponent];
            assert(step != 0);
            int64_t start = (int64_t) indexSet._lowerComponentAlongDim[iDim][iComponent];
            int64_t stop = (int64_t) indexSet._upperComponentAlongDim[iDim][iComponent];
            uint64_t iProc0 = _lowerOwnerPositionAlongDim[iDim][iComponent];
            uint64_t iProc1 = _upperOwnerPositionAlongDim[iDim][iComponent];
            if (step < 0) {
                int64_t temp = start;
                start = stop;
                stop = temp;
                step = -step;

                uint64_t temp2 = iProc0;
                iProc0 = iProc1;
                iProc1 = temp2;
            }

            for (uint64_t iProc = iProc0; iProc <= iProc1; iProc++) {
                int64_t nodeStart = (int64_t)getNode(iDim, iProc);
                int64_t nodeStop = (int64_t)getNode(iDim, iProc + 1);
                int64_t minStop = nodeStop < stop + 1 ? nodeStop : stop + 1;
                // if there exists a nonnegative integer k such that start + k * step >= nodeStart and < min(nodeStop, stop + 1), the proc owns elements in this region
                int64_t k = nodeStart <= start ? 0 : (nodeStart - start + step - 1) / step;

                if (start + k * step < minStop) {
                    nComponentsAtPositionAlongDim[iDim][iProc]++;
                }
            }
        }
    }

    // allocate more of auxiliary arrays
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        uint64_t nProcsAlongThisDim = partition.getProcCount(iDim);
        for (uint64_t iProc = 0; iProc < nProcsAlongThisDim; iProc++) {
            if (nComponentsAtPositionAlongDim[iDim][iProc] == 0)
                continue;
            lowerLocalOffsetAtPositionAlongDim[iDim][iProc] =
                    new uint64_t[nComponentsAtPositionAlongDim[iDim][iProc]];
            upperLocalOffsetAtPositionAlongDim[iDim][iProc] =
                    new uint64_t[nComponentsAtPositionAlongDim[iDim][iProc]];
            localStepAlongDim[iDim][iProc] = new int32_t[nComponentsAtPositionAlongDim[iDim][iProc]]; 
            if (lowerLocalOffsetAtPositionAlongDim[iDim][iProc] == NULL
                    || upperLocalOffsetAtPositionAlongDim[iDim][iProc] == NULL
                    || localStepAlongDim[iDim][iProc] == NULL)
                throw ENotEnoughMemory();
        }
    }

    // iterate through the components to fill in localOffsetAtPositionAlongDim
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        uint64_t nProcsAlongThisDim = partition.getProcCount(iDim);
        for (uint64_t iProc = 0; iProc < nProcsAlongThisDim; iProc++) {
            nComponentsAtPositionAlongDim[iDim][iProc] = 0;
        }

        for (uint64_t iComponent = 0; iComponent < _nComponentsAlongDim[iDim];
                iComponent++) {

            int32_t step = indexSet._stepAlongDim[iDim][iComponent];
            assert(step != 0);
            int64_t start = (int64_t) indexSet._lowerComponentAlongDim[iDim][iComponent];
            int64_t stop = (int64_t) indexSet._upperComponentAlongDim[iDim][iComponent];
            uint64_t iProc0 = _lowerOwnerPositionAlongDim[iDim][iComponent];
            uint64_t iProc1 = _upperOwnerPositionAlongDim[iDim][iComponent];
            bool negStep = false;
            if (step < 0) {
                negStep = true;
                int64_t temp = start;
                start = stop;
                stop = temp;
                step = -step;

                uint64_t temp2 = iProc0;
                iProc0 = iProc1;
                iProc1 = temp2;
            }
            for (uint64_t iProc = iProc0; iProc <= iProc1; iProc++) {
                int64_t nodeStart = (int64_t)getNode(iDim, iProc);
                int64_t nodeStop = (int64_t)getNode(iDim, iProc + 1);
                int64_t minStop = nodeStop < stop + 1 ? nodeStop : stop + 1;
                // if there exists a nonnegative integer k such that start + k * step >= nodeStart and < min(nodeStop, stop + 1), the proc owns elements in this region
                int64_t k = nodeStart <= start ? 0 : (nodeStart - start + step - 1) / step;
                if (start + k * step >= minStop)
                    continue;

                uint64_t iLocalComponent =
                        nComponentsAtPositionAlongDim[iDim][iProc]++;

                // lowerLocalOffset
                if (iProc == _lowerOwnerPositionAlongDim[iDim][iComponent]) {
                    lowerLocalOffsetAtPositionAlongDim[iDim][iProc][iLocalComponent] =
                            _lowerOffsetInOwnerAlongDim[iDim][iComponent];
                } else {
                    int64_t lower = start + k * step - nodeStart;
                    if (negStep) {
                        int64_t dK = (nodeStop - 1 - nodeStart - lower) / step;
                        lower += dK * step;
                    }
                    lowerLocalOffsetAtPositionAlongDim[iDim][iProc][iLocalComponent] = lower;
                }

                // upperLocalOffset
                if (iProc == _upperOwnerPositionAlongDim[iDim][iComponent]) {
                    upperLocalOffsetAtPositionAlongDim[iDim][iProc][iLocalComponent] =
                            _upperOffsetInOwnerAlongDim[iDim][iComponent];
                } else {
                    int upper = start + k * step - nodeStart;
                    if (!negStep) {
                        int64_t dK = (nodeStop - 1 - nodeStart - upper) / step;
                        upper += dK * step;
                    }
                    upperLocalOffsetAtPositionAlongDim[iDim][iProc][iLocalComponent] = upper;
                }

                // localStep
                localStepAlongDim[iDim][iProc][iLocalComponent] = negStep ? -step : step;
            }
        }
    }

    // iterate through the procs in grid to generate their index list and data list
    for (uint64_t iProc = 0; iProc < _nProcsInGrid; iProc++) {
        // decompose the iProc to position
        uint64_t position[7];
        partition.getPositionFromIProc(iProc, position);

        // check whether at this position holds no requests
        bool empty = false;
        for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
            if (nComponentsAtPositionAlongDim[iDim][position[iDim]] == 0) {
                empty = true;
                break;
            }
        }

        // skip empty positions
        if (empty)
            continue;

        // compute index length and data count
        uint64_t localElementCount = 1;
        _indexLength[iProc] = 1;
        for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
            uint64_t nLocalComponents =
                    nComponentsAtPositionAlongDim[iDim][position[iDim]];
            _indexLength[iProc] += 1 + 3 * nLocalComponents;
            uint64_t localElementCountAlongDim = 0;
            for (uint64_t iLocalComponent = 0;
                    iLocalComponent < nLocalComponents;
                    ++iLocalComponent) {
                localElementCountAlongDim +=
                        ((int64_t) upperLocalOffsetAtPositionAlongDim[iDim]
                        [position[iDim]][iLocalComponent]
                        - (int64_t) lowerLocalOffsetAtPositionAlongDim[iDim]
                        [position[iDim]][iLocalComponent]) 
                        / localStepAlongDim[iDim][position[iDim]][iLocalComponent]
                        + 1;
            }
            localElementCount *= localElementCountAlongDim;
        }
        _dataCount[iProc] = localElementCount;

        // allocate the index-list and the data-list
        allocateForProc(iProc);

        // fill in the index-list
        _indexList[iProc][0] = _dataCount[iProc] * _numberOfBytesPerElement;
        uint64_t *currentIndex = _indexList[iProc] + _numberOfDimensions + 1;
        for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
            _indexList[iProc][iDim + 1] =
                    nComponentsAtPositionAlongDim[iDim][position[iDim]];
            for (uint64_t iComponent = 0;
                    iComponent
                    < nComponentsAtPositionAlongDim[iDim][position[iDim]];
                    iComponent++) {
                currentIndex[(iComponent << 1) + iComponent] =
                        lowerLocalOffsetAtPositionAlongDim[iDim][position[iDim]][iComponent];
                currentIndex[(iComponent << 1) + iComponent + 1] =
                        (upperLocalOffsetAtPositionAlongDim[iDim][position[iDim]][iComponent]
                        - lowerLocalOffsetAtPositionAlongDim[iDim][position[iDim]][iComponent])
                        / localStepAlongDim[iDim][position[iDim]][iComponent]
                        + 1;
                currentIndex[(iComponent << 1) + iComponent + 2] =
                    localStepAlongDim[iDim][position[iDim]][iComponent];
            }
            currentIndex += nComponentsAtPositionAlongDim[iDim][position[iDim]] * 3;
        }
    }

    // delete temporarily auxiliary arrays
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        uint64_t nProcsAlongThisDim = partition.getProcCount(iDim);
        for (uint64_t iProc = 0; iProc < nProcsAlongThisDim; iProc++) {
            if (nComponentsAtPositionAlongDim[iDim][iProc] == 0)
                continue;
            delete[] lowerLocalOffsetAtPositionAlongDim[iDim][iProc];
            delete[] upperLocalOffsetAtPositionAlongDim[iDim][iProc];
            delete[] localStepAlongDim[iDim][iProc];
        }
        delete[] nComponentsAtPositionAlongDim[iDim];
        delete[] lowerLocalOffsetAtPositionAlongDim[iDim];
        delete[] upperLocalOffsetAtPositionAlongDim[iDim];
        delete[] localStepAlongDim[iDim];
    }

    if (_nData != indexSet.getNumberOfIndices())
        throw ECorruptedRuntime();
}

GlobalRequestRegionTensor::~GlobalRequestRegionTensor() {
}

void GlobalRequestRegionTensor::getData(const uint64_t numberOfBytesPerElement,
        const uint64_t nData, char *data) {
    if (nData < _nData)
        throw EClientArrayTooSmall(getRequestID(), nData, _nData);
    if (numberOfBytesPerElement
            != _numberOfBytesPerElement || nData < _nData || data == NULL)
        throw EInvalidArgument();

    // allocate auxiliary array
    uint64_t *dataIndexAtProc = new uint64_t[_nProcsInGrid];
    if (dataIndexAtProc == NULL)
        throw ENotEnoughMemory();
    for (uint64_t iProc = 0; iProc < _nProcsInGrid; iProc++) {
        dataIndexAtProc[iProc] = 0;
    }

    // iterate through the regions
    uint64_t iData = 0;
    uint64_t regionCombination[7];
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        regionCombination[iDim] = 0;
    }
    while (regionCombination[0] < _nComponentsAlongDim[0]) {
        uint64_t regionWidth[7];
        int32_t regionStep[7];
        for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
            regionWidth[iDim] = getRegionWidth(iDim, regionCombination[iDim]);
            regionStep[iDim] = _stepAlongDim[iDim][regionCombination[iDim]];
        }

        // iterate through the points within this region
        uint64_t combination[7], position[7], localWidth[7];
        int64_t localOffset[7];
        for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
            combination[iDim] = 0;
            position[iDim] =
                    _lowerOwnerPositionAlongDim[iDim][regionCombination[iDim]];
            localOffset[iDim] =
                    _lowerOffsetInOwnerAlongDim[iDim][regionCombination[iDim]];
            localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                    - getNode(iDim, position[iDim]);
        }
        while (combination[0] < regionWidth[0]) {
            // compute iProc
            uint64_t iProc = getIProcFromPosition(position);

            // copy data
            const char *src = _dataList[iProc]
                    + _numberOfBytesPerElement * dataIndexAtProc[iProc]++;
            char *dst = data + _numberOfBytesPerElement * iData++;
            for (uint64_t iByte = 0; iByte < _numberOfBytesPerElement; iByte++)
                dst[iByte] = src[iByte];

            // get next combination
            for (int iDim = _numberOfDimensions - 1; iDim >= 0; iDim--) {
                combination[iDim]++;
                if (combination[iDim] < regionWidth[iDim]) {
                    localOffset[iDim] += regionStep[iDim];
                    if (localOffset[iDim] >= localWidth[iDim]) {
                        while (localOffset[iDim] >= localWidth[iDim]) {
                            localOffset[iDim] -= localWidth[iDim];
                            ++ position[iDim];
                            localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                                    - getNode(iDim, position[iDim]);
                        }
                    } else if (localOffset[iDim] < 0) {
                        while (localOffset[iDim] < 0) {
                            -- position[iDim];
                            localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                                    - getNode(iDim, position[iDim]);
                            localOffset[iDim] += localWidth[iDim];
                        }
                    }
                }

                if (iDim == 0 || combination[iDim] < regionWidth[iDim])
                    break;

                combination[iDim] = 0;
                position[iDim] =
                        _lowerOwnerPositionAlongDim[iDim][regionCombination[iDim]];
                localOffset[iDim] =
                        _lowerOffsetInOwnerAlongDim[iDim][regionCombination[iDim]];
                localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                        - getNode(iDim, position[iDim]);
            }
        }
        // get next region combination
        for (int iDim = _numberOfDimensions - 1; iDim >= 0; iDim--) {
            regionCombination[iDim]++;
            if (iDim == 0
                    || regionCombination[iDim] < _nComponentsAlongDim[iDim])
                break;
            regionCombination[iDim] = 0;
        }
    }

    delete[] dataIndexAtProc;
}

void GlobalRequestRegionTensor::setData(const uint64_t numberOfBytesPerElement,
        const uint64_t nData, const char *data) {
    if (nData < _nData)
        throw EClientArrayTooSmall(getRequestID(), nData, _nData);
    if (numberOfBytesPerElement
            != _numberOfBytesPerElement || nData < _nData || data == NULL)
        throw EInvalidArgument();

    // allocate auxiliary array
    uint64_t *dataIndexAtProc = new uint64_t[_nProcsInGrid];
    if (dataIndexAtProc == NULL)
        throw ENotEnoughMemory();
    for (uint64_t iProc = 0; iProc < _nProcsInGrid; iProc++) {
        dataIndexAtProc[iProc] = 0;
    }
    // iterate through the regions
    uint64_t iData = 0;
    uint64_t regionCombination[7];
    for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
        regionCombination[iDim] = 0;
    }
    while (regionCombination[0] < _nComponentsAlongDim[0]) {
        uint64_t regionWidth[7];
        int32_t regionStep[7];
        for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
            regionWidth[iDim] = getRegionWidth(iDim, regionCombination[iDim]);
            regionStep[iDim] = _stepAlongDim[iDim][regionCombination[iDim]];
        }

        // iterate through the points within this region
        uint64_t combination[7], position[7], localWidth[7];
        int64_t localOffset[7];
        for (unsigned iDim = 0; iDim < _numberOfDimensions; iDim++) {
            combination[iDim] = 0;
            position[iDim] =
                    _lowerOwnerPositionAlongDim[iDim][regionCombination[iDim]];
            localOffset[iDim] =
                    _lowerOffsetInOwnerAlongDim[iDim][regionCombination[iDim]];
            localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                    - getNode(iDim, position[iDim]);
        }
        while (combination[0] < regionWidth[0]) {
            // compute iProc
            uint64_t iProc = getIProcFromPosition(position);

            // copy data
            char *dst = _dataList[iProc]
                    + _numberOfBytesPerElement * dataIndexAtProc[iProc]++;
            const char *src = data + _numberOfBytesPerElement * iData++;
            for (uint64_t iByte = 0; iByte < _numberOfBytesPerElement; iByte++)
                dst[iByte] = src[iByte];

            // get next combination
            for (int iDim = _numberOfDimensions - 1; iDim >= 0; iDim--) {
                combination[iDim]++;
                if (combination[iDim] < regionWidth[iDim]) {
                    localOffset[iDim] += regionStep[iDim];
                    if (localOffset[iDim] >= localWidth[iDim]) {
                        while (localOffset[iDim] >= localWidth[iDim]) {
                            localOffset[iDim] -= localWidth[iDim];
                            ++ position[iDim];
                            localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                                    - getNode(iDim, position[iDim]);
                        }
                    } else if (localOffset[iDim] < 0) {
                        while (localOffset[iDim] < 0) {
                            -- position[iDim];
                            localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                                    - getNode(iDim, position[iDim]);
                            localOffset[iDim] += localWidth[iDim];
                        }
                    }
                }

                if (iDim == 0 || combination[iDim] < regionWidth[iDim])
                    break;

                combination[iDim] = 0;
                position[iDim] =
                        _lowerOwnerPositionAlongDim[iDim][regionCombination[iDim]];
                localOffset[iDim] =
                        _lowerOffsetInOwnerAlongDim[iDim][regionCombination[iDim]];
                localWidth[iDim] = getNode(iDim, position[iDim] + 1)
                        - getNode(iDim, position[iDim]);
            }
        }
        // get next region combination
        for (int iDim = _numberOfDimensions - 1; iDim >= 0; iDim--) {
            regionCombination[iDim]++;
            if (iDim == 0
                    || regionCombination[iDim] < _nComponentsAlongDim[iDim])
                break;
            regionCombination[iDim] = 0;
        }
    }

    delete[] dataIndexAtProc;
}


