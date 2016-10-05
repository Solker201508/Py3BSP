/*
 * BSPGlobalArray.cpp
 *
 *  Created on: 2014-8-21
 *      Author: junfeng
 */

#include "BSPGlobalArray.hpp"
#include "BSPRuntime.hpp"
#include "BSPLocalArray.hpp"
#include <cassert>

using namespace BSP;

GlobalArray::GlobalArray(ArrayRegistration &registration):
	ArrayPartition(registration) {
	_registration = &registration;
}

GlobalArray::~GlobalArray() {
	// TODO Auto-generated destructor stub
}

void GlobalArray::getLocalRange(uint64_t *lower, uint64_t *upper) {
    uint64_t position[7];
    getPositionFromProcID(Runtime::getActiveRuntime()->getMyProcessID(), position);
    for (unsigned int iDim = 0; iDim < _numberOfDimensions; ++ iDim) {
        lower[iDim] = getNode(iDim, position[iDim]);
        upper[iDim] = getNode(iDim, position[iDim] + 1);
    }
}

void GlobalArray::getRange(uint64_t procID, uint64_t *lower, uint64_t *upper) {
    uint64_t position[7];
    getPositionFromProcID(procID, position);
    for (unsigned int iDim = 0; iDim < _numberOfDimensions; ++ iDim) {
        lower[iDim] = getNode(iDim, position[iDim]);
        upper[iDim] = getNode(iDim, position[iDim] + 1);
    }
}

char *GlobalArray::blockScatter(const unsigned int dimDivide, const unsigned int blockWidth,
        unsigned int *nBytesInBlock, uint64_t *blockSizeAlongDim,
        unsigned int *localBlockCount) {
    Runtime *runtime = Runtime::getActiveRuntime();
    uint64_t myProcID = runtime->getMyProcessID();
    uint64_t procStart = getStartProcID();
    uint64_t procCount = getProcCount(ArrayPartition::ALL_DIMS);
    if (myProcID < procStart || myProcID >= procStart + procCount || dimDivide >= _numberOfDimensions) {
        *localBlockCount = 0;
        return NULL;
    }

    // 计算块数
    unsigned int nDims = _numberOfDimensions;
    uint64_t dimSize[7];
    unsigned int nBlocks = 1;
    uint64_t blockSize = _numberOfBytesPerElement;
    for (unsigned int jDim = 0; jDim < nDims; ++ jDim) {
        dimSize[jDim] = getElementCount(jDim);
        if (dimDivide == jDim) {
            nBlocks = (dimSize[jDim] + blockWidth - 1) / blockWidth;
            blockSizeAlongDim[jDim] = blockWidth;
        } else {
            blockSizeAlongDim[jDim] = dimSize[jDim];
        }
        blockSize *= blockSizeAlongDim[jDim];
    }
    *nBytesInBlock = blockSize;
    uint64_t nLocalBlocks = nBlocks / procCount + (myProcID - procStart < nBlocks % procCount ? 1 : 0);
    *localBlockCount = nLocalBlocks;

    // 为本地的块申请内存
    char *localBlocks = NULL;
    if (*localBlockCount > 0) {
        localBlocks = new char[nLocalBlocks * blockSize];
        memset(localBlocks, 0, nLocalBlocks * blockSize);
    }

    // 取本地数据
    LocalArray *localArray = dynamic_cast<LocalArray *>(_registration->getArrayShape(myProcID));
    assert(NULL != localArray);
    char *localData = localArray->getData();

    // 取本地范围
    uint64_t lower[7], upper[7];
    getLocalRange(lower, upper);

    // 遍历partner
    uint64_t UBoundNProcs = 1;
    while (UBoundNProcs < procCount)
        UBoundNProcs <<= 1;
    for (uint64_t partner = 0; partner < UBoundNProcs; ++ partner) {
        uint64_t partnerID = procStart + ((myProcID - procStart) ^ partner); 
        if (partnerID >= procStart + procCount)
           continue; 

        uint64_t partnerLower[7], partnerUpper[7];
        getRange(partnerID, partnerLower, partnerUpper);

        uint64_t sendSize = 0, recvSize = 0;

        // 遍历partner所要求的块
        for (unsigned int iBlock = partnerID - procStart; iBlock < nBlocks; iBlock += procCount) {
            // 取当前块的范围
            uint64_t blockLower[7], blockUpper[7];
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
            }

            // 当前块的范围与本地范围求交
            uint64_t validLower[7], validUpper[7];
            uint64_t validSize = _numberOfBytesPerElement;
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                validLower[iDim] = lower[iDim] < blockLower[iDim] ? blockLower[iDim] : lower[iDim];
                validUpper[iDim] = upper[iDim] < blockUpper[iDim] ? upper[iDim] : blockUpper[iDim];
                if (validLower[iDim] >= validUpper[iDim]) {
                    validSize = 0;
                    break;
                }
                validSize *= validUpper[iDim] - validLower[iDim];
            }

            // 如果交集非空则把交集大小加入发送大小
            sendSize += validSize;
        }

        // 遍历本地所要求的块
        for (unsigned int iBlock = myProcID - procStart; iBlock < nBlocks; iBlock += procCount) {
            // 取当前块的范围
            uint64_t blockLower[7], blockUpper[7];
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
            }

            // 当前块的范围与partner范围求交
            uint64_t validLower[7], validUpper[7];
            uint64_t validSize = _numberOfBytesPerElement;
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                validLower[iDim] = partnerLower[iDim] < blockLower[iDim] ? blockLower[iDim] : partnerLower[iDim];
                validUpper[iDim] = partnerUpper[iDim] < blockUpper[iDim] ? partnerUpper[iDim] : blockUpper[iDim];
                if (validLower[iDim] >= validUpper[iDim]) {
                    validSize = 0;
                    break;
                }
                validSize *= validUpper[iDim] - validLower[iDim];
            }

            // 如果交集非空则把交集大小加入接收大小
            recvSize += validSize;
        }

        // 仅当发送大小和接收大小都非0时需要交换数据
        if (sendSize == 0 && recvSize == 0)
            continue;

        // 为发送和接收缓冲区申请内存
        char *sendBuffer = NULL; 
        if (sendSize > 0) {
            sendBuffer = new char[sendSize];

            char *blockData = sendBuffer;

            // 遍历partner所要求的块
            for (unsigned int iBlock = partnerID - procStart; iBlock < nBlocks; iBlock += procCount) {
                // 取当前块的范围
                uint64_t blockLower[7], blockUpper[7];
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                    blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
                }

                // 当前块的范围与本地范围求交
                uint64_t validLower[7], validUpper[7];
                uint64_t validSize = _numberOfBytesPerElement;
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    validLower[iDim] = lower[iDim] < blockLower[iDim] ? blockLower[iDim] : lower[iDim];
                    validUpper[iDim] = upper[iDim] < blockUpper[iDim] ? upper[iDim] : blockUpper[iDim];
                    if (validLower[iDim] >= validUpper[iDim]) {
                        validSize = 0;
                        break;
                    }
                    validSize *= validUpper[iDim] - validLower[iDim];
                }

                // 填充当前块
                if (validSize > 0) {
                    // 计算段数、段开始、段宽、段间隔
                    uint64_t nSegments = 1, segmentStart = 0, 
                             segmentWidth = _numberOfBytesPerElement, 
                             segmentStride = _numberOfBytesPerElement;
                    for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                        segmentStart *= upper[iDim] - lower[iDim];
                        segmentStart += validLower[iDim] - lower[iDim];
                        if (iDim >= dimDivide) {
                            segmentWidth *= validUpper[iDim] - validLower[iDim];
                            segmentStride *= upper[iDim] - lower[iDim];
                        } else {
                            nSegments *= validUpper[iDim] - validLower[iDim];
                        }
                    }
                    segmentStart *= _numberOfBytesPerElement;

                    // 遍历这些段并且拷贝数据
                    for (uint64_t iSegment = 0; iSegment < nSegments; ++ iSegment) {
                        memcpy(blockData + iSegment * segmentWidth,
                                localData + segmentStart + iSegment * segmentStride,
                                segmentWidth);
                    }

                    blockData += validSize;
                } // if (validSize > 0)
            } // for (iBlock)
            runtime->getNAL()->addDataBlockToSend(sendBuffer, sendSize);
        } // if (sendSize > 0)

        char *recvBuffer = NULL;
        if (recvSize > 0) {
            recvBuffer = new char[recvSize];
            runtime->getNAL()->addDataBlockToReceive(recvBuffer, recvSize);
        }

        runtime->getNAL()->exchangeDataBlocksWith(partnerID);
        if (sendBuffer)
            delete[] sendBuffer;

        if (recvSize > 0) {
            char *blockData = recvBuffer;
            char *currBlock = localBlocks;
            // 遍历本地所要求的块
            for (unsigned int iBlock = myProcID - procStart; iBlock < nBlocks; iBlock += procCount) {
                // 取当前块的范围
                uint64_t blockLower[7], blockUpper[7];
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                    blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
                }

                // 当前块的范围与partner范围求交
                uint64_t validLower[7], validUpper[7];
                uint64_t validSize = _numberOfBytesPerElement;
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    validLower[iDim] = partnerLower[iDim] < blockLower[iDim] ? blockLower[iDim] : partnerLower[iDim];
                    validUpper[iDim] = partnerUpper[iDim] < blockUpper[iDim] ? partnerUpper[iDim] : blockUpper[iDim];
                    if (validLower[iDim] >= validUpper[iDim]) {
                        validSize = 0;
                        break;
                    }
                    validSize *= validUpper[iDim] - validLower[iDim];
                }

                // 填充当前块
                if (validSize > 0) {
                    uint64_t segmentWidth = _numberOfBytesPerElement * (validUpper[nDims - 1] - validLower[nDims - 1]);
                    uint64_t segmentPos[7];
                    for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                        segmentPos[iDim] = validLower[iDim];
                    }
                    if (nDims <= 1) {
                        uint64_t segmentStart = (segmentPos[0] - blockLower[0]) * _numberOfBytesPerElement;
                        memcpy(currBlock + segmentStart, blockData, segmentWidth);
                        blockData += segmentWidth;
                    } else while (segmentPos[0] < validUpper[0]) {
                        // 计算段的位置
                        uint64_t segmentStart = 0;
                        for (unsigned int iDim = 0; iDim < nDims; ++ iDim) { 
                            segmentStart *= blockSizeAlongDim[iDim];
                            segmentStart += segmentPos[iDim] - blockLower[iDim];
                        }
                        segmentStart *= _numberOfBytesPerElement;
                        assert(segmentStart + segmentWidth <= blockSize);

                        // 拷贝段
                        memcpy(currBlock + segmentStart, blockData, segmentWidth);
                        blockData += segmentWidth;

                        // 下一个位置
                        for (int iDim = nDims - 2; iDim >= 0; -- iDim) {
                            ++ segmentPos[iDim];
                            if (iDim == 0
                                    || segmentPos[iDim] < validUpper[iDim])
                                break;
                            segmentPos[iDim] = validLower[iDim];
                        }
                    }
                } // if (validSize > 0)
                currBlock += blockSize;
            } // for (iBlock)
        }

        if (recvBuffer)
            delete[] recvBuffer;
    }

    return localBlocks;
}

void GlobalArray::blockGather(const unsigned int dimDivide, const unsigned int blockWidth,
        const unsigned int nBytesInBlock, const uint64_t *blockSizeAlongDim,
        const unsigned int localBlockCount, const char *localBlocks) {
    Runtime *runtime = Runtime::getActiveRuntime();
    uint64_t myProcID = runtime->getMyProcessID();
    uint64_t procStart = getStartProcID();
    uint64_t procCount = getProcCount(ArrayPartition::ALL_DIMS);
    if (myProcID < procStart || myProcID >= procStart + procCount || dimDivide >= _numberOfDimensions) {
        return;
    }

    // 计算块数
    unsigned int nDims = _numberOfDimensions;
    uint64_t dimSize[7];
    unsigned int nBlocks = 1;
    uint64_t blockSize = _numberOfBytesPerElement;
    for (unsigned int jDim = 0; jDim < nDims; ++ jDim) {
        dimSize[jDim] = getElementCount(jDim);
        if (dimDivide == jDim) {
            nBlocks = (dimSize[jDim] + blockWidth - 1) / blockWidth;
            assert(blockSizeAlongDim[jDim] == blockWidth);
        } else {
            assert(blockSizeAlongDim[jDim] == dimSize[jDim]);
        }
        blockSize *= blockSizeAlongDim[jDim];
    }
    uint64_t nLocalBlocks = nBlocks / procCount + (myProcID - procStart < nBlocks % procCount ? 1 : 0);
    assert(nBytesInBlock == blockSize);
    assert(localBlockCount == nLocalBlocks);

    // 取本地数据
    LocalArray *localArray = dynamic_cast<LocalArray *>(_registration->getArrayShape(myProcID));
    assert(NULL != localArray);
    char *localData = localArray->getData();

    // 取本地范围
    uint64_t lower[7], upper[7];
    getLocalRange(lower, upper);

    // 遍历partner
    uint64_t UBoundNProcs = 1;
    while (UBoundNProcs < procCount)
        UBoundNProcs <<= 1;
    for (uint64_t partner = 0; partner < UBoundNProcs; ++ partner) {
        uint64_t partnerID = procStart + ((myProcID - procStart) ^ partner); 
        if (partnerID >= procStart + procCount)
           continue; 

        uint64_t partnerLower[7], partnerUpper[7];
        getRange(partnerID, partnerLower, partnerUpper);

        uint64_t sendSize = 0, recvSize = 0;

        // 遍历partner所拥有的块
        for (unsigned int iBlock = partnerID - procStart; iBlock < nBlocks; iBlock += procCount) {
            // 取当前块的范围
            uint64_t blockLower[7], blockUpper[7];
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
            }

            // 当前块的范围与本地范围求交
            uint64_t validLower[7], validUpper[7];
            uint64_t validSize = _numberOfBytesPerElement;
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                validLower[iDim] = lower[iDim] < blockLower[iDim] ? blockLower[iDim] : lower[iDim];
                validUpper[iDim] = upper[iDim] < blockUpper[iDim] ? upper[iDim] : blockUpper[iDim];
                if (validLower[iDim] >= validUpper[iDim]) {
                    validSize = 0;
                    break;
                }
                validSize *= validUpper[iDim] - validLower[iDim];
            }

            // 如果交集非空则把交集大小加入接收大小
            recvSize += validSize;
        }

        // 遍历本地所拥有的块
        for (unsigned int iBlock = myProcID - procStart; iBlock < nBlocks; iBlock += procCount) {
            // 取当前块的范围
            uint64_t blockLower[7], blockUpper[7];
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
            }

            // 当前块的范围与partner范围求交
            uint64_t validLower[7], validUpper[7];
            uint64_t validSize = _numberOfBytesPerElement;
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                validLower[iDim] = partnerLower[iDim] < blockLower[iDim] ? blockLower[iDim] : partnerLower[iDim];
                validUpper[iDim] = partnerUpper[iDim] < blockUpper[iDim] ? partnerUpper[iDim] : blockUpper[iDim];
                if (validLower[iDim] >= validUpper[iDim]) {
                    validSize = 0;
                    break;
                }
                validSize *= validUpper[iDim] - validLower[iDim];
            }

            // 如果交集非空则把交集大小加入发送大小
            sendSize += validSize;
        }

        // 仅当发送大小和接收大小都非0时需要交换数据
        if (sendSize == 0 && recvSize == 0)
            continue;

        // 为发送和接收缓冲区申请内存
        char *sendBuffer = NULL; 
        if (sendSize > 0) {
            sendBuffer = new char[sendSize];

            char *blockData = sendBuffer;
            const char *currBlock = localBlocks;
            // 遍历本地所要求的块
            for (unsigned int iBlock = myProcID - procStart; iBlock < nBlocks; iBlock += procCount) {
                // 取当前块的范围
                uint64_t blockLower[7], blockUpper[7];
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                    blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
                }

                // 当前块的范围与partner范围求交
                uint64_t validLower[7], validUpper[7];
                uint64_t validSize = _numberOfBytesPerElement;
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    validLower[iDim] = partnerLower[iDim] < blockLower[iDim] ? blockLower[iDim] : partnerLower[iDim];
                    validUpper[iDim] = partnerUpper[iDim] < blockUpper[iDim] ? partnerUpper[iDim] : blockUpper[iDim];
                    if (validLower[iDim] >= validUpper[iDim]) {
                        validSize = 0;
                        break;
                    }
                    validSize *= validUpper[iDim] - validLower[iDim];
                }

                // 填充当前块
                if (validSize > 0) {
                    uint64_t segmentWidth = _numberOfBytesPerElement * (validUpper[nDims - 1] - validLower[nDims - 1]);
                    uint64_t segmentPos[7];
                    for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                        segmentPos[iDim] = validLower[iDim];
                    }
                    if (nDims <= 1) {
                        uint64_t segmentStart = (segmentPos[0] - blockLower[0]) * _numberOfBytesPerElement;
                        memcpy(blockData, currBlock + segmentStart, segmentWidth);
                        blockData += segmentWidth;
                    } else while (segmentPos[0] < validUpper[0]) {
                        // 计算段的位置
                        uint64_t segmentStart = 0;
                        for (unsigned int iDim = 0; iDim < nDims; ++ iDim) { 
                            segmentStart *= blockSizeAlongDim[iDim];
                            segmentStart += segmentPos[iDim] - blockLower[iDim];
                        }
                        segmentStart *= _numberOfBytesPerElement;
                        assert(segmentStart + segmentWidth <= blockSize);

                        // 拷贝段
                        memcpy(blockData, currBlock + segmentStart, segmentWidth);
                        blockData += segmentWidth;

                        // 下一个位置
                        for (int iDim = nDims - 2; iDim >= 0; -- iDim) {
                            ++ segmentPos[iDim];
                            if (iDim == 0
                                    || segmentPos[iDim] < validUpper[iDim])
                                break;
                            segmentPos[iDim] = validLower[iDim];
                        }
                    }
                } // if (validSize > 0)
                currBlock += nBytesInBlock;
            } // for (iBlock)

            runtime->getNAL()->addDataBlockToSend(sendBuffer, sendSize);
        } // if (sendSize > 0)

        char *recvBuffer = NULL;
        if (recvSize > 0) {
            recvBuffer = new char[recvSize];
            runtime->getNAL()->addDataBlockToReceive(recvBuffer, recvSize);
        }

        runtime->getNAL()->exchangeDataBlocksWith(partnerID);
        if (sendBuffer)
            delete[] sendBuffer;

        if (recvSize > 0) {
            char *blockData = recvBuffer;

            // 遍历partner所要求的块
            for (unsigned int iBlock = partnerID - procStart; iBlock < nBlocks; iBlock += procCount) {
                // 取当前块的范围
                uint64_t blockLower[7], blockUpper[7];
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    blockLower[iDim] = (iDim == dimDivide ? blockWidth * iBlock : 0);
                    blockUpper[iDim] = blockLower[iDim] + blockSizeAlongDim[iDim];
                }

                // 当前块的范围与本地范围求交
                uint64_t validLower[7], validUpper[7];
                uint64_t validSize = _numberOfBytesPerElement;
                for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                    validLower[iDim] = lower[iDim] < blockLower[iDim] ? blockLower[iDim] : lower[iDim];
                    validUpper[iDim] = upper[iDim] < blockUpper[iDim] ? upper[iDim] : blockUpper[iDim];
                    if (validLower[iDim] >= validUpper[iDim]) {
                        validSize = 0;
                        break;
                    }
                    validSize *= validUpper[iDim] - validLower[iDim];
                }

                // 填充当前块
                if (validSize > 0) {
                    // 计算段数、段开始、段宽、段间隔
                    uint64_t nSegments = 1, segmentStart = 0, 
                             segmentWidth = _numberOfBytesPerElement, 
                             segmentStride = _numberOfBytesPerElement;
                    for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                        segmentStart *= upper[iDim] - lower[iDim];
                        segmentStart += validLower[iDim] - lower[iDim];
                        if (iDim >= dimDivide) {
                            segmentWidth *= validUpper[iDim] - validLower[iDim];
                            segmentStride *= upper[iDim] - lower[iDim];
                        } else {
                            nSegments *= validUpper[iDim] - validLower[iDim];
                        }
                    }
                    segmentStart *= _numberOfBytesPerElement;

                    // 遍历这些段并且拷贝数据
                    for (uint64_t iSegment = 0; iSegment < nSegments; ++ iSegment) {
                        memcpy(localData + segmentStart + iSegment * segmentStride,
                                blockData + iSegment * segmentWidth,
                                segmentWidth);
                    }

                    blockData += validSize;
                } // if (validSize > 0)
            } // for (iBlock)
        }

        if (recvBuffer)
            delete[] recvBuffer;
    }
}

void GlobalArray::permute(const unsigned int dimPermute, const unsigned int *P) {
    uint64_t procStart = getStartProcID();
    uint64_t procCount = getProcCount(ArrayPartition::ALL_DIMS);
    uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
    if (myProcID < procStart || myProcID >= procStart + procCount)
        return;

    assert(dimPermute < _numberOfDimensions);

    unsigned int n = getElementCount(dimPermute); 
    unsigned *Q = new unsigned int[n];
    for (unsigned int i = 0; i < n; ++ i) {
        Q[i] = n;
    }
    for (unsigned int i = 0; i < n; ++ i) {
        assert(P[i] < n);
        Q[P[i]] = i;
    }
    for (unsigned int i = 0; i < n; ++ i) {
        assert(Q[i] < n);
        assert(P[Q[i]] == i);
    }
    delete[] Q;

    uint64_t lower[7], upper[7];
    getLocalRange(lower, upper);
    uint64_t nOuter = 1, nInner = _numberOfBytesPerElement;
    for (unsigned int iDim = 0; iDim < _numberOfDimensions; ++ iDim) {
        if (iDim < dimPermute)
            nOuter *= upper[iDim] - lower[iDim];
        else if (iDim > dimPermute)
            nInner *= upper[iDim] - lower[iDim];
    }
    uint64_t nPerOuter = nInner * (upper[dimPermute] - lower[dimPermute]);
    LocalArray *localArray = dynamic_cast<LocalArray *>(_registration->getArrayShape(myProcID));
    char *localData = localArray->getData();

    uint64_t nProcsAlongDim = getProcCount(dimPermute);
    if (nProcsAlongDim == 1) {
        char *temp = new char[nPerOuter];
        for (unsigned int iOuter = 0; iOuter < nOuter; ++ iOuter) {
            char *currData = localData + iOuter * nPerOuter;
            memcpy(temp, currData, nPerOuter);
            for (unsigned int i = 0; i < n; ++ i) {
                memcpy(currData + i * nInner, temp + P[i] * nInner, nInner);
            }
        }
        delete[] temp;
    } else {
        Net *nal = Runtime::getActiveRuntime()->getNAL();

        uint64_t pos[7];
        getPositionFromProcID(myProcID, pos);
        uint64_t myPosAlongDim = pos[dimPermute];

        unsigned int *iProcOfPos = new unsigned int[upper[dimPermute] - lower[dimPermute]];
        uint64_t *neighborProcID = new uint64_t[nProcsAlongDim];
        uint64_t *nRecvFromNeighbor = new uint64_t[nProcsAlongDim];
        uint64_t *nSendToNeighbor = new uint64_t[nProcsAlongDim];
        char *temp = new char[nOuter * nPerOuter];
        memcpy(temp, localData, nOuter * nPerOuter);
        for (uint64_t iProc = 0; iProc < nProcsAlongDim; ++ iProc) {
            pos[dimPermute] = iProc;
            neighborProcID[iProc] = getProcIDFromPosition(pos);
            nRecvFromNeighbor[iProc] = 0;
            nSendToNeighbor[iProc] = 0;
        }
        for (unsigned iNeighbor = 0; iNeighbor < nProcsAlongDim; ++ iNeighbor) {
            uint64_t i0 = getNode(dimPermute, iNeighbor);
            uint64_t i1 = getNode(dimPermute, iNeighbor + 1);
            for (unsigned int i = i0; i < i1; ++ i) {
                if (P[i] >= lower[dimPermute] && P[i] < upper[dimPermute]) {
                    ++ nSendToNeighbor[iNeighbor];
                }
            }
            for (unsigned int j = lower[dimPermute]; j < upper[dimPermute]; ++ j) {
                if (P[j] >= i0 && P[j] < i1) {
                    iProcOfPos[j - lower[dimPermute]] = iNeighbor;
                    ++ nRecvFromNeighbor[iNeighbor];
                }
            }
        }

        uint64_t UBoundNProcs = 1;
        while (UBoundNProcs < nProcsAlongDim)
            UBoundNProcs <<= 1;
        for (uint64_t partner = 0; partner < UBoundNProcs; ++ partner) {
            uint64_t partnerPosAlongDim = myPosAlongDim ^ partner;
            if (partnerPosAlongDim >= nProcsAlongDim)
                continue;
            if (nSendToNeighbor[partnerPosAlongDim] == 0 && nRecvFromNeighbor[partnerPosAlongDim] == 0)
                continue;
            char *sendBuffer = NULL;
            if (nSendToNeighbor[partnerPosAlongDim] > 0) {
                uint64_t sendSize = nOuter * nSendToNeighbor[partnerPosAlongDim] * nInner;
                sendBuffer = new char[sendSize];
                uint64_t i0 = getNode(dimPermute, partnerPosAlongDim);
                uint64_t i1 = getNode(dimPermute, partnerPosAlongDim + 1);
                char *dst = sendBuffer;
                for (unsigned int i = i0; i < i1; ++ i) {
                    if (P[i] >= lower[dimPermute] && P[i] < upper[dimPermute]) {
                        char *src = temp + (P[i] - lower[dimPermute]) * nInner;
                        for (unsigned int iOuter = 0; iOuter < nOuter; ++ iOuter) {
                            memcpy(dst, src, nInner);
                            src += nPerOuter;
                            dst += nInner;
                        }
                    }
                }

                nal->addDataBlockToSend(sendBuffer, sendSize);
            }
            char *recvBuffer = NULL;
            if (nRecvFromNeighbor[partnerPosAlongDim] > 0) {
                uint64_t recvSize = nOuter * nRecvFromNeighbor[partnerPosAlongDim] * nInner;
                recvBuffer = new char[recvSize];
                nal->addDataBlockToReceive(recvBuffer, recvSize);
            }

            nal->exchangeDataBlocksWith(neighborProcID[partnerPosAlongDim]);

            if (sendBuffer)
                delete[] sendBuffer;

            if (recvBuffer) {
                char *src = recvBuffer;
                for (unsigned int j = lower[dimPermute]; j < upper[dimPermute]; ++ j) {
                    if (iProcOfPos[j - lower[dimPermute]] != partnerPosAlongDim)
                        continue;
                    char *dst = localData + (j - lower[dimPermute]) * nInner;
                    for (unsigned int iOuter = 0; iOuter < nOuter; ++ iOuter) {
                        memcpy(dst, src, nInner); 
                        src += nInner;
                        dst += nPerOuter;
                    }
                }
                delete[] recvBuffer;
            }
        }

        delete[] iProcOfPos;
        delete[] neighborProcID;
        delete[] nRecvFromNeighbor;
        delete[] nSendToNeighbor;
        delete[] temp;
    }
}

