/*
 * BSPNet.cpp
 *
 *  Created on: 2014-7-9
 *      Author: Hongkai & Junfeng
 */
#include "BSPNet.hpp"
#include "BSPException.hpp"
#include "BSPRuntime.hpp"
#include <cassert>
#include <cstring>

using namespace BSP;
//! A private variable.
/*!
 A copy of MPI_COMM_WORLD. In convenient to coding.
 */
static MPI_Comm _communicator;

Net::Net(int *pArgc, char ***pArgv, uint64_t segmentThreshold,
        uint64_t maxNumberOfDataBlocks) {
    MPI_Init(pArgc, pArgv);
    initialize(MPI_COMM_WORLD, segmentThreshold, maxNumberOfDataBlocks);
}

void Net::initialize(MPI_Comm comm, uint64_t segmentThreshold,
        uint64_t maxNumberOfDataBlocks) {
    MPI_Comm_dup(comm, &_communicator);

    _segmentThreshold = segmentThreshold;
    _maxNumberOfDataBlocks = maxNumberOfDataBlocks;

    _dataBlockToSend = new void*[maxNumberOfDataBlocks];
    _lengthOfDataBlockToSend = new uint64_t[maxNumberOfDataBlocks];
    _numberOfDataBlocksToSend = 0;

    _dataBlockToReceive = new void*[maxNumberOfDataBlocks];
    _lengthOfDataBlockToReceive = new uint64_t[maxNumberOfDataBlocks];
    _numberOfDataBlocksToReceive = 0;
    _bufSumDouble = new double[1 << 20];
}

Net::~Net() {
    delete[] _dataBlockToSend;
    delete[] _lengthOfDataBlockToSend;
    delete[] _dataBlockToReceive;
    delete[] _lengthOfDataBlockToReceive;
    delete[] _bufSumDouble;
}

void Net::reset() {
    _numberOfDataBlocksToSend = 0;
    _numberOfDataBlocksToReceive = 0;
}

bool Net::addDataBlockToSend(void *dataBlock, uint64_t lengthOfDataBlock) {
    if (_numberOfDataBlocksToSend >= _maxNumberOfDataBlocks) {
        return false;
    }
    _dataBlockToSend[_numberOfDataBlocksToSend] = dataBlock;
    _lengthOfDataBlockToSend[_numberOfDataBlocksToSend] = lengthOfDataBlock;
    ++_numberOfDataBlocksToSend;
    return true;
}

bool Net::addDataBlockToReceive(void *dataBlock, uint64_t lengthOfDataBlock) {
    if (_numberOfDataBlocksToReceive >= _maxNumberOfDataBlocks) {
        return false;
    }
    _dataBlockToReceive[_numberOfDataBlocksToReceive] = dataBlock;
    _lengthOfDataBlockToReceive[_numberOfDataBlocksToReceive] =
            lengthOfDataBlock;
    ++_numberOfDataBlocksToReceive;
    return true;
}

void Net::debug() {
    printf("debug:%llu,%llu\n",_numberOfDataBlocksToReceive,_numberOfDataBlocksToSend);
}
void Net::exchangeDataBlocksWith(uint64_t procRank) {
    if (sizeof (char) != 1)
        MPI_Abort(_communicator, 1); //pointer type not available
    if (procRank == getProcessRank()) {
        uint64_t currentBlock;
        if (_numberOfDataBlocksToReceive != _numberOfDataBlocksToSend) {
            throw ECorruptedRuntime();
        }
        for (currentBlock = 0; currentBlock < _numberOfDataBlocksToReceive;
                ++currentBlock) {
            if (_lengthOfDataBlockToReceive[currentBlock]
                    != _lengthOfDataBlockToSend[currentBlock]) {
                throw ECorruptedRuntime();
            }
            memcpy(_dataBlockToReceive[currentBlock],
                    _dataBlockToSend[currentBlock],
                    _lengthOfDataBlockToReceive[currentBlock]);
        }
        _numberOfDataBlocksToSend = _numberOfDataBlocksToReceive = 0;
        return;
    }

    uint64_t restLengthOfDataBlock, lengthOfDataSegment, offset;
    uint64_t currentReceiveBlock, currentSendBlock, receiveTag, sendTag,
            numberOfSegmentToReceive;
    for (currentReceiveBlock = 0, numberOfSegmentToReceive = 0;
            currentReceiveBlock < _numberOfDataBlocksToReceive;
            ++currentReceiveBlock) {
        numberOfSegmentToReceive +=
                _lengthOfDataBlockToReceive[currentReceiveBlock]
                / _segmentThreshold + 1;
    }
    MPI_Status *statusOfReceive = new MPI_Status[numberOfSegmentToReceive];
    MPI_Request *requestOfReceive = new MPI_Request[numberOfSegmentToReceive];

    for (currentReceiveBlock = 0, receiveTag = 0;
            currentReceiveBlock < _numberOfDataBlocksToReceive;
            ++currentReceiveBlock) {
        restLengthOfDataBlock =
                _lengthOfDataBlockToReceive[currentReceiveBlock];
        offset = 0;
        do {
            if (restLengthOfDataBlock >= _segmentThreshold) {
                lengthOfDataSegment = _segmentThreshold;
            } else {
                lengthOfDataSegment = restLengthOfDataBlock;
            }
            MPI_Irecv(
                    (void*) ((char*) (_dataBlockToReceive[currentReceiveBlock])
                    + offset), lengthOfDataSegment, MPI_BYTE, procRank,
                    receiveTag, _communicator, &requestOfReceive[receiveTag]);
            offset += lengthOfDataSegment;
            restLengthOfDataBlock -= lengthOfDataSegment;
            ++receiveTag;
        } while (restLengthOfDataBlock > 0);
    }

    for (currentSendBlock = 0, sendTag = 0;
            currentSendBlock < _numberOfDataBlocksToSend; ++currentSendBlock) {
        restLengthOfDataBlock = _lengthOfDataBlockToSend[currentSendBlock];
        offset = 0;
        do {
            if (restLengthOfDataBlock >= _segmentThreshold) {
                lengthOfDataSegment = _segmentThreshold;
            } else {
                lengthOfDataSegment = restLengthOfDataBlock;
            }
            MPI_Send(
                    (void*) ((char*) (_dataBlockToSend[currentSendBlock])
                    + offset), lengthOfDataSegment, MPI_BYTE, procRank,
                    sendTag, _communicator);
            offset += lengthOfDataSegment;
            restLengthOfDataBlock -= lengthOfDataSegment;
            ++sendTag;
        } while (restLengthOfDataBlock > 0);
    }

    MPI_Waitall(receiveTag, requestOfReceive, statusOfReceive);

    delete[] requestOfReceive;
    delete[] statusOfReceive;

    _numberOfDataBlocksToSend = _numberOfDataBlocksToReceive = 0;
}

/// @brief get the processor name
/// @return processor name

std::string Net::getProcessorName() {
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(processorName, &nameLen);
    return processorName;
}

/// @brief get the number of processes
/// @return the number of processes

uint64_t Net::getNumberOfProcesses() {
    int commSize;
    MPI_Comm_size(_communicator, &commSize);
    return commSize;
}

/// @brief get the rank of current process
/// @return the rank of current process

uint64_t Net::getProcessRank() {
    int commRank;
    MPI_Comm_rank(_communicator, &commRank);
    return commRank;
}

/// @brief abort all processes

void Net::abort() {
    MPI_Abort(_communicator, -1);
}

void Net::finalize() {
    MPI_Finalize();
}

//! broadcast

void Net::broadcast(void *data, uint64_t length) {
    MPI_Bcast(data, length, MPI_CHAR, 0, _communicator);
}

//! send and receive

void Net::sendReceive(void *dataOut, void *dataIn, uint64_t length,
        uint64_t procID) {
    MPI_Status status;
    MPI_Sendrecv(dataOut, length, MPI_CHAR, procID, 0, dataIn, length, MPI_CHAR,
            procID, 0, _communicator, &status);
}

//! all gather

void Net::allGather(void *dataOut, void *dataIn, uint64_t lengthPerProc) {
    MPI_Allgather(dataOut, lengthPerProc, MPI_CHAR,
            dataIn, lengthPerProc, MPI_CHAR, _communicator);
}

void Net::broadcast(uint64_t rootProcID, uint64_t startProcID, uint64_t nProcsInGrid, void *data, uint64_t length) {
    // do nothing if it is an empty group
    if (nProcsInGrid <= 0)
        return;

    // do nothing if I am not in the group
    uint64_t myProcID = getProcessRank();
    if (myProcID < startProcID || myProcID >= startProcID + nProcsInGrid)
        return;

    // create group communicator for the grid
    MPI_Group fullGroup, newGroup;
    MPI_Comm_group(_communicator, &fullGroup);
    int *ranksInGroup = new int[nProcsInGrid];
    if (NULL == ranksInGroup)
        throw ENotEnoughMemory();
    for (uint64_t iProc = 0; iProc < nProcsInGrid; ++iProc) {
        ranksInGroup[iProc] = iProc + startProcID;
    }
    MPI_Group_incl(fullGroup, nProcsInGrid, ranksInGroup, &newGroup);
    MPI_Comm newComm;
    MPI_Comm_create(_communicator, newGroup, &newComm);

    // broadcast in the new grid
    MPI_Bcast(data, length, MPI_CHAR, rootProcID - startProcID, newComm);

    // release the new grid
    MPI_Comm_free(&newComm);
    delete[] ranksInGroup;
}

void Net::jointGather(void* dataOut, void* dataIn, uint64_t lengthPerProc,
        uint64_t startProcID, uint64_t nProcsInGrid) {
    // do nothing if it is an empty group
    if (nProcsInGrid <= 0)
        return;

    // do nothing if I am not in the group
    uint64_t myProcID = getProcessRank();
    if (myProcID < startProcID || myProcID >= startProcID + nProcsInGrid)
        return;

    // create group communicator for the grid
    MPI_Group fullGroup, newGroup;
    MPI_Comm_group(_communicator, &fullGroup);
    int *ranksInGroup = new int[nProcsInGrid];
    if (NULL == ranksInGroup)
        throw ENotEnoughMemory();
    for (uint64_t iProc = 0; iProc < nProcsInGrid; ++iProc) {
        ranksInGroup[iProc] = iProc + startProcID;
    }
    MPI_Group_incl(fullGroup, nProcsInGrid, ranksInGroup, &newGroup);
    MPI_Comm newComm;
    MPI_Comm_create(_communicator, newGroup, &newComm);

    // allgather in the new grid
    MPI_Allgather(dataOut, lengthPerProc, MPI_CHAR,
            dataIn, lengthPerProc, MPI_CHAR, newComm);

    // release the new grid
    MPI_Comm_free(&newComm);
    delete[] ranksInGroup;
}

//! all_sum_double
void Net::allSumDouble(double *data, uint64_t n) {
    if ((n >> 20) == 0) {
        int myN = (int)n;
        for (int i = 0; i < myN; ++ i) {
            _bufSumDouble[i] = data[i];
            data[i] = 0.0;
        }
        MPI_Allreduce(_bufSumDouble, data, myN, MPI_DOUBLE, MPI_SUM, _communicator);
    } else {
        uint64_t myM = n >> 20;
        int iBound = 1 << 20;
        for (uint64_t k = 0; k < myM; ++ k) {
            double *myData = data + (k << 20);
            for (int i = 0; i < iBound; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, iBound, MPI_DOUBLE, MPI_SUM, _communicator);
        }

        int myN = (int)(n & (iBound - 1));
        if (myN > 0) {
            double *myData = data + (myM << 20);
            for (int i = 0; i < myN; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, myN, MPI_DOUBLE, MPI_SUM, _communicator);
        }
    }
}

void Net::jointSumDouble(double *data, uint64_t n,
        uint64_t startProcID, uint64_t nProcsInGrid) {
    // do nothing if it is an empty group
    if (nProcsInGrid <= 0)
        return;

    // do nothing if I am not in the group
    uint64_t myProcID = getProcessRank();
    if (myProcID < startProcID || myProcID >= startProcID + nProcsInGrid)
        return;

    // create group communicator for the grid
    MPI_Group fullGroup, newGroup;
    MPI_Comm_group(_communicator, &fullGroup);
    int *ranksInGroup = new int[nProcsInGrid];
    if (NULL == ranksInGroup)
        throw ENotEnoughMemory();
    for (uint64_t iProc = 0; iProc < nProcsInGrid; ++iProc) {
        ranksInGroup[iProc] = iProc + startProcID;
    }
    MPI_Group_incl(fullGroup, nProcsInGrid, ranksInGroup, &newGroup);
    MPI_Comm newComm;
    MPI_Comm_create(_communicator, newGroup, &newComm);

    if ((n >> 20) == 0) {
        int myN = (int)n;
        for (int i = 0; i < myN; ++ i) {
            _bufSumDouble[i] = data[i];
            data[i] = 0.0;
        }
        MPI_Allreduce(_bufSumDouble, data, myN, MPI_DOUBLE, MPI_SUM, newComm);
    } else {
        uint64_t myM = n >> 20;
        int iBound = 1 << 20;
        for (uint64_t k = 0; k < myM; ++ k) {
            double *myData = data + (k << 20);
            for (int i = 0; i < iBound; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, iBound, MPI_DOUBLE, MPI_SUM, newComm);
        }

        int myN = (int)(n & (iBound - 1));
        if (myN > 0) {
            double *myData = data + (myM << 20);
            for (int i = 0; i < myN; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, myN, MPI_DOUBLE, MPI_SUM, newComm);
        }
    }

    // release the new grid
    MPI_Comm_free(&newComm);
    delete[] ranksInGroup;
}

void Net::allMaxDouble(double *data, uint64_t n) {
    if ((n >> 20) == 0) {
        int myN = (int)n;
        for (int i = 0; i < myN; ++ i) {
            _bufSumDouble[i] = data[i];
            data[i] = 0.0;
        }
        MPI_Allreduce(_bufSumDouble, data, myN, MPI_DOUBLE, MPI_MAX, _communicator);
    } else {
        uint64_t myM = n >> 20;
        int iBound = 1 << 20;
        for (uint64_t k = 0; k < myM; ++ k) {
            double *myData = data + (k << 20);
            for (int i = 0; i < iBound; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, iBound, MPI_DOUBLE, MPI_MAX, _communicator);
        }

        int myN = (int)(n & (iBound - 1));
        if (myN > 0) {
            double *myData = data + (myM << 20);
            for (int i = 0; i < myN; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, myN, MPI_DOUBLE, MPI_MAX, _communicator);
        }
    }
}

void Net::jointMaxDouble(double *data, uint64_t n,
        uint64_t startProcID, uint64_t nProcsInGrid) {
    // do nothing if it is an empty group
    if (nProcsInGrid <= 0)
        return;

    // do nothing if I am not in the group
    uint64_t myProcID = getProcessRank();
    if (myProcID < startProcID || myProcID >= startProcID + nProcsInGrid)
        return;

    // create group communicator for the grid
    MPI_Group fullGroup, newGroup;
    MPI_Comm_group(_communicator, &fullGroup);
    int *ranksInGroup = new int[nProcsInGrid];
    if (NULL == ranksInGroup)
        throw ENotEnoughMemory();
    for (uint64_t iProc = 0; iProc < nProcsInGrid; ++iProc) {
        ranksInGroup[iProc] = iProc + startProcID;
    }
    MPI_Group_incl(fullGroup, nProcsInGrid, ranksInGroup, &newGroup);
    MPI_Comm newComm;
    MPI_Comm_create(_communicator, newGroup, &newComm);

    if ((n >> 20) == 0) {
        int myN = (int)n;
        for (int i = 0; i < myN; ++ i) {
            _bufSumDouble[i] = data[i];
            data[i] = 0.0;
        }
        MPI_Allreduce(_bufSumDouble, data, myN, MPI_DOUBLE, MPI_MAX, newComm);
    } else {
        uint64_t myM = n >> 20;
        int iBound = 1 << 20;
        for (uint64_t k = 0; k < myM; ++ k) {
            double *myData = data + (k << 20);
            for (int i = 0; i < iBound; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, iBound, MPI_DOUBLE, MPI_MAX, newComm);
        }

        int myN = (int)(n & (iBound - 1));
        if (myN > 0) {
            double *myData = data + (myM << 20);
            for (int i = 0; i < myN; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, myN, MPI_DOUBLE, MPI_MAX, newComm);
        }
    }

    // release the new grid
    MPI_Comm_free(&newComm);
    delete[] ranksInGroup;
}

void Net::allMulDouble(double *data, uint64_t n) {
    if ((n >> 20) == 0) {
        int myN = (int)n;
        for (int i = 0; i < myN; ++ i) {
            _bufSumDouble[i] = data[i];
            data[i] = 0.0;
        }
        MPI_Allreduce(_bufSumDouble, data, myN, MPI_DOUBLE, MPI_PROD, _communicator);
    } else {
        uint64_t myM = n >> 20;
        int iBound = 1 << 20;
        for (uint64_t k = 0; k < myM; ++ k) {
            double *myData = data + (k << 20);
            for (int i = 0; i < iBound; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, iBound, MPI_DOUBLE, MPI_PROD, _communicator);
        }

        int myN = (int)(n & (iBound - 1));
        if (myN > 0) {
            double *myData = data + (myM << 20);
            for (int i = 0; i < myN; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, myN, MPI_DOUBLE, MPI_PROD, _communicator);
        }
    }
}

void Net::jointMulDouble(double *data, uint64_t n, uint64_t startProcID, uint64_t nProcsInGrid) {
    // do nothing if it is an empty group
    if (nProcsInGrid <= 0)
        return;

    // do nothing if I am not in the group
    uint64_t myProcID = getProcessRank();
    if (myProcID < startProcID || myProcID >= startProcID + nProcsInGrid)
        return;

    // create group communicator for the grid
    MPI_Group fullGroup, newGroup;
    MPI_Comm_group(_communicator, &fullGroup);
    int *ranksInGroup = new int[nProcsInGrid];
    if (NULL == ranksInGroup)
        throw ENotEnoughMemory();
    for (uint64_t iProc = 0; iProc < nProcsInGrid; ++iProc) {
        ranksInGroup[iProc] = iProc + startProcID;
    }
    MPI_Group_incl(fullGroup, nProcsInGrid, ranksInGroup, &newGroup);
    MPI_Comm newComm;
    MPI_Comm_create(_communicator, newGroup, &newComm);

    if ((n >> 20) == 0) {
        int myN = (int)n;
        for (int i = 0; i < myN; ++ i) {
            _bufSumDouble[i] = data[i];
            data[i] = 0.0;
        }
        MPI_Allreduce(_bufSumDouble, data, myN, MPI_DOUBLE, MPI_PROD, newComm);
    } else {
        uint64_t myM = n >> 20;
        int iBound = 1 << 20;
        for (uint64_t k = 0; k < myM; ++ k) {
            double *myData = data + (k << 20);
            for (int i = 0; i < iBound; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, iBound, MPI_DOUBLE, MPI_PROD, newComm);
        }

        int myN = (int)(n & (iBound - 1));
        if (myN > 0) {
            double *myData = data + (myM << 20);
            for (int i = 0; i < myN; ++ i) {
                _bufSumDouble[i] = myData[i];
                myData[i] = 0.0;
            }
            MPI_Allreduce(_bufSumDouble, myData, myN, MPI_DOUBLE, MPI_PROD, newComm);
        }
    }

    // release the new grid
    MPI_Comm_free(&newComm);
    delete[] ranksInGroup;
}

uint64_t Net::probe() {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, _communicator, &status);
    return (uint64_t)status.MPI_SOURCE;
}
