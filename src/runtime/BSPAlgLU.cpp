#include "BSPAlgLU.hpp"
#include "BSPRuntime.hpp"
#include "BSPException.hpp"
#include "BSPGlobalArray.hpp"
#include <cmath>
#include <cassert>
#include <stdexcept>

using namespace BSP;
using namespace BSP::Algorithm;

LU::LU(GlobalArray *globalArray, unsigned int blockWidth) {
    if (globalArray == NULL)
        throw EInvalidArgument();
    if (globalArray->getElementType() != ArrayShape::DOUBLE || globalArray->getNumberOfDimensions() != 2
            || globalArray->getElementCount(0) != globalArray->getElementCount(1))
        throw EInvalidArgument();
    _oddPermutation = false;
    _globalArray = globalArray;
    _blockWidth = blockWidth;
    _localBlock = (double *)_globalArray->blockScatter(1, _blockWidth, &_nBytesInBlock, _blockSize, &_nLocalBlocks); 
    _nProcsInGrid = _globalArray->getProcCount(ArrayPartition::ALL_DIMS);
    unsigned int procStart = _globalArray->getStartProcID();
    _myIProc = (int)Runtime::getActiveRuntime()->getMyProcessID() - procStart;
    _nElementsInBlock = _nBytesInBlock / sizeof(double);
    unsigned int n = _blockSize[0];
    unsigned int myN = _blockSize[1];
    _P = NULL;
    _bufA = NULL;
    _bufPivot = NULL;
    if (_myIProc < 0 || _myIProc >= _nProcsInGrid)
        return;

    _P = new unsigned int[n];
    for (int i = 0; i < n; ++ i) {
        _P[i] = i;
    }
    _bufA = new double[n * myN];
    _bufPivot = new unsigned int[myN];

    Net *nal = Runtime::getActiveRuntime()->getNAL();
    unsigned int nBlocks = n / blockWidth + (n % blockWidth == 0 ? 0 : 1);
    for (unsigned int iBlock = 0; iBlock < nBlocks; ++ iBlock) {
        unsigned int j0 = iBlock * myN;
        unsigned int j1 = j0 + (j0 + myN > n ? n - j0 : myN);
        if (iBlock % _nProcsInGrid == _myIProc) {
            for (unsigned int j = j0; j < j1; ++ j) {
                permute(j, pivot(j));
                if (!localUpdate(j))
                    break;
            }
        }
        nal->broadcast(iBlock % _nProcsInGrid, procStart, _nProcsInGrid, _bufA, sizeof(double) * (n - j0) * myN);
        nal->broadcast(iBlock % _nProcsInGrid, procStart, _nProcsInGrid, _bufPivot, sizeof(unsigned int) * (j1 - j0));
        for (unsigned int myJ = 0; myJ < j1 - j0; ++ myJ) {
            if (_bufA[myJ * myN + myJ] == 0.0)
                throw std::runtime_error("the matrix is singular");
        }
        globalUpdate(j0, j1);
    }
    double *permutedBlock = NULL;
    if (_nLocalBlocks > 0) {
        permutedBlock = new double[_nLocalBlocks * _nElementsInBlock];
        unsigned int segmentWidth = myN * sizeof(double);
        for (unsigned int iLocalBlock = 0; iLocalBlock < _nLocalBlocks; ++ iLocalBlock) {
            double *block = _localBlock + iLocalBlock * _nElementsInBlock;
            double *newBlock = permutedBlock + iLocalBlock * _nElementsInBlock;
            for (int i = 0; i < n; ++ i) {
                memcpy(newBlock + i * myN, block + _P[i] * myN, segmentWidth);
            }
        }
    }
    globalArray->blockGather(1, _blockWidth, _nBytesInBlock, _blockSize, _nLocalBlocks, (const char *)permutedBlock);
    if (permutedBlock)
        delete[] permutedBlock;
}

LU::LU(GlobalArray *globalArray, unsigned int blockWidth, unsigned int *P) {
    if (globalArray == NULL)
        throw EInvalidArgument();
    if (globalArray->getElementType() != ArrayShape::DOUBLE || globalArray->getNumberOfDimensions() != 2
            || globalArray->getElementCount(0) != globalArray->getElementCount(1))
        throw EInvalidArgument();
    _oddPermutation = false;
    _globalArray = globalArray;
    _blockWidth = blockWidth;
    _localBlock = (double *)_globalArray->blockScatter(0, _blockWidth, &_nBytesInBlock, _blockSize, &_nLocalBlocks); 
    _nProcsInGrid = _globalArray->getProcCount(ArrayPartition::ALL_DIMS);
    unsigned int procStart = _globalArray->getStartProcID();
    _myIProc = (int)Runtime::getActiveRuntime()->getMyProcessID() - procStart;
    _nElementsInBlock = _nBytesInBlock / sizeof(double);
    unsigned int n = _blockSize[1];
    _P = new unsigned int[n];
    for (unsigned int i = 0; i < n; ++ i) {
        if (P[i] >= n)
            throw std::runtime_error("P is not a valid permutation");
        _P[i] = P[i];
    }
    _bufA = NULL;
    _bufPivot = NULL;
}

LU::~LU() {
    if (_nLocalBlocks > 0 && _localBlock != NULL)
        delete[] (char *)_localBlock;
    if (_P)
        delete[] _P;
    if (_bufA)
        delete[] _bufA;
    if (_bufPivot)
        delete[] _bufPivot;
}

unsigned int LU::getP(unsigned int i) {
    assert(i < _blockSize[0]);
    return _P[i];
}

unsigned int LU::pivot(unsigned int j) {
    unsigned int pivot = j; 
    unsigned int n = _blockSize[0];
    unsigned int myN = _blockSize[1];
    unsigned int iBlock = j / myN;
    assert(iBlock % _nProcsInGrid == _myIProc);
    unsigned int iLocalBlock = iBlock / _nProcsInGrid;
    unsigned int jInBlock = j - iBlock * myN;
    double *block = _localBlock + iLocalBlock * _nElementsInBlock;
    double vPivot = fabs(block[_P[j] * myN + jInBlock]);
    for (unsigned int i = j + 1; i < n; ++ i) {
        double vij = fabs(block[_P[i] * myN + jInBlock]);
        if (vij > vPivot) {
            pivot = i;
            vPivot = vij;
        }
    }
    _bufPivot[jInBlock] = pivot;
    return pivot;
}

void LU::permute(unsigned int j, unsigned int pivot) {
    if (j == pivot)
        return;
    unsigned int pj = _P[j];
    _P[j] = _P[pivot];
    _P[pivot] = pj;
    _oddPermutation = !_oddPermutation;
}

bool LU::localUpdate(unsigned int j) {
    unsigned int n = _blockSize[0];
    unsigned int myN = _blockSize[1];
    unsigned int iBlock = j / myN;
    assert(iBlock % _nProcsInGrid == _myIProc);
    unsigned int iLocalBlock = iBlock / _nProcsInGrid;
    unsigned int jInBlock = j - iBlock * myN;
    double *block = _localBlock + iLocalBlock * _nElementsInBlock;
    double ajj = block[_P[j] * myN + jInBlock];
    if (ajj == 0.0)
        return false;

    double oAjj = 1.0 / ajj;
    double *rowJ = block + _P[j] * myN;
    double *rowJInBufA = _bufA + jInBlock * myN;
    for (unsigned int myJ = 0; myJ < myN; ++ myJ) {
        rowJInBufA[myJ] = rowJ[myJ];
    }
    for (unsigned int i = j + 1; i < n; ++ i) {
        double *myRowI = block + _P[i] * myN;
        myRowI[jInBlock] *= oAjj;
        for (unsigned int myJ = jInBlock + 1; myJ < myN; ++ myJ) {
            myRowI[myJ] -= myRowI[jInBlock] * rowJ[myJ];
        }
    }
    if (jInBlock + 1 == myN || j + 1 == n) {
        double *rowIInBufA = rowJInBufA;
        for (unsigned int i = j + 1; i < n; ++ i) {
            rowIInBufA += myN;
            double *myRowI = block + _P[i] * myN;
            for (unsigned int myJ = 0; myJ < myN; ++ myJ) {
                rowIInBufA[myJ] = myRowI[myJ];
            }
        }
    }
    return true;
}

void LU::globalUpdate(unsigned int j0, unsigned int j1) {
    unsigned int n = _blockSize[0];
    unsigned int myN = _blockSize[1];
    unsigned int iBlock = j0 / myN;
    unsigned int iLocalBlock = iBlock / _nProcsInGrid;
    unsigned int j0InBlock = j0 - iBlock * myN;
    assert(j0InBlock == 0);
    assert(j1 <= j0 + myN);
    if (iBlock % _nProcsInGrid != _myIProc) {
        for (unsigned int j = j0; j < j1; ++ j) {
            permute(j, _bufPivot[j - j0]);
        }
    }
    if (iBlock % _nProcsInGrid <= _myIProc) {
        ++ iLocalBlock;
    }
#pragma omp parallel for schedule(dynamic)
    for (unsigned int jLocalBlock = iLocalBlock; jLocalBlock < _nLocalBlocks; ++ jLocalBlock) {
        double *block = _localBlock + jLocalBlock * _nElementsInBlock;
        for (unsigned int myJ = 0; myJ < j1 - j0; ++ myJ) {
            double *myRowJ = block + _P[j0 + myJ] * myN;
            for (unsigned int i = j0 + myJ + 1; i < n; ++ i) {
                double *rowI = _bufA + (i - j0) * myN;
                double *myRowI = block + _P[i] * myN;
                for (unsigned int myK = 0; myK < myN; ++ myK) {
                    myRowI[myK] -= rowI[myJ] * myRowJ[myK];
                }
            }
        }
    }
}

bool LU::isPOdd() {
    return _oddPermutation;
}

