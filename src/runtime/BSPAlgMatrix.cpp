#include "BSPAlgMatrix.hpp"
#include "BSPRuntime.hpp"
#include "BSPLocalArray.hpp"
#include "BSPAlgLU.hpp"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace BSP {
    namespace Algorithm {
        void matrixMul(GlobalArray *A, GlobalArray *B, GlobalArray *C) {
            assert(A != NULL && B != NULL & C != NULL);
            assert(A->getNumberOfDimensions() == 2 && C->getNumberOfDimensions() == B->getNumberOfDimensions());
            assert(A->getElementType() == ArrayShape::DOUBLE && B->getElementType() == ArrayShape::DOUBLE && C->getElementType() == ArrayShape::DOUBLE);
            uint64_t m = A->getElementCount(0);
            uint64_t n = A->getElementCount(1);
            assert(B->getElementCount(0) == n && C->getElementCount(0) == m);
            unsigned int nDimsB = B->getNumberOfDimensions();
            for (unsigned int iDim = 1; iDim < nDimsB; ++ iDim) {
                assert(B->getElementCount(iDim) == C->getElementCount(iDim));
            }
            uint64_t procStart = A->getStartProcID();
            uint64_t procCount = A->getProcCount(ArrayPartition::ALL_DIMS);
            assert(B->getStartProcID() == procStart && C->getStartProcID() == procStart);
            assert(B->getProcCount(ArrayPartition::ALL_DIMS) == procCount && C->getProcCount(ArrayPartition::ALL_DIMS));

            uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
            if (myProcID < procStart || myProcID >= procStart + procCount)
                return;

            unsigned int nBytesInBlockA;
            uint64_t blockSizeA[2];
            unsigned int localBlockCountA;
            double *localBlockA = (double *)A->blockScatter(0, 32, &nBytesInBlockA, blockSizeA, &localBlockCountA);
            unsigned int nElementsInBlockA = nBytesInBlockA / sizeof(double);

            unsigned int nBytesInBlockB;
            uint64_t blockSizeB[7];
            unsigned int localBlockCountB;
            double *localBlockB = (double *)B->blockScatter(nDimsB - 1, 32, &nBytesInBlockB, blockSizeB, &localBlockCountB);
            unsigned int nElementsInBlockB = nBytesInBlockB / sizeof(double);

            uint64_t lastDimSizeB = B->getElementCount(nDimsB - 1);
            unsigned int blockCountB = (lastDimSizeB + 31) / 32;
            uint64_t myIProc = myProcID - procStart;
            assert(localBlockCountB == blockCountB / procCount + (myIProc < blockCountB % procCount ? 1 : 0));
            
            uint64_t blockSizeC[7];
            blockSizeC[0] = blockSizeA[0];
            unsigned int nBytesInBlockC = sizeof(double) * blockSizeC[0];
            uint64_t nOuter = 1, nInner = lastDimSizeB;
            for (unsigned int iDim = 1; iDim < nDimsB; ++ iDim) {
                if (iDim + 1 < nDimsB) {
                    blockSizeC[iDim] = blockSizeB[iDim];
                    nBytesInBlockC *= blockSizeC[iDim];
                    nOuter *= blockSizeC[iDim];
                } else {
                    blockSizeC[iDim] = nInner;
                    nBytesInBlockC *= nInner;
                }
            }
            uint64_t nOuterInner = nOuter * nInner;
            unsigned int localBlockCountC = localBlockCountA;
            unsigned int nElementsInBlockC = nBytesInBlockC / sizeof(double);
            double *localBlockC = new double[nElementsInBlockC * localBlockCountC];
            Net *nal = Runtime::getActiveRuntime()->getNAL();

            memset(localBlockC, 0, nBytesInBlockC * localBlockCountC);

            uint64_t currIProc = myIProc;
            for (uint64_t iProc = 0; iProc < procCount; ++ iProc) {
                for (unsigned int iLocalBlockA = 0; iLocalBlockA < localBlockCountA; ++ iLocalBlockA) {
                    double *blockA = localBlockA + iLocalBlockA * nElementsInBlockA;
                    double *blockC = localBlockC + iLocalBlockA * nElementsInBlockC;
                    unsigned int iBlockA = myIProc + iLocalBlockA * procCount;
                    if (nDimsB == 1) {
                        unsigned int iBlockB = currIProc;
                        for (unsigned int iLocalBlockB = 0; iLocalBlockB < localBlockCountB; ++ iLocalBlockB) {
                            assert(iBlockB < blockCountB);
                            double *blockB = localBlockB + iLocalBlockB * nElementsInBlockB;
#pragma omp parallel for schedule(dynamic)
                            for (unsigned int iRow = 0; iRow < blockSizeA[0]; ++ iRow) {
                                double *rowA = blockA + iRow * n;
                                double rowInc = 0.0;
                                unsigned int j0 = iBlockB * 32;
                                unsigned int j1 = j0 + 32;
                                if (j1 > lastDimSizeB)
                                    j1 = lastDimSizeB;
                                for (unsigned int j = j0; j < j1; ++ j) {
                                    rowInc += rowA[j] * blockB[j - j0];
                                }
                                blockC[iRow] += rowInc;
                            }
                            iBlockB += procCount;
                        }
                    } else {
                        unsigned iBlockB = currIProc;
                        for (unsigned int iLocalBlockB = 0; iLocalBlockB < localBlockCountB; ++ iLocalBlockB) {
                            assert(iBlockB < blockCountB);
                            double *blockB = localBlockB + iLocalBlockB * nElementsInBlockB;
                            unsigned int maxILocalRow = blockSizeA[0];
                            if (iBlockA * 32 + maxILocalRow > n)
                                maxILocalRow = n - iBlockA * 32;
#pragma omp parallel for schedule(dynamic)
                            for (unsigned int iLocalRow = 0; iLocalRow < maxILocalRow; ++ iLocalRow) {
                                double *rowA = blockA + iLocalRow * n;
                                double *rowC = blockC + iLocalRow * nOuterInner;
                                for (unsigned int iOuter = 0; iOuter < nOuter; ++ iOuter) {
                                    double *segmentC = rowC + iOuter * nInner;
                                    unsigned int j0 = iBlockB * 32;
                                    unsigned int j1 = j0 + 32;
                                    if (j1 > lastDimSizeB)
                                        j1 = lastDimSizeB;
                                    for (unsigned int j = j0; j < j1; ++ j) {
                                        double *elemB = blockB + j - j0 + iOuter * 32;
                                        double value = 0.0;
                                        for (unsigned int k = 0; k < n; ++ k) {
                                            value += rowA[k] * *elemB;
                                            elemB += nOuter * 32;
                                        }
                                        segmentC[j] = value;
                                    }
                                }
                            }
                            iBlockB += procCount;
                        }
                    }
                }

                if (iProc + 1 >= procCount)
                    break;
                currIProc += procCount - 1;
                if (currIProc >= procCount)
                    currIProc -= procCount;
                uint64_t newLocalBlockCountB = blockCountB / procCount + (currIProc < blockCountB % procCount ? 1 : 0);
                double *newLocalBlockB = NULL;
                if (newLocalBlockCountB > 0) {
                    newLocalBlockB = new double[nElementsInBlockB * newLocalBlockCountB];
                }

                if (myIProc % 2 == 0) {
                    if (myIProc + 1 < procCount && localBlockCountB > 0) {
                        nal->addDataBlockToSend(localBlockB, localBlockCountB * nBytesInBlockB);
                    }
                    nal->exchangeDataBlocksWith(myProcID + 1);
                } else {
                    if (newLocalBlockCountB > 0) {
                        nal->addDataBlockToReceive(newLocalBlockB, newLocalBlockCountB * nBytesInBlockB);
                    }
                    nal->exchangeDataBlocksWith(myProcID - 1);
                }

                if (myIProc % 2 == 1) {
                    if (myIProc + 1 < procCount && localBlockCountB > 0) {
                        nal->addDataBlockToSend(localBlockB, localBlockCountB * nBytesInBlockB);
                    }
                    nal->exchangeDataBlocksWith(myProcID + 1);
                } else {
                    if (myIProc > 0 && newLocalBlockCountB > 0) {
                        nal->addDataBlockToReceive(newLocalBlockB, newLocalBlockCountB * nBytesInBlockB);
                    }
                    nal->exchangeDataBlocksWith(myProcID - 1);
                }

                if (myIProc + 1 >= procCount) {
                    if (localBlockCountB > 0) {
                        nal->addDataBlockToSend(localBlockB, localBlockCountB * nBytesInBlockB);
                    }
                } 
                if (myIProc == 0) {
                    if (newLocalBlockCountB > 0) {
                        nal->addDataBlockToReceive(newLocalBlockB, newLocalBlockCountB * nBytesInBlockB);
                    }
                }

                if (myIProc + 1 >= procCount) {
                    nal->exchangeDataBlocksWith(procStart);
                } else if (myIProc == 0) {
                    nal->exchangeDataBlocksWith(procStart + procCount - 1);
                }

                if (localBlockB)
                    delete[] localBlockB;
                localBlockB = newLocalBlockB;
                localBlockCountB = newLocalBlockCountB;
            }

            C->blockGather(0, 32, nBytesInBlockC, blockSizeC, localBlockCountC, (char *)localBlockC);

            if (localBlockA)
                delete[] localBlockA;
            if (localBlockB)
                delete[] localBlockB;
            if (localBlockC)
                delete[] localBlockC;
        }

        void transpose(GlobalArray *A, GlobalArray *At) {
            assert(A != NULL);
            assert(A->getNumberOfDimensions() == 2);
            assert(A->getElementCount(0) == A->getElementCount(1));
            assert(A->getElementType() == ArrayShape::DOUBLE);
            assert(At->getElementCount(0) == At->getElementCount(1) && At->getElementCount(0) == A->getElementCount(0));
            uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
            uint64_t procStart = A->getStartProcID();
            uint64_t procCount = A->getProcCount(ArrayPartition::ALL_DIMS);
            if (myProcID < procStart || myProcID >= procStart + procCount)
                return;

            unsigned int nBytesInBlock;
            uint64_t blockSize[2];
            unsigned int localBlockCount;
            double *localBlock = (double *)A->blockScatter(0, 32, &nBytesInBlock, blockSize, &localBlockCount);
            uint64_t m = blockSize[0];
            uint64_t n = blockSize[1];

            // 局部转置
            double *temp = new double[m * n];
            for (unsigned int iLocalBlock = 0; iLocalBlock < localBlockCount; ++ iLocalBlock) {
                double *currBlock = localBlock + iLocalBlock * m * n;
                memcpy(temp, currBlock, nBytesInBlock);
#pragma omp parallel for
                for (unsigned int i = 0; i < m; ++ i) {
                    for (unsigned int j = 0; j < n; ++ j) {
                        currBlock[j * m + i] = temp[i * n + j];
                    }
                }
            }

            blockSize[0] = n;
            blockSize[1] = m;
            At->blockGather(1, 32, nBytesInBlock, blockSize, localBlockCount, (const char *)localBlock);
            
            if (localBlock)
                delete[] localBlock;

            delete[] temp;
        }

        void inverse(GlobalArray *A, GlobalArray *invA) {
        }

        double tr(GlobalArray *A) {
            assert(A != NULL);
            assert(A->getNumberOfDimensions() == 2);
            assert(A->getElementCount(0) == A->getElementCount(1));
            assert(A->getElementType() == ArrayShape::DOUBLE);
            uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
            uint64_t procStart = A->getStartProcID();
            uint64_t procCount = A->getProcCount(ArrayPartition::ALL_DIMS);
            if (myProcID < procStart || myProcID >= procStart + procCount)
                return 0.0;
            unsigned int n = A->getElementCount(0);

            uint64_t diag, myDiag, lower[2], upper[2], pos[2];
            diag = 0;
            myDiag = 0;
            pos[0] = 0;
            pos[1] = 0;
            bool needMe = false;
            while (diag < n) {
                unsigned int procID = A->getProcIDFromPosition(pos);
                if (procID == myProcID) {
                    needMe = true;
                    myDiag = diag;
                    break;
                }
                A->getRange(procID, lower, upper);
                uint64_t myM = upper[0] - diag;
                uint64_t myN = upper[1] - diag;
                if (myM < myN) {
                    ++ pos[0];
                    diag += myM;
                } else if (myM > myN) {
                    ++ pos[1];
                    diag += myN;
                } else {
                    ++ pos[0];
                    ++ pos[1];
                    diag += myM;
                }
            }

            double myTrace = 1.0;
            if (needMe) {
                A->getLocalRange(lower, upper);
                LocalArray *localA = dynamic_cast<LocalArray *>(A->getRegistration()->getArrayShape(myProcID));
                double *localData = (double *)localA->getData();
                uint64_t myN = upper[1] - lower[1];
                uint64_t i0 = lower[0];
                if (i0 < myDiag)
                    i0 = myDiag;
                for (uint64_t i = i0; i < upper[0] && i < upper[1]; ++ i) {
                    double *row = localData + (i - lower[0]) * myN;
                    myTrace *= row[i - lower[1]];
                }
            }
            Runtime::getActiveRuntime()->getNAL()->jointMulDouble(&myTrace, 1, procStart, procCount);

            return myTrace;
        }

        double det(GlobalArray *A) {
            assert(A != NULL);
            assert(A->getNumberOfDimensions() == 2);
            assert(A->getElementCount(0) == A->getElementCount(1));
            assert(A->getElementType() == ArrayShape::DOUBLE);
            uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
            uint64_t procStart = A->getStartProcID();
            uint64_t procCount = A->getProcCount(ArrayPartition::ALL_DIMS);
            if (myProcID < procStart || myProcID >= procStart + procCount)
                return 0.0;

            double result = 0.0;
            LocalArray *localA = dynamic_cast<LocalArray *>(A->getRegistration()->getArrayShape(myProcID));
            char *localData = localA->getData();
            unsigned int lenData = localA->getElementCount(ArrayShape::ALL_DIMS) * localA->getNumberOfBytesPerElement();
            char *dataCopy = new char[lenData];
            memcpy(dataCopy, localData, lenData);
            try {
                LU lu(A, 32);
                result = BSP::Algorithm::tr(A);
                if (lu.isPOdd())
                    result = -result;
            } catch (const std::runtime_error &e) {
                if (0 == strcmp(e.what(), "the matrix is singular")) {
                    return 0.0;
                }
            }
            memcpy(localData, dataCopy, lenData);
            delete[] dataCopy;
            return result;
        }

        double norm1(GlobalArray *A) {
            assert(A != NULL);
            assert(A->getNumberOfDimensions() == 2);
            assert(A->getElementType() == ArrayShape::DOUBLE);
            uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
            uint64_t procStart = A->getStartProcID();
            uint64_t procCount = A->getProcCount(ArrayPartition::ALL_DIMS);
            if (myProcID < procStart || myProcID >= procStart + procCount)
                return 0.0;

            double maxColSumAbs = 0.0;
            if (A->getProcCount(1) == 1) {
                LocalArray *localA = dynamic_cast<LocalArray *>(A->getRegistration()->getArrayShape(myProcID));
                assert(localA != NULL);
                double *localData = (double *)localA->getData();
                uint64_t m = localA->getElementCount(0);
                uint64_t n = localA->getElementCount(1);
#pragma omp parallel for schedule(dynamic) reduction(max: maxColSumAbs)
                for (unsigned int j = 0; j < n; ++ j) {
                    double *col = localData + j;
                    double colSumAbs = 0.0;
                    for (uint64_t i = 0; i < m; ++ i) {
                        colSumAbs += fabs(*col);
                        col += n;
                    }
                    if (maxColSumAbs < colSumAbs)
                        maxColSumAbs = colSumAbs;
                }
            } else {
                unsigned int nBytesInBlock;
                uint64_t blockSize[2];
                unsigned int localBlockCount;
                double *localBlock = (double *)A->blockScatter(0, 32, &nBytesInBlock, blockSize, &localBlockCount);
                uint64_t m = blockSize[0];
                uint64_t n = blockSize[1];
#pragma omp parallel for schedule(dynamic) reduction(max: maxColSumAbs)
                for (unsigned int j = 0; j < n; ++ j) {
                    double *col = localBlock + j;
                    double colSumAbs = 0.0;
                    for (uint64_t i = 0; i < m; ++ i) {
                        colSumAbs += fabs(*col);
                        col += n;
                    }
                    if (maxColSumAbs < colSumAbs)
                        maxColSumAbs = colSumAbs;
                }
                if (localBlock)
                    delete[] (char *)localBlock;
            }
            Runtime::getActiveRuntime()->getNAL()->jointMaxDouble(&maxColSumAbs, 1, procStart, procCount);

            return maxColSumAbs;
        }

        double normInf(GlobalArray *A) {
            assert(A != NULL);
            assert(A->getNumberOfDimensions() == 2);
            assert(A->getElementType() == ArrayShape::DOUBLE);
            uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
            uint64_t procStart = A->getStartProcID();
            uint64_t procCount = A->getProcCount(ArrayPartition::ALL_DIMS);
            if (myProcID < procStart || myProcID >= procStart + procCount)
                return 0.0;

            double maxRowSumAbs = 0.0;
            if (A->getProcCount(0) == 1) {
                LocalArray *localA = dynamic_cast<LocalArray *>(A->getRegistration()->getArrayShape(myProcID));
                assert(localA != NULL);
                double *localData = (double *)localA->getData();
                uint64_t m = localA->getElementCount(0);
                uint64_t n = localA->getElementCount(1);
#pragma omp parallel for schedule(dynamic) reduction(max: maxRowSumAbs)
                for (unsigned int i = 0; i < m; ++ i) {
                    double *row = localData + i * n;
                    double rowSumAbs = 0.0;
                    for (unsigned int j = 0; j < n; ++ j) {
                        rowSumAbs += fabs(row[j]);
                    }
                    if (maxRowSumAbs < rowSumAbs)
                        maxRowSumAbs = rowSumAbs;
                }
            } else {
                unsigned int nBytesInBlock;
                uint64_t blockSize[2];
                unsigned int localBlockCount;
                double *localBlock = (double *)A->blockScatter(1, 32, &nBytesInBlock, blockSize, &localBlockCount);
                uint64_t m = blockSize[0];
                uint64_t n = blockSize[1];
#pragma omp parallel for schedule(dynamic) reduction(max: maxRowSumAbs)
                for (unsigned int i = 0; i < m; ++ i) {
                    double *row = localBlock + i * n;
                    double rowSumAbs = 0.0;
                    for (unsigned int j = 0; j < n; ++ j) {
                        rowSumAbs += fabs(row[j]);
                    }
                    if (maxRowSumAbs < rowSumAbs)
                        maxRowSumAbs = rowSumAbs;
                }
                if (localBlock)
                    delete[] (char *)localBlock;
            }
            Runtime::getActiveRuntime()->getNAL()->jointMaxDouble(&maxRowSumAbs, 1, procStart, procCount);

            return maxRowSumAbs;
        }

        double normF(GlobalArray *A) {
            assert(A != NULL);
            uint64_t myProcID = Runtime::getActiveRuntime()->getMyProcessID();
            uint64_t procStart = A->getStartProcID();
            uint64_t procCount = A->getProcCount(ArrayPartition::ALL_DIMS);
            if (myProcID < procStart || myProcID >= procStart + procCount)
                return 0.0;
            LocalArray *localA = dynamic_cast<LocalArray *>(A->getRegistration()->getArrayShape(myProcID));
            assert(localA != NULL);
            double localSum = 0.0;
            if (localA->getElementType() != ArrayShape::DOUBLE) {
                LocalArray copyA(ArrayShape::DOUBLE, *localA);
                double *localData = (double *)copyA.getData();
                uint64_t n = copyA.getElementCount(ArrayShape::ALL_DIMS);
#pragma omp parallel for schedule(dynamic,32) reduction(+: localSum)
                for (uint64_t i = 0; i < n; ++ i) {
                    localSum += localData[i] * localData[i];
                }
            } else {
                double *localData = (double *)localA->getData();
                uint64_t n = localA->getElementCount(ArrayShape::ALL_DIMS);
#pragma omp parallel for schedule(dynamic,32) reduction(+: localSum)
                for (uint64_t i = 0; i < n; ++ i) {
                    localSum += localData[i] * localData[i];
                }
            }
            Runtime::getActiveRuntime()->getNAL()->jointSumDouble(&localSum, 1, procStart, procCount);
            return localSum;
        }

        double norm2(GlobalArray *A) {
            return 0.0;
        }
    }
}

