#ifndef __BSP_ALG_LU_HPP__
#define __BSP_ALG_LU_HPP__
#include "BSPGlobalArray.hpp"
namespace BSP {
    namespace Algorithm {
        class LU {
            private:
                GlobalArray *_globalArray;
                unsigned int _nDims;
                unsigned int _blockWidth;
                unsigned int _nLocalBlocks;
                unsigned int _nBytesInBlock;
                unsigned int _nElementsInBlock;
                unsigned int _nProcsInGrid;
                bool _oddPermutation;
                int _myIProc;
                uint64_t _blockSize[7];
                unsigned int *_P;
                double *_localBlock;
                double *_bufA;
                unsigned int *_bufPivot;
            protected:
                unsigned int pivot(unsigned int j);
                void permute(unsigned int j, unsigned int pivot);
                bool localUpdate(unsigned int j);
                void globalUpdate(unsigned int j0, unsigned int j1);
            public:
                LU(GlobalArray *globalArray, unsigned int blockWidth);
                LU(GlobalArray *globalArray, unsigned int blockWidth, unsigned int *P);
                ~LU();
                unsigned int getP(unsigned int i);
                bool isPOdd();
                void solve(GlobalArray *Y, GlobalArray *X);
        };
    }
}
#endif

