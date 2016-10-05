#ifndef __BSP_ALG_MATRIX_HPP__
#define __BSP_ALG_MATRIX_HPP__
#include "BSPGlobalArray.hpp"

namespace BSP {
    namespace Algorithm {
        void matrixMul(GlobalArray *A, GlobalArray *B, GlobalArray *C);
        void transpose(GlobalArray *A, GlobalArray *At);
        void inverse(GlobalArray *A, GlobalArray *invA);
        double tr(GlobalArray *A);
        double det(GlobalArray *A);
        double norm1(GlobalArray *A);
        double normInf(GlobalArray *A);
        double normF(GlobalArray *A);
        double norm2(GlobalArray *A);
    }
}

#endif
