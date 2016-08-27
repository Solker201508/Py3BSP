#ifndef __BSP_ALG_CG_HPP__
#define __BSP_ALG_CG_HPP__
#include "BSPAlgLineSearch.hpp"
namespace BSP {
    namespace Algorithm {
        class CG: public LineSearch {
            private:
                unsigned long _iter;
                double _g2, _newG2;
            public:
                CG(unsigned long nParams, FunValue funValue, unsigned long maxIter, 
                        Gradient gradient, double *params0 = NULL);
                virtual ~CG();
                virtual void minimize();
                virtual void maximize();
            protected:
                virtual void optimize();
                void findDirection();
                void update();
        };
    }
}

#endif

