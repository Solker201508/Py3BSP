#ifndef __BSP_ALG_BFGS_HPP__
#define __BSP_ALG_BFGS_HPP__
#include "BSPAlgLineSearch.hpp"
namespace BSP {
    namespace Algorithm {
        class BFGS: public LineSearch {
            private:
                double *_h;
                double *_v;
                double *_temp;
                double *_y;
                double *_s;
                double _rho;
                unsigned long _iter;
                double _g2, _newG2;

            public:
                BFGS(unsigned long nParams, FunValue funValue, unsigned long maxIter, 
                        Gradient gradient, double *params0 = NULL);
                virtual ~BFGS();
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

