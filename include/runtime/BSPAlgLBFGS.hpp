#ifndef __BSP_ALG_LBFGS_HPP__
#define __BSP_ALG_LBFGS_HPP__
#include "BSPAlgLineSearch.hpp"
namespace BSP {
    namespace Algorithm {
        class LBFGS: public LineSearch {
            private:
                unsigned long _mLim;

                double **_y;
                double **_s;
                double *_rho;
                unsigned long _iter;
                double _g2, _newG2;
            public:
                LBFGS(unsigned long nParams, FunValue funValue, unsigned long maxIter, 
                        Gradient gradient, unsigned long mLim,  
                        double *params0 = NULL);
                virtual ~LBFGS();
                virtual void minimize();
                virtual void maximize();
            protected:
                virtual void optimize();
                void findDirection(unsigned long myMLim);
                void update();
        };
    }
}

#endif
