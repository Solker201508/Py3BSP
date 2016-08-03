#ifndef __BSP_ALG_LBFGS_HPP__
#define __BSP_ALG_LBFGS_HPP__
#include "BSPAlgGradientBasedOptimization.hpp"
namespace BSP {
    namespace Algorithm {
        class LBFGS: public GradientBasedOptimization {
            private:
                unsigned long _mLim;

                double **_y;
                double **_s;
                double *_rho;
                double *_direction;
                unsigned long _iter;

                bool _toMaximize;
                double _g2, _newG2;
            public:
                LBFGS(unsigned long nParams, FunValue funValue, unsigned long maxIter, 
                        Gradient gradient, unsigned long mLim, double tol = 1e-5, 
                        double *params0 = NULL);
                virtual ~LBFGS();
                virtual void minimize();
                virtual void maximize();
            protected:
                void optimize();
                void findDirection(unsigned long myMLim);
                void update();
        };
    }
}

#endif
