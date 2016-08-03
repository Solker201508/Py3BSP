#ifndef __BSP_ALG_CG_HPP__
#define __BSP_ALG_CG_HPP__
#include "BSPAlgGradientBasedOptimization.hpp"
namespace BSP {
    namespace Algorithm {
        class CG: public GradientBasedOptimization {
            private:
                double *_direction;
                unsigned long _iter;
                double _g2, _newG2;

                bool _toMaximize;
            public:
                CG(unsigned long nParams, FunValue funValue, unsigned long maxIter, 
                        Gradient gradient, double tol = 1e-5, double *params0 = NULL);
                virtual ~CG();
                virtual void minimize();
                virtual void maximize();
            protected:
                void optimize();
                void findDirection();
                void update();
        };
    }
}

#endif

