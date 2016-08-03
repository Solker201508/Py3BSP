#ifndef __BSP_ALG_GRADIENT_BASED_OPTIMIZATION_HPP__
#define __BSP_ALG_GRADIENT_BASED_OPTIMIZATION_HPP__
#include "BSPAlgOptimization.hpp"
namespace BSP {
    namespace Algorithm {
        class GradientBasedOptimization: public Optimization {
            public:
                typedef void (*Gradient)(unsigned long nParams, double *params, double *result);
            protected:
                Gradient _gradient;

                double *_g;
                double *_newG;
            public:
                GradientBasedOptimization(unsigned long nParams, FunValue funValue, unsigned long maxIter, double tol, Gradient gradient);
                virtual ~GradientBasedOptimization();
                void g();
                void newG();
        };
    }
}
#endif
