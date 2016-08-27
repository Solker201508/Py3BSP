#ifndef __BSP_ALG_PARALLEL_OPTIMIZATION_HPP__
#define __BSP_ALG_PARALLEL_OPTIMIZATION_HPP__
#include "BSPAlgGradientBasedOptimization.hpp"
namespace BSP {
    namespace Algorithm {
        class ParallelOptimization {
            private:
                Optimization::FunValue _funValue;
                GradientBasedOptimization::Gradient _gradient;
            public:
                ParallelOptimization(Optimization::FunValue funValue, GradientBasedOptimization::Gradient gradient);
                ~ParallelOptimization();
                double f(unsigned long nParams, double *params);
                void g(unsigned long nParams, double *params, double *result);
        };
    }
}
double parallelFunValue(unsigned long nParams, double *params);
void parallelGradient(unsigned long nParams, double *params, double *result);

#endif

