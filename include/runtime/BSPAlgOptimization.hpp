#ifndef __BSP_ALG_OPTIMIZATION_HPP__
#define __BSP_ALG_OPTIMIZATION_HPP__
#include <cstdlib>
double concensus(double proximityLevel, double centerLevel,
        unsigned long nWorkers, unsigned long nParamsPerWorker,
        double *params, double *multipliers, double *center);
namespace BSP {
    namespace Algorithm {
        class Optimization {
            public:
                typedef double (*FunValue)(unsigned long nParams, double *params);
            protected:
                unsigned long _nParams;
                FunValue _funValue;
                unsigned long _maxIter;

                double *_params;
                double *_newParams;
                double _f;
                double _newF;
                double _df;
                double _newDF;
                double _tol;
            public:
                Optimization(unsigned long nParams, FunValue funValue, unsigned long maxIter, double tol);
                virtual ~Optimization();
                virtual void f();
                virtual void newF();
                void updateDf();
                double reductionScale();
                virtual void minimize() = 0;
                virtual void maximize() = 0;
                inline double value() {return _f;}
                inline double param(unsigned long i) {return _params[i];}
        };
    }
}
#endif

