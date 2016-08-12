#ifndef __BSP_ALG_LINE_SEARCH_HPP__
#define __BSP_ALG_LINE_SEARCH_HPP__
#include "BSPAlgGradientBasedOptimization.hpp"
namespace BSP {
    namespace Algorithm {
        class LineSearch : public GradientBasedOptimization {
            private:
                double *_direction;
                double *_prevParams;
                double _prevF;
                unsigned long _iter;
                bool _toMaximize;
                double _u, _v, _w;
            public:
                LineSearch(unsigned long nParams, FunValue funValue, unsigned long maxIter, double *params0, double *direction);
                virtual ~LineSearch();
                virtual void minimize();
                virtual void maximize();
            protected:
                void optimize();
        };
    }
}
#endif

