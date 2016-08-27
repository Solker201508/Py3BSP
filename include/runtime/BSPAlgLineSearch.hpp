#ifndef __BSP_ALG_LINE_SEARCH_HPP__
#define __BSP_ALG_LINE_SEARCH_HPP__
#include "BSPAlgGradientBasedOptimization.hpp"
namespace BSP {
    namespace Algorithm {
        class LineSearch : public GradientBasedOptimization {
            private:
                double *_params0;
                double *_prevParams;
                double _prevF;
                unsigned long _iter;
                double _u, _v, _w;
            public:
                LineSearch(unsigned long nParams, FunValue funValue, unsigned long maxIter, Gradient gradient);
                virtual ~LineSearch();
                virtual void minimize();
                virtual void maximize();
            protected:
                double *_direction;
                bool _toMaximize;
                virtual void optimize();
        };
    }
}
#endif

