#ifndef __BSP_ALG_GRADIENT_BASED_OPTIMIZATION_HPP__
#define __BSP_ALG_GRADIENT_BASED_OPTIMIZATION_HPP__
#include "BSPAlgOptimization.hpp"
namespace BSP {
    namespace Algorithm {
        class GradientBasedOptimization: public Optimization {
            public:
                typedef void (*Gradient)(unsigned long nParams, double *params, double *result);
                enum Penalty {PENALTY_NONE, PENALTY_L2, PENALTY_L1, PENALTY_LOGSUM};
            protected:
                Gradient _gradient;

                double *_g;
                double *_newG;
                double *_coParams;
                double *_coMultipliers;
                double _penaltyLevel;
                double _coLevel;
                Penalty _penalty;
                unsigned long _coStart;
                unsigned long _coEnd;
            public:
                GradientBasedOptimization(unsigned long nParams, FunValue funValue, unsigned long maxIter, double tol, Gradient gradient);
                virtual ~GradientBasedOptimization();
                virtual void f();
                virtual void newF();
                virtual void g();
                virtual void newG();
                void setPenalty(Penalty penalty);
                void setPenaltyLevel(double level, bool toMaximize);
                void setCoLevel(double level, bool toMaximize);
                void setCoParams(double *coParams);
                void setCoMultipliers(double *multipliers);
                void setCoRange(long i);
                void setCoRange(unsigned long start, unsigned long end);
            private:
                double penaltyLogSum(double *params);
                double penaltyL1(double *params);
                double penaltyL2(double *params);

                void applyLogSumPenalty(double *params, double *result);
                void applyL1Penalty(double *params, double *result);
                void applyL2Penalty(double *params, double *result);

                double concensus(double *params);
                void applyConcensus(double *params, double *result);

                void richardson(double *params, double *g);
        };
    }
}
#endif
