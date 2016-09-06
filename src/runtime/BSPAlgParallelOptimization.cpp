#include "BSPAlgParallelOptimization.hpp"
#include "BSPRuntime.hpp"
#include <cassert>
#include <iostream>

BSP::Algorithm::ParallelOptimization *activeParalleOptimzation_ = NULL;

double parallelFunValue(unsigned long nParams, double *params) {
    assert(NULL != activeParalleOptimzation_);
    return activeParalleOptimzation_->f(nParams, params);
}

void parallelGradient(unsigned long nParams, double *params, double *result) {
    assert(NULL != activeParalleOptimzation_);
    activeParalleOptimzation_->g(nParams, params, result);
}

using namespace BSP;
using namespace BSP::Algorithm;


ParallelOptimization::ParallelOptimization(Optimization::FunValue funValue, GradientBasedOptimization::Gradient gradient){
    assert(NULL == activeParalleOptimzation_);
    activeParalleOptimzation_ = this;
    _funValue = funValue;
    _gradient = gradient;
}

ParallelOptimization::~ParallelOptimization() {
    assert(this == activeParalleOptimzation_);
    activeParalleOptimzation_ = NULL;
}

double ParallelOptimization::f(unsigned long nParams, double *params) {
    double result = _funValue(nParams, params);
    Runtime::getActiveRuntime()->getNAL()->allSumDouble(&result, 1);
    return result;
}

void ParallelOptimization::g(unsigned long nParams, double *params, double *result) {
    _gradient(nParams, params, result);
    Runtime::getActiveRuntime()->getNAL()->allSumDouble(result, nParams);
}

