#include "BSPAlgGradientBasedOptimization.hpp"
#include <stdexcept>

using namespace BSP::Algorithm;

GradientBasedOptimization::GradientBasedOptimization(unsigned long nParams, FunValue funValue, unsigned long maxIter, double tol, Gradient gradient):
    Optimization(nParams, funValue, maxIter, tol), _gradient(gradient)
{
    _g = new double[_nParams];
    _newG = new double[_nParams];
    if (NULL == _g || NULL == _newG)
        throw std::runtime_error("not enough memory");
}

GradientBasedOptimization::~GradientBasedOptimization() {
    delete []_g;
    delete []_newG;
}

void GradientBasedOptimization::g() {
    _gradient(_nParams, _params, _g);
}

void GradientBasedOptimization::newG() {
    _gradient(_nParams, _newParams, _newG);
}

