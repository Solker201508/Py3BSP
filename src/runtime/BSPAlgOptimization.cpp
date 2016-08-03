#include "BSPAlgOptimization.hpp"
#include <cmath>
#include <stdexcept>

using namespace BSP::Algorithm;

Optimization::Optimization(unsigned long nParams, FunValue funValue, unsigned long maxIter, double tol):
    _nParams(nParams), _funValue(funValue), _maxIter(maxIter), _tol(tol)
{
    _params = new double[_nParams];
    _newParams = new double[_nParams];
    _df = 0.0;
    _newDF = 0.0;
    if (NULL == _params || NULL == _newParams)
        throw std::runtime_error("not enough memory");
}

Optimization::~Optimization() {
    delete[] _params;
    delete[] _newParams;
}

void Optimization::f() {
    _f = _funValue(_nParams, _params);
}

void Optimization::newF() {
    _newF = _funValue(_nParams, _newParams);
}

void Optimization::updateDf() {
    _df = _newDF;
    _newDF = _f - _newF;
}

double Optimization::reductionScale() {
    if (_df == 0.0)
        return 1.0;
    else
        return _newDF / _df;
}
