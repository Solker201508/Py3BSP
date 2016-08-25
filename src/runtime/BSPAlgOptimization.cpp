#include "BSPAlgOptimization.hpp"
#include <cmath>
#include <stdexcept>

void concensus(double proximityLevel, double centerLevel,
        unsigned long nWorkers, unsigned long nParamsPerWorker,
        double *params, double *multipliers, double *center) {
    proximityLevel = fabs(proximityLevel);
    bool reflected = false;
    if (centerLevel < 0) {
        reflected = true;
        centerLevel = -centerLevel;
        unsigned long nParams = nParamsPerWorker * nWorkers;
        for (unsigned long k = 0; k < nParams; ++ k) {
            multipliers[k] = -multipliers[k];
        }
    }
    double dominator = nWorkers * centerLevel + proximityLevel;
    double scale = 1.0 / dominator;
    for (unsigned int i = 0; i < nParamsPerWorker; ++ i) {
        center[i] *= proximityLevel;
        unsigned int k = i;
        for (unsigned int j = 0; j < nWorkers; ++ j) {
            center[i] += multipliers[k] + centerLevel * params[k];
            k += nParamsPerWorker;
        }
        center[i] *= scale;

        k = i;
        for (unsigned int j = 0; j < nWorkers; ++ j) {
            multipliers[k] += centerLevel * (params[k] - center[i]);
            k += nParamsPerWorker;
        }
    }

    if (reflected) {
        unsigned long nParams = nParamsPerWorker * nWorkers;
        for (unsigned long k = 0; k < nParams; ++ k) {
            multipliers[k] = -multipliers[k];
        }
    }
}

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

