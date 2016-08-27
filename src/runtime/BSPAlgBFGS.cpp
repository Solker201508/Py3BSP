#include "BSPAlgBFGS.hpp"
#include "BSPAlgLineSearch.hpp"
#include <stdexcept>
#include <cstring>
//#include <iostream>

using namespace BSP::Algorithm;

BFGS::BFGS(unsigned long nParams, FunValue funValue, unsigned long maxIter, Gradient gradient, double *params0):
    LineSearch(nParams, funValue, maxIter, gradient)
{
    _h = new double[_nParams * _nParams];
    _v = new double[_nParams * _nParams];
    _temp = new double[_nParams * _nParams];
    _y = new double[_nParams];
    _s = new double[_nParams];
    if (NULL == _direction || NULL == _h || NULL == _v || NULL == _temp || NULL == _y || NULL == _s)
        throw std::runtime_error("not enough memory");
    if (params0) {
        memcpy(_params, params0, sizeof(double) * _nParams);
    } else {
        for (unsigned long i = 0; i < _nParams; ++i) {
            _params[i] = 0.0;
        }
    }
    _iter = 0;
}

BFGS::~BFGS() {
    delete[] _h;
    delete[] _v;
    delete[] _temp;
    delete[] _y;
    delete[] _s;
}

void BFGS::minimize() {
    _toMaximize = false;
    optimize();
}

void BFGS::maximize() {
    _toMaximize = true;
    optimize();
}

void BFGS::optimize() {
    unsigned long k = 0;
    double diagValue = _toMaximize ? -1.0 : 1.0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        for (unsigned long j = 0; j < _nParams; ++j) {
            _h[k ++] = ((i == j) ? diagValue : 0.0);
        }
    }
    f();
    g();
    _g2 = 0.0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _g2 += _g[i] * _g[i];
    }
    memcpy(_direction, _g, sizeof(double) * _nParams);
    if (!_toMaximize) {
        for (unsigned long i = 0; i < _nParams; ++i) {
            _direction[i] = -_direction[i];
        }
    }
    for (_iter = 0; _iter < _maxIter; ++_iter) {
        LineSearch::optimize();
        newG();
        findDirection();
        if (_rho == 0.0)
            break;
        updateDf();
        //std::cout << "iter = " << _iter << ", f = " << _f << ", scale = " << reductionScale() << ", tol = " << _tol << std::endl;
        if (_newF != _newF)
            break;
        if (_toMaximize? _newF <= _f : _newF >= _f)
            break;
        if (_iter > 3 && reductionScale() < _tol)
            break;
        update();
    }
}

void BFGS::findDirection() {
    _rho = 0.0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _y[i] = _newG[i] - _g[i];
        _s[i] = _newParams[i] - _params[i];
        _rho += _y[i] * _s[i];
    }
    if (_rho == 0.0)
        return;
    _rho = 1.0 / _rho;

    unsigned long k = 0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        for (unsigned long j = 0; j < _nParams; ++j) {
            _v[k ++] = ((i == j) ? 1.0 : 0.0) - _rho * _y[i] * _s[j];
        }
    }
    
    k = 0;
    double *rowA = _h;
    for (unsigned long i = 0; i < _nParams; ++i) {
        for (unsigned long j = 0; j < _nParams; ++j) {
            double *b = _v + j;
            double elem = 0.0;
            for (unsigned long l = 0; l < _nParams; ++l) {
                elem += rowA[l] * *b;
                b += _nParams;
            }
            _temp[k ++] = elem;
        }
        rowA += _nParams;
    }

    k = 0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        for (unsigned long j = 0; j < _nParams; ++j) {
            double *a = _v + i;
            double *b = _temp + j;
            double elem = 0.0;
            for (unsigned long l = 0; l < _nParams; ++l) {
                elem += *a * *b;
                a += _nParams;
                b += _nParams;
            }
            _h[k ++] = elem + _rho * _s[i] * _s[j];
        }
    }
    
    _newG2 = 0.0;
    rowA = _h;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _newG2 += _newG[i] * _newG[i];
        _direction[i] = 0.0;
        for (unsigned long j = 0; j < _nParams; ++j) {
            _direction[i] -= rowA[j] * _newG[j];
        }
        rowA += _nParams;
    }
}

void BFGS::update() {
    memcpy(_params, _newParams, sizeof(double) * _nParams);
    _f = _newF;
    memcpy(_g, _newG, sizeof(double) * _nParams);
    _g2 = _newG2;
}
