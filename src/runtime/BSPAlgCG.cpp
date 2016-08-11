#include "BSPAlgCG.hpp"
#include "BSPAlgLineSearch.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>

using namespace BSP::Algorithm;

CG::CG(unsigned long nParams, FunValue funValue, unsigned long maxIter, Gradient gradient, double tol, double *params0):
    GradientBasedOptimization(nParams, funValue, maxIter, tol, gradient)
{
    _direction = new double[_nParams];
    if (NULL == _direction)
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

CG::~CG() {
    delete[] _direction;
}

void CG::minimize() {
    _toMaximize = false;
    optimize();
}

void CG::maximize() {
    _toMaximize = true;
    optimize();
}

void CG::optimize() {
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
        LineSearch lineSearch(_nParams, _funValue, _maxIter, _params, _direction);
        if (_toMaximize)
            lineSearch.maximize();
        else
            lineSearch.minimize();
        for (unsigned long i = 0; i < _nParams; ++i) {
            _newParams[i] = lineSearch.param(i);
        }
        newF();
        newG();
        findDirection();
        updateDf();
        std::cout << "iter = " << _iter << ", f = " << _f << ", scale = " << reductionScale() << ", tol = " << _tol << std::endl;
        if (_newF != _newF)
            break;
        if (_toMaximize? _newF <= _f : _newF >= _f) {
            break;
        }
        if (_iter > 3 && reductionScale() < _tol)
            break;
        update();
    }
}

void CG::findDirection() {
    _newG2 = 0.0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _newG2 += _newG[i] * _newG[i];
    }
    double beta = _newG2 / _g2;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _direction[i] = -_newG[i] + beta * _direction[i];
    }
}

void CG::update() {
    memcpy(_params, _newParams, sizeof(double) * _nParams);
    _f = _newF;
    memcpy(_g, _newG, sizeof(double) * _nParams);
    _g2 = _newG2;
}

