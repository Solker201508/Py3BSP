#include "BSPAlgLineSearch.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <iostream>

using namespace BSP::Algorithm;

LineSearch::LineSearch(unsigned long nParams, FunValue funValue, unsigned long maxIter, double *params0, double *direction):
    Optimization(nParams, funValue, maxIter, 1.0)
{
    _prevParams = new double[_nParams];
    _direction = new double[_nParams];
    if (NULL == _prevParams || NULL == _direction)
        throw std::runtime_error("not enough memory");
    memcpy(_params, params0, _nParams * sizeof(double));
    memcpy(_direction, direction, _nParams * sizeof(double));
    _iter = 0;
    _toMaximize = false;
}

LineSearch::~LineSearch() {
    delete[] _prevParams;
    delete[] _direction;
}

void LineSearch::minimize() {
    _toMaximize = false;
    optimize();
}

void LineSearch::maximize() {
    _toMaximize = true;
    optimize();
}

void LineSearch::optimize() {
    for (unsigned long i = 0; i < _nParams; ++ i) {
        if (_direction[i] != _direction[i])
            return;
    }
    f();
    memcpy(_prevParams, _params, _nParams * sizeof(double));
    _prevF = _f;
    _u = 0.0;

    for (unsigned long i = 0; i < _nParams; ++i) {
        _params[i] = _prevParams[i] + _direction[i];
    }
    f();
    _v = 1.0;

    if (_toMaximize) {
        _w = (_f < _prevF) ? 0.5 : 2.0;
    } else {
        _w = (_f > _prevF) ? 0.5 : 2.0;
    }
    for (unsigned long i = 0; i < _nParams; ++i) {
        _newParams[i] = _prevParams[i] + _w * _direction[i];
    }
    newF();

    //std::cout << "0: f(u) = " << _prevF << ", f(v) = " << _f << ", f(w) = " << _newF << std::endl;

    for (_iter = 0; _iter < _maxIter; ++_iter) {
        double umvfw = (_u - _v) * _newF;
        double upv = _u + _v;
        double umwfv = (_u - _w) * _f;
        double upw = _u + _w;
        double vmwfu = (_v - _w) * _prevF;
        double vpw = _v + _w;
        double dominiator = vmwfu - umwfv + umvfw;
        if (dominiator == 0.0) {
            break;
        }
        double newW = 0.5 * (vmwfu * vpw - umwfv * upw + umvfw * upv) / dominiator;
        if (_toMaximize) {
            if (_newF < _f) {
                if (_newF < _prevF) {
                    // _newF is the smallest
                    if (_f < _prevF) {
                        for (unsigned long i = 0; i < _nParams; ++i) {
                            _params[i] = _prevParams[i];
                        }
                        _f = _prevF;
                    }
                    return;
                } else {
                    // _prevF is the smallest
                    memcpy(_prevParams, _newParams, _nParams * sizeof(double));
                    _u = _w;
                    _prevF = _newF;
                }
            } else {
                if (_f < _prevF) {
                    // _f is the smallest
                    memcpy(_params, _newParams, _nParams * sizeof(double));
                    _v = _w;
                    _f = _newF;
                } else {
                    // _prevF is the smallest
                    memcpy(_prevParams, _newParams, _nParams * sizeof(double));
                    _u = _w;
                    _prevF = _newF;
                }
            }
        } else {
            if (_newF > _f) {
                if (_newF > _prevF) {
                    // _newF is the largest
                    if (_f > _prevF) {
                        for (unsigned long i = 0; i < _nParams; ++i) {
                            _params[i] = _prevParams[i];
                        }
                        _f = _prevF;
                    }
                    return;
                } else {
                    // _prevF is the largest
                    memcpy(_prevParams, _newParams, _nParams * sizeof(double));
                    _u = _w;
                    _prevF = _newF;
                }
            } else {
                if (_f > _prevF) {
                    // _f is the largest
                    memcpy(_params, _newParams, _nParams * sizeof(double));
                    _v = _w;
                    _f = _newF;
                } else {
                    // _prevF is the largest
                    memcpy(_prevParams, _newParams, _nParams * sizeof(double));
                    _u = _w;
                    _prevF = _newF;
                }
            }
        }
        //std::cout << "0: f(u) = " << _prevF << ", f(v) = " << _f << ", f(w) = " << _newF << std::endl;
        for (unsigned long i = 0; i < _nParams; ++i) {
            _newParams[i] = _params[i] + (newW - _v) * _direction[i];
        }
        _w = newW;
        newF();
        if (fabs(_w - _v) < 1e-8 || fabs(_w - _u) < 1e-8) {
            return;
        }
    }
}
