#include "BSPAlgLineSearch.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <iostream>

using namespace BSP::Algorithm;

LineSearch::LineSearch(unsigned long nParams, FunValue funValue, unsigned long maxIter, double *params0, double *direction):
    GradientBasedOptimization(nParams, funValue, maxIter, 1.0, NULL)
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
    double d2 = 0.0;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        if (_direction[i] != _direction[i])
            return;
        d2 += _direction[i] * _direction[i];
    }
    f();
    if (d2 == 0.0)
        return;
    //std::cout << "d2 = " << d2 << std::endl;

    memcpy(_prevParams, _params, _nParams * sizeof(double));
    _prevF = _f;
    _u = 0.0;

    _v = 1.0/ sqrt(d2);
    for (unsigned long i = 0; i < _nParams; ++i) {
        _params[i] = _prevParams[i] + _v * _direction[i];
    }
    f();

    if (_toMaximize) {
        while (_f < _prevF) {
            _v *= 0.5;
            for (unsigned long i = 0; i < _nParams; ++i) {
                _params[i] = _prevParams[i] + _v * _direction[i];
            }
            f();
        }
    } else {
        while (_f > _prevF) {
            _v *= 0.5;
            for (unsigned long i = 0; i < _nParams; ++i) {
                _params[i] = _prevParams[i] + _v * _direction[i];
            }
            f();
        }
    }
    _w = 0.5 * _v;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _newParams[i] = _prevParams[i] + _w * _direction[i];
    }
    newF();

    //std::cout << "0: f(u) = " << _prevF << ", f(v) = " << _f << ", f(w) = " << _newF << std::endl;
    //std::cout << "0: u = " << _u << ", v = " << _v << ", w = " << _w << std::endl;

    for (_iter = 0; _iter < _maxIter; ++_iter) {
        double umvfw = (_u - _v) * _newF;
        double upv = _u + _v;
        double umwfv = (_u - _w) * _f;
        double upw = _u + _w;
        double vmwfu = (_v - _w) * _prevF;
        double vpw = _v + _w;
        double dominiator = vmwfu - umwfv + umvfw;
        if (dominiator == 0.0) {
            //std::cout << "dominiator == 0 " << std::endl;
            break;
        }
        double newW = 0.5 * (vmwfu * vpw - umwfv * upw + umvfw * upv) / dominiator;
        if (_toMaximize) {
            if (_newF < _f) {
                if (_newF < _prevF) {
                    // _newF is the smallest
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
        //std::cout << _iter << ": f(u) = " << _prevF << ", f(v) = " << _f << ", f(w) = " << _newF << std::endl;
        //std::cout << _iter << ": u = " << _u << ", v = " << _v << ", w = " << _w << std::endl;
        if (fabs(newW - _w) < 1e-8 * fabs(_w)) {
            //std::cout << _iter << "newW == w" << std::endl;
            break;
        }
        //std::cout << _iter << ": newW - w = " << newW - _w << std::endl;
        for (unsigned long i = 0; i < _nParams; ++i) {
            _newParams[i] = _params[i] + (newW - _v) * _direction[i];
        }
        _w = newW;
        newF();
        if (_newF != _newF) {
            break;
        }

        if (fabs(_w - _v) < 1e-8 || fabs(_w - _u) < 1e-8) {
            break;
        }
    }

    if (_toMaximize) {
        if (_newF == _newF) {
            if (_newF <= _f) {
                if (_f < _prevF) {
                    memcpy(_params, _prevParams, _nParams * sizeof(double));
                    _f = _prevF;
                } else {
                }
            } else {
                if (_newF < _prevF) {
                    memcpy(_params, _prevParams, _nParams * sizeof(double));
                    _f = _prevF;
                } else {
                    memcpy(_params, _newParams, _nParams * sizeof(double));
                    _f = _newF;
                }
            }
        } else {
            if (_prevF > _f) {
                memcpy(_params, _prevParams, _nParams * sizeof(double));
                _f = _prevF;
            }
        }
    } else {
        if (_newF == _newF) {
            if (_newF >= _f) {
                if (_f > _prevF) {
                    memcpy(_params, _prevParams, _nParams * sizeof(double));
                    _f = _prevF;
                }
            } else {
                if (_newF > _prevF) {
                    memcpy(_params, _prevParams, _nParams * sizeof(double));
                    _f = _prevF;
                } else {
                    memcpy(_params, _newParams, _nParams * sizeof(double));
                    _f = _newF;
                }
            }
        } else {
            if (_prevF < _f) {
                memcpy(_params, _prevParams, _nParams * sizeof(double));
                _f = _prevF;
            }
        }
    }
}
