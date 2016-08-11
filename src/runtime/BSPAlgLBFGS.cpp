#include "BSPAlgLBFGS.hpp"
#include "BSPAlgLineSearch.hpp"
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace BSP;
using namespace BSP::Algorithm;

LBFGS::LBFGS(unsigned long nParams, FunValue funValue, unsigned long maxIter, 
                        Gradient gradient, unsigned long mLim, double tol, double *params0):
    GradientBasedOptimization(nParams, funValue, maxIter, tol, gradient), _mLim(mLim)
{
    if (params0) {
        for (unsigned long i = 0; i < _nParams; ++i) {
            _params[i] = params0[i];
        }
    } else {
        for (unsigned long i = 0; i < _nParams; ++i) {
            _params[i] = 0.0;
        }
    }

    _y = new double*[_mLim];
    _s = new double*[_mLim];
    _rho = new double[_mLim];
    _direction = new double[_nParams];
    if (NULL == _y || NULL == _s || NULL == _rho || NULL == _direction)
        throw std::runtime_error("not enough memory");
    for (unsigned long i = 0; i < _mLim; ++i) {
        _y[i] = new double[_nParams];
        _s[i] = new double[_nParams];
        if (NULL == _y[i] || NULL == _s[i])
            throw std::runtime_error("not enough memory");
    }

    _toMaximize = false;
    _iter = 0;
} 

LBFGS::~LBFGS() {
    for (unsigned long i = 0; i < _mLim; ++i) {
        delete[] _y[i];
        delete[] _s[i];
    }
    delete[] _y;
    delete[] _s;
    delete[] _rho;
    delete[] _direction;
}

void LBFGS::maximize() {
    _toMaximize = true;
    optimize();
}

void LBFGS::minimize() {
    _toMaximize = false;
    optimize();
}

void LBFGS::optimize() {
    f();
    g();
    _g2 = 0.0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _g2 += _g[i] * _g[i];
    }
    memcpy(_direction, _g, sizeof(double) * _nParams);
    for (_iter = 0; _iter < _maxIter; ++_iter) {
        if (_iter < _mLim)
            findDirection(_iter);
        else
            findDirection(_mLim);
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
        _newG2 = 0.0;
        for (unsigned long i = 0; i < _nParams; ++i) {
            _newG2 += _newG[i] * _newG[i];
        }
        updateDf();
        //std::cout << "iter = " << _iter << ", f = " << _f << ", scale = " << reductionScale() << ", tol = " << _tol << std::endl;
        if (_iter > 3 && (reductionScale() < _tol || _newF == _f))
            break;
        update();
    }
}

void LBFGS::findDirection(unsigned long myMLim) {
    if (myMLim == 0) {
        if (!_toMaximize) {
            for (unsigned long i = 0; i < _nParams; ++i) {
                _direction[i] = -_direction[i];
            }
        }
        return;
    }

    double beta = 0.0;
    double *myS = _s[myMLim - 1];
    double *myY = _y[myMLim - 1];
    double myRho = _rho[myMLim - 1];

    for (unsigned long i = 0; i < _nParams; ++i) {
        beta += myS[i] * _direction[i];
    }
    double rhoBeta = myRho * beta;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _direction[i] -= rhoBeta * myY[i];
    }
    findDirection(myMLim - 1);
    double zeta = 0.0;
    for (unsigned long i = 0; i < _nParams; ++i) {
        zeta += myY[i] * _direction[i];
    }
    double rhoBZ = rhoBeta + myRho * zeta;
    for (unsigned long i = 0; i < _nParams; ++i) {
        _direction[i] -= rhoBZ * myS[i];
    }
}

void LBFGS::update() {
    if (_iter < _mLim) {
        double orho = 0.0;
        for (unsigned long i = 0; i < _nParams; ++i) {
            _s[_iter][i] = _newParams[i] - _params[i];
            _y[_iter][i] = _newG[i] - _g[i];
            orho += _y[_iter][i] * _s[_iter][i];
        }
        _rho[_iter] = 1.0 / orho;
    } else {
        double *myS = _s[0];
        double *myY = _y[0];
        for (unsigned long k = 1; k < _mLim; ++k) {
            _s[k - 1] = _s[k];
            _y[k - 1] = _y[k];
            _rho[k - 1] = _rho[k];
        }
        _s[_mLim - 1] = myS;
        _y[_mLim - 1] = myY;
        double orho = 0.0;
        for (unsigned long i = 0; i < _nParams; ++i) {
            myS[i] = _newParams[i] - _params[i];
            myY[i] = _newG[i] - _g[i];
            orho += myS[i] * myY[i];
        }
        _rho[_mLim - 1] = 1.0 / orho;
    }
    memcpy(_params, _newParams, sizeof(double) * _nParams);
    memcpy(_g, _newG, sizeof(double) * _nParams);
    _f = _newF;
    memcpy(_direction, _newG, sizeof(double) * _nParams);
    _g2 = _newG2;
}

