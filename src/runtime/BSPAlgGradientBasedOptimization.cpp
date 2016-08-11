#include "BSPAlgGradientBasedOptimization.hpp"
#include <stdexcept>
#include <cmath>

using namespace BSP::Algorithm;

GradientBasedOptimization::GradientBasedOptimization(unsigned long nParams, FunValue funValue, unsigned long maxIter, double tol, Gradient gradient):
    Optimization(nParams, funValue, maxIter, tol), _gradient(gradient)
{
    _g = new double[_nParams];
    _newG = new double[_nParams];
    if (NULL == _g || NULL == _newG)
        throw std::runtime_error("not enough memory");

    _coParams = NULL;
    _coMultipliers = NULL;
    _proximity = NULL;
    _penaltyLevel = 0.0;
    _coLevel = 0.0;
    _proximityLevel = 0.0;
    _penalty = PENALTY_NONE;
}

GradientBasedOptimization::~GradientBasedOptimization() {
    delete []_g;
    delete []_newG;
    if (_coParams)
        delete []_coParams;
    if (_coMultipliers)
        delete []_coMultipliers;
    if (_proximity)
        delete []_proximity;
}

void GradientBasedOptimization::setPenaltyLevel(double level, bool toMaximize) {
    if (toMaximize) {
        _penaltyLevel = -fabs(level);
    } else {
        _penaltyLevel = fabs(level);
    }
}

void GradientBasedOptimization::setCoLevel(double level, bool toMaximize) {
    if (toMaximize) {
        _coLevel = -fabs(level);
    } else {
        _coLevel = fabs(level);
    }
}

void GradientBasedOptimization::setCoParams(double *coParams) {
    if (NULL == _coParams) {
        _coParams = new double[_nParams];
    }
    for (unsigned long i = 0; i < _nParams; ++i) {
        _coParams[i] = coParams[i];
    }
}

void GradientBasedOptimization::setCoMultipliers(double *multipliers) {
    if (NULL == _coMultipliers) {
        _coMultipliers = new double[_nParams];
    }
    for (unsigned long i = 0; i < _nParams; ++ i) {
        _coMultipliers[i] = multipliers[i];
    }
}

void GradientBasedOptimization::setProximity(double *proximity) {
    if (NULL == _proximity) {
        _proximity = new double[_nParams];
    }
    for (unsigned long i = 0; i < _nParams; ++ i) {
        _proximity[i] = proximity[i];
    }
}

void GradientBasedOptimization::setProximityLevel(double proximityLevel, bool toMaximize) {
    if (toMaximize) {
        _proximityLevel = -fabs(proximityLevel);
    } else {
        _proximityLevel = fabs(proximityLevel);
    }
}


void GradientBasedOptimization::f() {
    Optimization::f();
    _f += proximity(_params);
    _f += concensus(_params);
    switch (_penalty) {
        case PENALTY_LOGSUM:
            _f += penaltyLogSum(_params);
            break;
        case PENALTY_L1:
            _f += penaltyL1(_params);
            break;
        case PENALTY_L2:
            _f += penaltyL2(_params);
            break;
        default:
            break;
    }
}

void GradientBasedOptimization::newF() {
    Optimization::newF();
    _newF += proximity(_newParams);
    _newF += concensus(_newParams);
    switch (_penalty) {
        case PENALTY_LOGSUM:
            _newF += penaltyLogSum(_newParams);
            break;
        case PENALTY_L1:
            _newF += penaltyL1(_newParams);
            break;
        case PENALTY_L2:
            _newF += penaltyL2(_newParams);
            break;
        default:
            break;
    }
}

void GradientBasedOptimization::g() {
    if (_gradient) {
        _gradient(_nParams, _params, _g);
    } else {
        richardson(_params, _g);
    }
    applyProximity(_params, _g);
    applyConcensus(_params, _g);
    switch (_penalty) {
        case PENALTY_LOGSUM:
            applyLogSumPenalty(_params, _g);
            break;
        case PENALTY_L1:
            applyL1Penalty(_params, _g);
            break;
        case PENALTY_L2:
            applyL2Penalty(_params, _g);
            break;
        default:
            break;
    }
}

void GradientBasedOptimization::newG() {
    if (_gradient) {
        _gradient(_nParams, _newParams, _newG);
    } else {
        richardson(_newParams, _newG);
    }
    applyProximity(_newParams, _newG);
    applyConcensus(_newParams, _newG);
    switch (_penalty) {
        case PENALTY_LOGSUM:
            applyLogSumPenalty(_newParams, _newG);
            break;
        case PENALTY_L1:
            applyL1Penalty(_newParams, _newG);
            break;
        case PENALTY_L2:
            applyL2Penalty(_newParams, _newG);
            break;
        default:
            break;
    }
}

void GradientBasedOptimization::setPenalty(Penalty penalty) {
    _penalty = penalty;
}

void GradientBasedOptimization::applyLogSumPenalty(double *params, double *result) {
    if (_penaltyLevel == 0.0)
        return;
    if (_penaltyLevel > 0) {
        for (unsigned long i = 0; i < _nParams; ++ i) {
            if (params[i] == 0) {
                if (result[i] > 0) {
                    if (result[i] <= _penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] -= _penaltyLevel;
                } else if (result[i] < 0) {
                    if (result[i] >= -_penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] += _penaltyLevel;
                }
            } else if (params[i] > 0) {
                result[i] += _penaltyLevel / (1.0 + params[i]);
            } else {
                result[i] -= _penaltyLevel / (1.0 - params[i]);
            }
        }
    } else {
        for (unsigned long i = 0; i < _nParams; ++ i) {
            if (params[i] == 0) {
                if (result[i] > 0) {
                    if (result[i] <= -_penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] += _penaltyLevel;
                } else if (result[i] < 0) {
                    if (result[i] >= _penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] -= _penaltyLevel;
                } 
            } else if (params[i] > 0) {
                result[i] += _penaltyLevel / (1.0 + params[i]);
            } else {
                result[i] -= _penaltyLevel / (1.0 - params[i]);
            }
        }
    }
}

void GradientBasedOptimization::applyL1Penalty(double *params, double *result) {
    if (_penaltyLevel == 0.0)
        return;
    if (_penaltyLevel > 0) {
        for (unsigned long i = 0; i < _nParams; ++ i) {
            if (params[i] == 0) {
                if (result[i] > 0) {
                    if (result[i] <= _penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] -= _penaltyLevel;
                } else if (result[i] < 0) {
                    if (result[i] >= -_penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] += _penaltyLevel;
                }
            } else if (params[i] > 0) {
                result[i] += _penaltyLevel;
            } else {
                result[i] -= _penaltyLevel;
            }
        }
    } else {
        for (unsigned long i = 0; i < _nParams; ++ i) {
            if (params[i] == 0) {
                if (result[i] > 0) {
                    if (result[i] <= -_penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] += _penaltyLevel;
                } else if (result[i] < 0) {
                    if (result[i] >= _penaltyLevel)
                        result[i] = 0;
                    else
                        result[i] -= _penaltyLevel;
                } 
            } else if (params[i] > 0) {
                result[i] += _penaltyLevel;
            } else {
                result[i] -= _penaltyLevel;
            }
        }
    }
}

void GradientBasedOptimization::applyL2Penalty(double *params, double *result) {
    if (_penaltyLevel == 0.0)
        return;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        result[i] += _penalty * params[i];
    }
}

void GradientBasedOptimization::applyProximity(double *params, double *result) {
    if (NULL == _proximity || 0 == _proximityLevel)
        return;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        double d = params[i] - _proximity[i];
        result[i] += _proximityLevel * d;
    }
}

void GradientBasedOptimization::applyConcensus(double *params, double *result) {
    if (NULL == _coParams || NULL == _coMultipliers)
        return;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        double d = params[i] - _coParams[i];
        result[i] += _coLevel * d + _coMultipliers[i];
    }
}

double GradientBasedOptimization::penaltyLogSum(double *params) {
    double result = 0.0;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        result += log(1.0 + fabs(params[i]));
    }
    result *= _penaltyLevel;
    return result;
}

double GradientBasedOptimization::penaltyL1(double *params) {
    double result = 0.0;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        result += fabs(params[i]);
    }
    result *= _penaltyLevel;
    return result;
}

double GradientBasedOptimization::penaltyL2(double *params) {
    double result = 0.0;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        result += params[i] * params[i];
    }
    result *= 0.5 * _penaltyLevel;
    return result;
}

double GradientBasedOptimization::proximity(double *params) {
    if (NULL == _proximity || 0 == _proximityLevel)
        return 0.0;
    double result = 0.0;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        double d = params[i] - _proximity[i];
        result += d * d;
    }
    result *= 0.5 * _proximityLevel;
    return result;
}

double GradientBasedOptimization::concensus(double *params) {
    if (NULL == _coParams || NULL == _coMultipliers)
        return 0.0;
    double result = 0.0;
    for (unsigned long i = 0; i < _nParams; ++ i) {
        double d = params[i] - _coParams[i];
        result += (0.5 * _coLevel * d + _coMultipliers[i]) * d;
    }
    return result;
}

void GradientBasedOptimization::richardson(double *params, double *g) {
    double G[8];
    double *x = new double[_nParams];
    for (unsigned long i = 0; i < _nParams; ++ i) {
        x[i] = params[i];
    }
    for (unsigned long i = 0; i < _nParams; ++ i) {
        double h = 1e-6;
        for (unsigned int j = 0; j < 8; ++ j) {
            x[i] = params[i] + h;
            double fp = _funValue(_nParams, x);
            x[i] = params[i] - h;
            double fn = _funValue(_nParams, x);
            G[j] = (fp - fn) / (2.0 * h);
            h *= 0.5;
        }
        double w = 1;
        for (unsigned int k = 1; k < 8; ++ k) {
            w *= 4;
            for (unsigned int j = k; j < 8; ++ j) {
                G[j - k] = (w * G[j - k + 1] - G[j - k]) / (w - 1);
            }
        }
        g[i] = G[0];
    }
    delete []x;
}

