#ifndef __PCF_PY_MODULE_HPP__
#define __PCF_PY_MODULE_HPP__

extern "C" {
#include <Python.h>
}
extern "C" {
    void initBSP(int *pArgc, char ***pArgv);
    void finiBSP();
}

#endif

