#include "module.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "BSP.hpp"
#include <cassert>
#include <string>
#include <cstring>
#include <sstream>
#include <map>
#include <iostream>
#include <sstream>
#include <cmath>
#include <sys/time.h>

using namespace BSP;
using namespace BSP::Algorithm;
PyObject *pickle_ = NULL;
PyObject *ctypes_ = NULL;
PyObject *pickle_dumps_ = NULL;
PyObject *pickle_loads_ = NULL;
PyObject *ctypes_addressof_ = NULL;
PyObject *traceback_ = NULL;
PyObject *traceback_extractStack_ = NULL;
PyObject **fromProc_ = NULL;
PyObject *module_ = NULL;
uint64_t nProcs_ = 0;
struct timeval tvStart_, tvStop_;
std::map<std::string, PyObject *> futures_;
std::map<PyObject *, std::string> futureIDs_;
std::map<std::string, std::map<std::string, LocalArray *> > requests_;
std::map<std::string, std::map<std::string, LocalArray *> > updates_;
std::string scriptPos_;

Runtime *runtime_ = NULL;

void bsp_getScriptPos() {
    std::stringstream ss;
    //ss << runtime_->getMyProcessID() << ": ------ call stack begin ------ " << std::endl;
    PyObject *param = Py_BuildValue("()");
    PyObject *stack = PyObject_CallObject(traceback_extractStack_,param);
    Py_DECREF(param);
    if (stack) {
        Py_ssize_t stackSize = PyList_GET_SIZE(stack);
        for (Py_ssize_t level = 0; level < stackSize; ++level) {
            PyObject *frame = PyList_GET_ITEM(stack, stackSize - 1 - level);
            char *fileName = NULL;
            long line = 0;
            char *funcName = NULL;
            char *lineText = NULL;
            int ok = PyArg_ParseTuple(frame,"slss:bsp.getScriptPos", &fileName,&line,&funcName,&lineText);
            if (!ok) {
                PyErr_SetString(PyExc_RuntimeError, "error occured when parsing stack frames in bsp.getScriptPos");
                scriptPos_ = "";
                return;
            }
            ss << "#" << level << ": FILE:" << fileName << ", LINE:" << line << ", FUNCTION:" << funcName << ", CODE:" <<std::endl
                << ">>> " << lineText << std::endl;
        }
    }
    //ss << runtime_->getMyProcessID() << ": ------  call stack end  ------ " << std::endl;
    scriptPos_ =  ss.str();
}

void bsp_typeError(std::string strErr) {
    std::cerr << runtime_->getMyProcessID() << ": TypeErr : " << strErr << std::endl
        << scriptPos_;
}

void bsp_runtimeError(std::string strErr) {
    std::cerr << runtime_->getMyProcessID() << ": RuntimeErr : " << strErr << std::endl
        << scriptPos_;
}

PyObject *PyRetVal(PyObject *obj) {
    Py_XINCREF(obj);
    return obj;
}

class PyRef {
    private:
        PyObject *_obj;
    public:
        PyRef(PyObject *obj) {
            Py_XINCREF(_obj = obj);
        }
        ~PyRef() {
            Py_XDECREF(_obj);
        }
};

extern "C" {
    PyObject *localArrayAsNumpy(LocalArray *localArray) {
        PyObject *output = NULL;
        try {
            ArrayShape::ElementType elemType = localArray->getElementType();
            int typeNum = NPY_DOUBLE;
            switch (elemType) {
                case ArrayShape::INT8:
                    typeNum = NPY_INT8;
                    break;
                case ArrayShape::INT16:
                    typeNum = NPY_INT16;
                    break;
                case ArrayShape::INT32:
                    typeNum = NPY_INT32;
                    break;
                case ArrayShape::INT64:
                    typeNum = NPY_INT64;
                    break;
                case ArrayShape::UINT8:
                    typeNum = NPY_UINT8;
                    break;
                case ArrayShape::UINT16:
                    typeNum = NPY_UINT16;
                    break;
                case ArrayShape::UINT32:
                    typeNum = NPY_UINT32;
                    break;
                case ArrayShape::UINT64:
                    typeNum = NPY_UINT64;
                    break;
                case ArrayShape::FLOAT:
                    typeNum = NPY_FLOAT;
                    break;
                case ArrayShape::DOUBLE:
                    typeNum = NPY_DOUBLE;
                    break;
                case ArrayShape::CFLOAT:
                    typeNum = NPY_CFLOAT;
                    break;
                case ArrayShape::CDOUBLE:
                    typeNum = NPY_CDOUBLE;
                    break;
                default:
                    break;
            }
            int nDims = (int) localArray->getNumberOfDimensions();
            npy_intp dimSize[7];
            for (int iDim = 0; iDim < nDims; ++iDim) {
                dimSize[iDim] = (npy_intp) localArray->getElementCount(iDim);
            }
            output = PyArray_SimpleNewFromData(nDims,dimSize,typeNum, localArray->getData());
            if (!PyArray_Check(output)) {
                bsp_runtimeError("failed to call PyArray_SimpleNewWithData() in bsp.toNumpy(arrayPath)");
                Py_XDECREF(output);
                Py_RETURN_NONE;
            }
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_XDECREF(output);
            Py_RETURN_NONE;
        }
        return output;
    }

    PyObject *arrayAsNumpy(char *path) {
        try {
            NamedObject *nobj = runtime_->getObject(std::string(path));
            LocalArray *localArray = nobj->_localArray();
            return localArrayAsNumpy(localArray);
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_NONE;
        }
    }

    static PyObject *Dim_init(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *objThis = NULL;
        unsigned long dimSize = 0;
        int ok = PyArg_ParseTuple(args, "Ok:Dim.init", &objThis, &dimSize);
        if (!ok) {
            bsp_typeError("invalid params for Dim.init");
            Py_RETURN_NONE;
        }
        try {
            PyObject *objAttr = Py_BuildValue("k",dimSize);
            PyObject_SetAttrString(objThis, "dimSize", objAttr);
            Py_CLEAR(objAttr);
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
        }
        Py_RETURN_NONE;
    }

    static PyObject *Dim_del(PyObject *self, PyObject *args) {
        PyObject *objThis = NULL;
        int ok = PyArg_ParseTuple(args, "O:Dim.del", &objThis);
        if (!ok) {
            bsp_typeError("invalid params for Dim.del");
            Py_RETURN_NONE;
        }
        try {
            PyObject_DelAttrString(objThis, "dimSize");
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
        }
        Py_RETURN_NONE;
    }

    typedef struct{ LocalArray *_upper, *_lower, *_step; } DimItem;
    PyObject *dimToObject(unsigned long dimSize, PyObject *objIndex) {
        LocalArray *lower = NULL, *upper = NULL, *stepSize = NULL;
        Py_ssize_t myStart = 0, myStop = 0, myStep = 0;
        uint64_t n = 1;
        if (PyTuple_Check(objIndex)) {
            n = (uint64_t)PyTuple_GET_SIZE(objIndex);
            lower = new LocalArray("",ArrayShape::UINT64,ArrayShape::elementSize(ArrayShape::UINT64),
                    1, &n);
            upper = new LocalArray("",ArrayShape::UINT64,ArrayShape::elementSize(ArrayShape::UINT64),
                    1, &n);
            stepSize = new LocalArray("",ArrayShape::INT32,ArrayShape::elementSize(ArrayShape::INT32),
                    1, &n);
            uint64_t *start = (uint64_t *)lower->getData();
            uint64_t *stop = (uint64_t *)upper->getData();
            int32_t *step = (int32_t *)stepSize->getData();
            for (Py_ssize_t i = 0; i < n; ++ i) {
                PyObject *objComponent = PyTuple_GetItem(objIndex, i);
                if (!PySlice_Check(objComponent)) {
                    bsp_typeError("only slice indices are supported");
                    Py_RETURN_NONE;
                }
                PySlice_GetIndices(objComponent, dimSize, &myStart, &myStop, &myStep);
                start[i] = myStart;
                stop[i] = myStop;
                step[i] = myStep;
            }
        } else {
            if (!PySlice_Check(objIndex)) {
                bsp_typeError("only slice indices are supported");
                Py_RETURN_NONE;
            }
            lower = new LocalArray("",ArrayShape::UINT64,ArrayShape::elementSize(ArrayShape::UINT64),
                    1, &n);
            upper = new LocalArray("",ArrayShape::UINT64,ArrayShape::elementSize(ArrayShape::UINT64),
                    1, &n);
            stepSize = new LocalArray("",ArrayShape::INT32,ArrayShape::elementSize(ArrayShape::INT32),
                    1, &n);
            uint64_t *start = (uint64_t *)lower->getData();
            uint64_t *stop = (uint64_t *)upper->getData();
            int32_t *step = (int32_t *)stepSize->getData();
            PySlice_GetIndices(objIndex, dimSize, &myStart, &myStop, &myStep);
            start[0] = myStart;
            stop[0] = myStop;
            step[0] = myStep;
        }

        uint64_t *start = (uint64_t *)lower->getData();
        uint64_t *stop = (uint64_t *)upper->getData();
        for (unsigned int i = 0; i < (unsigned int)n; ++ i) {
            if (start[i] >= dimSize || stop[i] > dimSize) {
                bsp_typeError("out-of-range index");
                Py_RETURN_NONE;
            }
        }

        DimItem *item = new DimItem;
        item->_lower = lower;
        item->_upper = upper;
        item->_step = stepSize;
        return PyRetVal(PyLong_FromSize_t((size_t)item));
    }

    static PyObject *Dim_getItem(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *objSelf = NULL, *objIndex = NULL;
        int ok = PyArg_ParseTuple(args, "OO", &objSelf, &objIndex);
        if (!ok) {
            bsp_typeError("invalid index");
            Py_RETURN_NONE;
        }
        PyObject *objDimSize = PyObject_GetAttrString(objSelf, "dimSize");
        if (NULL == objDimSize) {
            bsp_runtimeError("dim object has no dimSize");
            Py_RETURN_NONE;
        }
        unsigned long dimSize = 0;
        ok = PyArg_Parse(objDimSize, "k", &dimSize);
        if (!ok) {
            bsp_runtimeError("invalid dimSize");
            Py_RETURN_NONE;
        }
        return dimToObject(dimSize, objIndex);
    }

    static PyMethodDef DimMethods[] = {
        {"__init__", Dim_init, METH_VARARGS,  "init"},
        {"__del__", Dim_del, METH_VARARGS,  "delete"},
        {"__getitem__", Dim_getItem, METH_VARARGS, "get item"},
        {NULL, NULL, 0, NULL}
    };

    PyObject *bsp_getDimType() {
        PyObject *moduleDict = PyModule_GetDict(module_);
        return PyDict_GetItemString(moduleDict, "Dim");
    }

    bool allSlices(PyObject *objIndex) {
        if (PyTuple_Check(objIndex)) {
            Py_ssize_t nDims = PyTuple_GET_SIZE(objIndex);
            for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
                PyObject *objDim = PyTuple_GetItem(objIndex, iDim);
                if (!PySlice_Check(objDim)) {
                    return false;
                }
            }
        } else {
            if (!PySlice_Check(objIndex)) {
                return false;
            }
        }
        return true;
    }

    IndexSet *allSlicesToIndexSet(PyObject *objIndex, unsigned int nDims, Py_ssize_t sliceLength[], std::string &rangeID,
            uint64_t elemCountAlongDim[]) {
        Py_ssize_t start[7], stop[7], step[7];
        if (PyTuple_Check(objIndex)) {
            if (PyTuple_GET_SIZE(objIndex) != nDims) {
                bsp_typeError("unmatched nDims");
                return NULL;
            }
            for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
                PyObject *objDim = PyTuple_GetItem(objIndex, iDim);
                if (!PySlice_Check(objDim)) {
                    bsp_typeError("only slice indices are supported");
                    return NULL;
                }
                PySlice_GetIndices(objDim, sliceLength[iDim], start + iDim, stop + iDim, step + iDim);
            }
        } else {
            if (1 != nDims) {
                bsp_typeError("unmatched nDims");
                return NULL;
            }
            if (!PySlice_Check(objIndex)) {
                bsp_typeError("only slice indices are supported");
                return NULL;
            }
            PySlice_GetIndices(objIndex, sliceLength[0], start, stop, step);
        }

        uint64_t myStart[7], myStop[7];
        int32_t myStep[7];
        std::stringstream ss;
        for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
            myStart[iDim] = start[iDim];
            myStop[iDim] = stop[iDim];
            myStep[iDim] = step[iDim];
            int64_t dimSize = ((int64_t)stop[iDim] - (int64_t)start[iDim] - 1) / (int32_t)step[iDim] + 1;
            if (dimSize <= 0) {
                bsp_typeError("invalid slice");
                return NULL;
            }
            elemCountAlongDim[iDim] = dimSize;
            ss << start[iDim] << ":" << stop[iDim] << ":" << step[iDim] << ";";
        }
        rangeID = ss.str();

        for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
            if (myStart[iDim] >= sliceLength[iDim] || myStop[iDim] > sliceLength[iDim]) {
                bsp_typeError("out-of-range index");
                return NULL;
            }
        }
        return new IndexSetRegionTensor(nDims, myStart, myStop, myStep);
    }

    IndexSet *objectToIndexSet(PyObject *objIndex, unsigned int nDims, Py_ssize_t sliceLength[], std::string &rangeID,
            uint64_t elemCountAlongDim[]) {
        if (allSlices(objIndex))
            return allSlicesToIndexSet(objIndex, nDims, sliceLength, rangeID, elemCountAlongDim);
        LocalArray *lower[7], *upper[7], *step[7];
        if (PyTuple_Check(objIndex)) {
            if (PyTuple_GET_SIZE(objIndex) != nDims) {
                bsp_typeError("unmatched nDims");
                return NULL;
            }
            for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
                PyObject *objDim = PyTuple_GetItem(objIndex, iDim);
                bool needDel = false;
                if (PySlice_Check(objDim)) {
                    objDim = dimToObject(sliceLength[iDim], objDim);
                    needDel = true;
                }
                DimItem *item = (DimItem *)PyLong_AsSize_t(objDim);
                lower[iDim] = item->_lower;
                upper[iDim] = item->_upper;
                step[iDim] = item->_step;
                if (needDel)
                    Py_CLEAR(objDim);
            }
        } else {
            if (1 != nDims) {
                bsp_typeError("unmatched nDims");
                return NULL;
            }
            PyObject *objDim = objIndex;
            bool needDel = false;
            if (PySlice_Check(objIndex)) {
                objDim = dimToObject(sliceLength[0], objDim);
                needDel = true;
            }
            DimItem *item = (DimItem *)PyLong_AsSize_t(objDim);
            lower[0] = item->_lower;
            upper[0] = item->_upper;
            step[0] = item->_step;
            if (needDel)
                Py_CLEAR(objDim);
        }

        std::stringstream ss;
        for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
            uint64_t *start = (uint64_t *)lower[iDim]->getData();
            uint64_t *stop = (uint64_t *)upper[iDim]->getData();
            int32_t *myStep = (int32_t *)step[iDim]->getData();
            uint64_t n = lower[iDim]->getElementCount(LocalArray::ALL_DIMS);
            elemCountAlongDim[iDim] = 0;
            for (uint64_t i = 0; i < n; ++ i) {
                int64_t dimSize = ((int64_t)stop[i] - (int64_t)start[i] - 1) / myStep[i] + 1;
                if (dimSize <= 0) {
                    bsp_typeError("invalid slice");
                    return NULL;
                }
                elemCountAlongDim[iDim] += dimSize;
                ss << start[i] << ":" << stop[i] << ":" << step[i] << ";";
            }
            ss << ";";
        }
        rangeID = ss.str();

        return new IndexSetRegionTensor(nDims, lower, upper, step);
    }

    static PyObject *Future_init(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        unsigned long procID = 0;
        char *varName = NULL;
        char *op = NULL;
        PyObject *objSelf = NULL;
        int ok = PyArg_ParseTuple(args, "Oks|s:Future.init", &objSelf, &procID, &varName, &op);
        if (!ok) {
            bsp_typeError("invalid params for Future.init");
            Py_RETURN_NONE;
        }

        if (procID >= runtime_->getNumberOfProcesses()) {
            bsp_typeError("invalid procID");
            Py_RETURN_NONE;
        }

        NamedObject *nobjServer = runtime_->getObject(varName);
        if (ARRAY != nobjServer->getType()) {
            bsp_typeError("invalid varName");
            Py_RETURN_NONE;
        }
        PyObject *objDimType = bsp_getDimType();
        if (nobjServer->isGlobal()) {
            GlobalArray *globalArray = nobjServer->_globalArray();
            unsigned int nDims = globalArray->getNumberOfDimensions();
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                uint64_t dimSize = globalArray->getElementCount(iDim);
                PyObject *param = Py_BuildValue("(k)",dimSize);
                PyObject *dimObject = PyObject_CallObject(objDimType,param);
                Py_CLEAR(param);
                std::stringstream ss;
                ss << "dim" << iDim;
                PyObject_SetAttrString(objSelf,ss.str().c_str(),dimObject);
            }
        } else {
            ArrayRegistration *registration = nobjServer->_localArray()->getRegistration();
            if (NULL == registration) {
                bsp_typeError("unregistered server array");
                Py_RETURN_NONE;
            }
            ArrayShape *shape = registration->getArrayShape(procID);
            unsigned int nDims = shape->getNumberOfDimensions();
            for (unsigned int iDim = 0; iDim < nDims; ++ iDim) {
                uint64_t dimSize = shape->getElementCount(iDim);
                PyObject *param = Py_BuildValue("(k)",dimSize);
                PyObject *dimObject = PyObject_CallObject(objDimType,param);
                Py_CLEAR(param);
                std::stringstream ss;
                ss << "dim" << iDim;
                PyObject_SetAttrString(objSelf,ss.str().c_str(),dimObject);
            }
        }

        uint16_t opID = LocalArray::OPID_ASSIGN;
        if (op) {
            if (0 == strcmp(op, "="))
                opID = LocalArray::OPID_ASSIGN;
            else if (0 == strcmp(op, "+") || 0 == strcmp(op, "+="))
                opID = LocalArray::OPID_ADD;
            else if (0 == strcmp(op, "*") || 0 == strcmp(op, "*="))
                opID = LocalArray::OPID_MUL;
            else if (0 == strcmp(op, "&") || 0 == strcmp(op, "&="))
                opID = LocalArray::OPID_AND;
            else if (0 == strcmp(op, "|") || 0 == strcmp(op, "|="))
                opID = LocalArray::OPID_OR;
            else if (0 == strcmp(op, "^") || 0 == strcmp(op, "^="))
                opID = LocalArray::OPID_XOR;
            else if (0 == strcmp(op, "min") || 0 == strcmp(op, "min="))
                opID = LocalArray::OPID_MIN;
            else if (0 == strcmp(op, "max") || 0 == strcmp(op, "max="))
                opID = LocalArray::OPID_MAX;
            else {
                bsp_typeError("unrecognized op");
                Py_RETURN_NONE;
            }
        }

        std::stringstream ss;
        ss << procID << ":" << opID << ":" << varName;
        std::string futureID = ss.str();
        if (futures_.find(futureID) != futures_.end()) {
            bsp_runtimeError(futureID + " already exists in futures_");
            Py_RETURN_NONE;
        }
        futures_[futureID] = objSelf;
        futureIDs_[objSelf] = futureID;
        Py_RETURN_NONE;
    }

    void Future_clearUpdates(PyObject *self) {
        std::string futureID = futureIDs_[self];
        std::map<std::string, LocalArray *> &myUpdates = updates_[futureID];
        for (std::map<std::string, LocalArray *>::iterator iter = myUpdates.begin();
                iter != myUpdates.end(); ++ iter) {
            delete iter->second;
        }
        myUpdates.clear();
    }

    void Future_clearRequests(PyObject *self) {
        std::string futureID = futureIDs_[self];
        std::map<std::string, LocalArray *> &myRequests = requests_[futureID];
        for (std::map<std::string, LocalArray *>::iterator iter = myRequests.begin();
                iter != myRequests.end(); ++ iter) {
            delete iter->second;
        }
        myRequests.clear();
    }

    static PyObject *Future_del(PyObject *self, PyObject *args) {
        PyObject *objSelf = NULL;
        int ok = PyArg_ParseTuple(args, "O:Future.__del__",&objSelf);
        if (!ok) {
            bsp_typeError("invalid params for Future.del");
            Py_RETURN_NONE;
        }
        Future_clearUpdates(objSelf);
        Future_clearRequests(objSelf);
        std::string futureID = futureIDs_[objSelf];
        futures_.erase(futureID);
        futureIDs_.erase(objSelf);

        for (unsigned int iDim = 0; iDim < 7; ++ iDim) {
            std::stringstream ss;
            ss << "dim" << iDim;
            std::string s = ss.str();
            if (PyObject_HasAttrString(objSelf, s.c_str()))
                PyObject_DelAttrString(objSelf, s.c_str());
            else
                break;
        }
        Py_RETURN_NONE;
    }

    static PyObject *Future_getItem(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *objSelf = NULL, *objIndex = NULL;
        int ok = PyArg_ParseTuple(args, "OO", &objSelf, &objIndex);
        if (!ok) {
            bsp_typeError("invalid index");
            Py_RETURN_NONE;
        }

        std::string futureID = futureIDs_[objSelf];

        unsigned long procID = 0;
        size_t posColon = futureID.find(':', 0); 
        assert(posColon != (size_t) (-1));
        assert (1 == sscanf(futureID.substr(0, posColon).c_str(), "%lu", &procID));

        unsigned int opIDL = LocalArray::OPID_ASSIGN;
        std::string remain = futureID.substr(posColon + 1);
        posColon = remain.find(':', 0);
        assert(posColon != (size_t) (-1));
        assert (1 == sscanf(remain.substr(0, posColon).c_str(), "%u", &opIDL));
        uint16_t opID = (uint16_t) opIDL;

        if (opID != LocalArray::OPID_ASSIGN) {
            bsp_typeError("unsupported op for getItem");
            Py_RETURN_NONE;
        }

        std::string varName = remain.substr(posColon + 1);

        NamedObject *nobjServer = runtime_->getObject(varName);
        if (ARRAY != nobjServer->getType()) {
            bsp_typeError("invalid varName");
            Py_RETURN_NONE;
        }

        Py_ssize_t nDims = 1;
        if (PyTuple_Check(objIndex)) {
            nDims = PyTuple_GET_SIZE(objIndex);
            assert(nDims <= 7 && nDims > 0);
        }

        Py_ssize_t sliceLength[7];
        if (nobjServer->isGlobal()) {
            GlobalArray *globalArray = nobjServer->_globalArray();
            if (globalArray->getNumberOfDimensions() != nDims) {
                bsp_typeError("invalid nDims of index");
                Py_RETURN_NONE;
            }
            for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
                uint64_t dimSize = globalArray->getElementCount(iDim);
                sliceLength[iDim] = dimSize;
            }
        } else {
            ArrayRegistration *registration = nobjServer->_localArray()->getRegistration();
            assert(NULL != registration);
            ArrayShape *shape = registration->getArrayShape(procID);
            if (shape->getNumberOfDimensions() != nDims) {
                bsp_typeError("invalid nDims of index");
                Py_RETURN_NONE;
            }
            if (shape->getNumberOfDimensions() != nDims) {
                bsp_typeError("invalid nDims of index");
                Py_RETURN_NONE;
            }
            for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
                uint64_t dimSize = shape->getElementCount(iDim);
                sliceLength[iDim] = dimSize;
            }
        }

        std::string rangeID;
        uint64_t elemCountAlongDim[7];
        IndexSet *indexSet = objectToIndexSet(objIndex, nDims, sliceLength, rangeID, elemCountAlongDim);

        ArrayShape::ElementType elementType;
        uint64_t numberOfBytesPerElement;
        if (nobjServer->isGlobal()) {
            GlobalArray *globalArray = nobjServer->_globalArray();
            elementType = globalArray->getElementType();
            numberOfBytesPerElement = globalArray->getNumberOfBytesPerElement();
        } else {
            ArrayRegistration *registration = nobjServer->_localArray()->getRegistration();
            assert(NULL != registration);
            ArrayShape *shape = registration->getArrayShape(procID);
            elementType = shape->getElementType();
            numberOfBytesPerElement = shape->getNumberOfBytesPerElement();
        }

        std::map<std::string, LocalArray *> &myRequests = requests_[futureID];
        std::map<std::string, LocalArray *>::iterator iter = myRequests.find(rangeID);
        LocalArray *clientArray = NULL;
        if (iter == myRequests.end()) {
            clientArray = new LocalArray("", elementType, numberOfBytesPerElement, nDims, elemCountAlongDim);
            myRequests[rangeID] = clientArray;
        } else {
            clientArray = iter->second;
        }

        if (nobjServer->isGlobal()) {
            runtime_->requestFrom(*nobjServer->_globalArray(), *indexSet, *clientArray, scriptPos_);
        } else {
            runtime_->requestFrom(*nobjServer->_localArray(), procID, *indexSet, *clientArray, scriptPos_);
        }
        delete indexSet;
        return localArrayAsNumpy(clientArray);
    }

    static PyObject *Future_setItem(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *objSelf = NULL, *objIndex = NULL, *objValue = NULL;
        int ok = PyArg_ParseTuple(args, "OOO", &objSelf, &objIndex, &objValue);
        if (!ok) {
            bsp_typeError("invalid params");
            Py_RETURN_NONE;
        }
        bool requested = false;

        std::string futureID = futureIDs_[objSelf];

        unsigned long procID = 0;
        size_t posColon = futureID.find(':', 0); 
        assert(posColon != (size_t) (-1));
        assert (1 == sscanf(futureID.substr(0, posColon).c_str(), "%lu", &procID));

        unsigned int opIDL = LocalArray::OPID_ASSIGN;
        std::string remain = futureID.substr(posColon + 1);
        posColon = remain.find(':', 0);
        assert(posColon != (size_t) (-1));
        assert (1 == sscanf(remain.substr(0, posColon).c_str(), "%u", &opIDL));
        uint16_t opID = (uint16_t) opIDL;

        std::string varName = remain.substr(posColon + 1);

        NamedObject *nobjServer = runtime_->getObject(varName);
        if (ARRAY != nobjServer->getType()) {
            bsp_typeError("invalid varName");
            Py_RETURN_NONE;
        }

        Py_ssize_t nDims = 1;
        if (PyTuple_Check(objIndex)) {
            nDims = PyTuple_GET_SIZE(objIndex);
            assert(nDims <= 7 && nDims > 0);
        }

        Py_ssize_t sliceLength[7];
        if (nobjServer->isGlobal()) {
            GlobalArray *globalArray = nobjServer->_globalArray();
            if (globalArray->getNumberOfDimensions() != nDims) {
                bsp_typeError("invalid nDims of index");
                Py_RETURN_NONE;
            }
            for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
                uint64_t dimSize = globalArray->getElementCount(iDim);
                sliceLength[iDim] = dimSize;
            }
        } else {
            ArrayRegistration *registration = nobjServer->_localArray()->getRegistration();
            assert(NULL != registration);
            ArrayShape *shape = registration->getArrayShape(procID);
            if (shape->getNumberOfDimensions() != nDims) {
                bsp_typeError("invalid nDims of index");
                Py_RETURN_NONE;
            }
            if (shape->getNumberOfDimensions() != nDims) {
                bsp_typeError("invalid nDims of index");
                Py_RETURN_NONE;
            }
            for (Py_ssize_t iDim = 0; iDim < nDims; ++ iDim) {
                uint64_t dimSize = shape->getElementCount(iDim);
                sliceLength[iDim] = dimSize;
            }
        }

        std::string rangeID;
        uint64_t elemCountAlongDim[7];
        IndexSet *indexSet = objectToIndexSet(objIndex, nDims, sliceLength, rangeID, elemCountAlongDim);

        ArrayShape::ElementType elementType;
        uint64_t numberOfBytesPerElement;
        if (nobjServer->isGlobal()) {
            GlobalArray *globalArray = nobjServer->_globalArray();
            elementType = globalArray->getElementType();
            numberOfBytesPerElement = globalArray->getNumberOfBytesPerElement();
        } else {
            ArrayRegistration *registration = nobjServer->_localArray()->getRegistration();
            assert(NULL != registration);
            ArrayShape *shape = registration->getArrayShape(procID);
            elementType = shape->getElementType();
            numberOfBytesPerElement = shape->getNumberOfBytesPerElement();
        }

        std::map<std::string, LocalArray *> &myUpdates = updates_[futureID];
        std::map<std::string, LocalArray *>::iterator iter = myUpdates.find(rangeID);
        LocalArray *clientArray = NULL;
        if (iter == myUpdates.end()) {
            clientArray = new LocalArray("", elementType, numberOfBytesPerElement, nDims, elemCountAlongDim);
            myUpdates[rangeID] = clientArray;
        } else {
            requested = true;
            clientArray = iter->second;
        }

        try {
            PyArrayObject *updArray = (PyArrayObject *)localArrayAsNumpy(clientArray);
            if (PyArray_Check(objValue)) {
                PyArrayObject *valueArray = (PyArrayObject *)objValue;
                if (0 > PyArray_CopyInto(updArray, valueArray)) {
                    bsp_runtimeError("invalid value");
                    Py_RETURN_NONE;
                }
            } else {
                PyArray_FillWithScalar(updArray, objValue);
            }
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_NONE;
        }

        if (!requested) {
            if (nobjServer->isGlobal()) {
                runtime_->requestTo(*nobjServer->_globalArray(), *indexSet, *clientArray, opID, scriptPos_);
            } else {
                runtime_->requestTo(*nobjServer->_localArray(), procID, *indexSet, *clientArray, opID, scriptPos_);
            }
        }
        delete indexSet;

        Py_RETURN_NONE;
    }

    static PyMethodDef FutureMethods[] = {
        {"__init__", Future_init, METH_VARARGS,  "init future object with procID, varName and op"},
        {"__del__", Future_del, METH_VARARGS,  "delete future object"},
        {"__getitem__", Future_getItem, METH_VARARGS, "get item"},
        {"__setitem__", Future_setItem, METH_VARARGS, "set item"},
        {NULL, NULL, 0, NULL}
    };


    void finiBSP() {
        Py_XDECREF(pickle_dumps_);
        Py_XDECREF(pickle_loads_);
        Py_XDECREF(ctypes_addressof_);
        Py_XDECREF(pickle_);
        Py_XDECREF(ctypes_);
        Py_XDECREF(traceback_extractStack_);
        Py_XDECREF(traceback_);
        for (unsigned i = 0; i < nProcs_; ++i) {
            Py_XDECREF(fromProc_[i]);
        }
        delete[] fromProc_;
        pickle_ = NULL;
        pickle_dumps_ = NULL;
        pickle_loads_ = NULL;
        ctypes_ = NULL;
        ctypes_addressof_ = NULL;
        fromProc_ = NULL;
        nProcs_ = 0;
        delete runtime_;
        try {
            Py_Finalize();
        } catch (...) {
        }
    }

    // myProcID = bsp.myProcID()
    static PyObject *bsp_myProcID(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        int ok = PyArg_ParseTuple(args,":bsp.myProcID");
        if (!ok) {
            bsp_typeError("bsp.myProcID requires no arguments");
            Py_RETURN_NONE;
        }
        uint64_t result = runtime_->getMyProcessID();
        return PyRetVal(Py_BuildValue("l",(long)result));
    }

    // procCount = bsp.procCount()
    static PyObject *bsp_procCount(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        int ok = PyArg_ParseTuple(args,":bsp.procCount");
        if (!ok) {
            bsp_typeError("bsp.procCount requires no arguments");
            Py_RETURN_NONE;
        }
        return PyRetVal(Py_BuildValue("l",(long)nProcs_));
    }

    // importedFromProcID = bsp.fromProc(procID)
    static PyObject *bsp_fromProc(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        long procID;
        int ok = PyArg_ParseTuple(args,"l:bsp.fromProc", &procID);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.fromProc(procID)");
            Py_RETURN_NONE;
        }
        if (NULL == fromProc_) {
            return PyRetVal(Py_BuildValue("{}"));
        } else {
            return PyRetVal(fromProc_[procID]);
        }
    }

    // OK = bsp.fromObject(object,arrayPath)
    static PyObject *bsp_fromObject(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *input = NULL;
        char *path = NULL;
        int ok = PyArg_ParseTuple(args,"Os:bsp.fromObject",&input,&path);
        PyRef refInput(input);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.fromObject(object,arrayPath)");
            return PyRetVal(Py_BuildValue("O",Py_False));
        }
        PyObject *param = Py_BuildValue("(O)",input);
        PyObject *strObj = PyObject_CallObject(pickle_dumps_,param);
        Py_XDECREF(param);
        if (NULL == strObj) {
            bsp_runtimeError("failed to call pickle.dumps()");
            return PyRetVal(Py_BuildValue("O",Py_False));
        }
        Py_buffer buf;
        ok = PyArg_Parse(strObj,"s*:bsp.fromObject.getStr",&buf);
        Py_XDECREF(strObj);
        if (!ok) {
            bsp_runtimeError("failed to get string from output of pickle.dumps()");
            Py_RETURN_FALSE;
        }
        try {
            runtime_->fromBuffer((const char *)buf.buf, 'u', 1, &buf.len, &buf.itemsize, std::string(path));
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    }

    LocalArray *bsp_convertNumpy(PyArrayObject *numpyArray, const char *path) {
        int nDims = PyArray_NDIM(numpyArray);
        npy_intp *dimSize = PyArray_DIMS(numpyArray);
        npy_intp *strides = PyArray_STRIDES(numpyArray);
        char *data = PyArray_BYTES(numpyArray);
        char kind = 'i';
        if (PyArray_ISINTEGER(numpyArray)) {
            if (PyArray_ISUNSIGNED(numpyArray))
                kind = 'u';
            else
                kind = 'i';
        } else if (PyArray_ISCOMPLEX(numpyArray)) {
            kind = 'c';
        } else if (PyArray_ISFLOAT(numpyArray)) {
            kind = 'f';
        }
        try {
            return runtime_->fromBuffer(data, kind, nDims, dimSize, strides, path);
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            return NULL;
        }
    }

    LocalArray *bsp_convertObject(PyObject *object, const char *path) {
        if (PyArray_Check(object))
            return bsp_convertNumpy((PyArrayObject *)object, path);
        else {
            PyArrayObject *numpyArray = (PyArrayObject *)PyArray_FROM_O(object);
            LocalArray *result = bsp_convertNumpy(numpyArray, path);
            Py_XDECREF(numpyArray);
            return result;
        }
    }

    // OK = bsp.fromNumpy(numpyArray,arrayPath)
    static PyObject *bsp_fromNumpy(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *input = NULL;
        char *path = NULL;
        int ok = PyArg_ParseTuple(args,"Os:bsp.fromNumpy",&input,&path);
        PyRef refInput(input);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.fromNumpy(NumpyArray,arrayPath)");
            Py_RETURN_FALSE;
        }
        if (!PyArray_Check(input)) {
            bsp_typeError("invalid arguments for bsp.fromNumpy(NumpyArray,arrayPath)");
            Py_RETURN_FALSE;
        }
        PyArrayObject *numpyArray = (PyArrayObject *)input;
        LocalArray *localArray = bsp_convertNumpy(numpyArray,path);
        if (localArray)
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }

    // object = bsp.toObject(arrayPath)
    static PyObject *bsp_toObject(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *output = NULL;
        char *path = NULL;
        int ok = PyArg_ParseTuple(args,"s:bsp.toObject",&path);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.toObject(arrayPath)");
            Py_RETURN_NONE;
        }
        try {
            NamedObject *nobj = runtime_->getObject(std::string(path));
            LocalArray *localArray = nobj->_localArray();
            PyObject *bytes = PyBytes_FromStringAndSize(localArray->getData(), localArray->getByteCount());
            assert(bytes != NULL);
            PyObject *objStr = Py_BuildValue("(O)",bytes);
            output = PyObject_CallObject(pickle_loads_,objStr);
            Py_XDECREF(objStr);
            Py_XDECREF(bytes);
            if (NULL == output) {
                bsp_runtimeError("failed to call pickle.loads()");
                Py_RETURN_NONE;
            }
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_NONE;
        }
        return output;
    }

    // numpyArray = bsp.toNumpy(arrayPath)
    static PyObject *bsp_toNumpy(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *output = NULL;
        char *path = NULL;
        int ok = PyArg_ParseTuple(args,"s:bsp.toNumpy",&path);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.toNumpy(arrayPath)");
            Py_RETURN_NONE;
        }
        try {
            NamedObject *nobj = runtime_->getObject(std::string(path));
            LocalArray *localArray = nobj->_localArray();
            ArrayShape::ElementType elemType = localArray->getElementType();
            int typeNum = NPY_DOUBLE;
            switch (elemType) {
                case ArrayShape::INT8:
                    typeNum = NPY_INT8;
                    break;
                case ArrayShape::INT16:
                    typeNum = NPY_INT16;
                    break;
                case ArrayShape::INT32:
                    typeNum = NPY_INT32;
                    break;
                case ArrayShape::INT64:
                    typeNum = NPY_INT64;
                    break;
                case ArrayShape::UINT8:
                    typeNum = NPY_UINT8;
                    break;
                case ArrayShape::UINT16:
                    typeNum = NPY_UINT16;
                    break;
                case ArrayShape::UINT32:
                    typeNum = NPY_UINT32;
                    break;
                case ArrayShape::UINT64:
                    typeNum = NPY_UINT64;
                    break;
                case ArrayShape::FLOAT:
                    typeNum = NPY_FLOAT;
                    break;
                case ArrayShape::DOUBLE:
                    typeNum = NPY_DOUBLE;
                    break;
                case ArrayShape::CFLOAT:
                    typeNum = NPY_CFLOAT;
                    break;
                case ArrayShape::CDOUBLE:
                    typeNum = NPY_CDOUBLE;
                    break;
                default:
                    break;
            }
            int nDims = (int) localArray->getNumberOfDimensions();
            npy_intp dimSize[7];
            for (int iDim = 0; iDim < nDims; ++iDim) {
                dimSize[iDim] = (npy_intp) localArray->getElementCount(iDim);
            }
            output = PyArray_SimpleNew(nDims,dimSize,typeNum);
            if (!PyArray_Check(output)) {
                bsp_runtimeError("failed to call PyArray_SimpleNew() in bsp.toNumpy(arrayPath)");
                Py_XDECREF(output);
                Py_RETURN_NONE;
            }
            char *numpyData = (char *)PyArray_DATA((PyArrayObject *)output);
            memcpy(numpyData, localArray->getData(), localArray->getByteCount());
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_XDECREF(output);
            Py_RETURN_NONE;
        }
        return output;
    }

    // numpyArray = bsp.asNumpy(arrayPath)
    static PyObject *bsp_asNumpy(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        char *path = NULL;
        int ok = PyArg_ParseTuple(args,"s:bsp.asNumpy",&path);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.asNumpy(arrayPath)");
            Py_RETURN_NONE;
        }
        return arrayAsNumpy(path);
    }

    // OK = bsp.array(arrayPath,dtype,arrayShape)
    static PyObject *bsp_array(PyObject *self, PyObject *args, PyObject *kwargs) {
        bsp_getScriptPos();
        static const char * kwlist[] = {"path", "shape", "dtype", "fill", NULL};
        char *arrayPath = NULL;
        char *dtype = NULL;
        PyObject *arrayShape = NULL;
        uint64_t dimSize[7] = {0,0,0,0,0,0,0};
        double fill = sqrt(-1);
        int ok = PyArg_ParseTupleAndKeywords(args,kwargs,"sO|sd:bsp.array",(char **)kwlist,&arrayPath,&arrayShape,&dtype,&fill);

        PyRef refArrayShape(arrayShape);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.array(arrayPath,arrayShape)");
            Py_RETURN_FALSE;
        }
        if (PyTuple_Check(arrayShape)) {
            ok = PyArg_ParseTuple(arrayShape,"k|kkkkkk:bsp.array.extractArrayShape",
                    (unsigned long *)(dimSize + 0),
                    (unsigned long *)(dimSize + 1),
                    (unsigned long *)(dimSize + 2),
                    (unsigned long *)(dimSize + 3),
                    (unsigned long *)(dimSize + 4),
                    (unsigned long *)(dimSize + 5),
                    (unsigned long *)(dimSize + 6));
            if (!ok || dimSize[0] == 0) {
                bsp_typeError("invalid array shape for bsp.array(arrayPath,arrayShape)");
                Py_RETURN_FALSE;
            }
        } else if (PyList_Check(arrayShape)) {
            Py_ssize_t nItems = PyList_GET_SIZE(arrayShape);
            for (Py_ssize_t iItem = 0; iItem < nItems; ++iItem) {
                PyObject *item = PyList_GET_ITEM(arrayShape,iItem);
                PyRef refItem(item);
                unsigned long sizeOfThisDim = 0;
                ok = PyArg_Parse(item, "k", &sizeOfThisDim);
                if (!ok)
                    break;
                dimSize[iItem] = sizeOfThisDim;
            }
            if (!ok || dimSize[0] == 0) {
                bsp_typeError("invalid array shape for bsp.array(arrayPath,arrayShape)");
                Py_RETURN_FALSE;
            }
        } else {
            bsp_typeError("invalid array shape for bsp.array(arrayPath,arrayShape)");
            Py_RETURN_FALSE;
        }

        unsigned nDims = 1;
        for (unsigned iDim = 1; iDim < 7; ++iDim) {
            if (dimSize[iDim] == 0) {
                break;
            } 
            ++ nDims;
        }

        ArrayShape::ElementType elemType = ArrayShape::DOUBLE;
        if (dtype) {
            if (0 == strcmp(dtype,"i1") || 0 == strcmp(dtype,"int8"))
                elemType = ArrayShape::INT8;
            else if (0 == strcmp(dtype,"i2") || 0 == strcmp(dtype,"int16"))
                elemType = ArrayShape::INT16;
            else if (0 == strcmp(dtype,"i4") || 0 == strcmp(dtype,"int32"))
                elemType = ArrayShape::INT32;
            else if (0 == strcmp(dtype,"i8") || 0 == strcmp(dtype,"int64"))
                elemType = ArrayShape::INT64;
            else if (0 == strcmp(dtype,"u1") || 0 == strcmp(dtype,"uint8"))
                elemType = ArrayShape::UINT8;
            else if (0 == strcmp(dtype,"u2") || 0 == strcmp(dtype,"uint16"))
                elemType = ArrayShape::UINT16;
            else if (0 == strcmp(dtype,"u4") || 0 == strcmp(dtype,"uint32"))
                elemType = ArrayShape::UINT32;
            else if (0 == strcmp(dtype,"u8") || 0 == strcmp(dtype,"uint64"))
                elemType = ArrayShape::UINT64;
            else if (0 == strcmp(dtype,"f") || 0 == strcmp(dtype,"f4") || 0 == strcmp(dtype,"float32"))
                elemType = ArrayShape::FLOAT;
            else if (0 == strcmp(dtype,"d") || 0 == strcmp(dtype,"f8") || 0 == strcmp(dtype,"float64"))
                elemType = ArrayShape::DOUBLE;
            else if (0 == strcmp(dtype,"c8") || 0 == strcmp(dtype,"complex64"))
                elemType = ArrayShape::CFLOAT;
            else if (0 == strcmp(dtype,"c") || 0 == strcmp(dtype,"c16") || 0 == strcmp(dtype,"complex128"))
                elemType = ArrayShape::CDOUBLE;
            if (elemType == ArrayShape::BINARY) {
                bsp_typeError("invalid dtype for bsp.array(arrayPath,arrayShape)");
                Py_RETURN_FALSE;
            }
        }

        uint64_t elemSize = ArrayShape::elementSize(elemType);
        try {
            LocalArray *localArray = new LocalArray(std::string(arrayPath), elemType, elemSize, nDims, dimSize);
            if (localArray == NULL) {
                bsp_runtimeError("failed to call bsp.array(arrayPath,arrayShape)");
                Py_RETURN_FALSE;
            }
            runtime_->setObject(std::string(arrayPath), localArray);
            if (fill == fill) {
                uint64_t arraySize = localArray->getElementCount(LocalArray::ALL_DIMS);
                switch (elemType) {
                    case ArrayShape::INT8: 
                        {
                            int8_t src = round(fill);
                            int8_t *dst = (int8_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::INT16:
                        {
                            int16_t src = round(fill);
                            int16_t *dst = (int16_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::INT32:
                        {
                            int32_t src = round(fill);
                            int32_t *dst = (int32_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::INT64:
                        {
                            int64_t src = round(fill);
                            int64_t *dst = (int64_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::UINT8:
                        {
                            uint8_t src = round(fill);
                            uint8_t *dst = (uint8_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::UINT16:
                        {
                            uint16_t src = round(fill);
                            uint16_t *dst = (uint16_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::UINT32:
                        {
                            uint32_t src = round(fill);
                            uint32_t *dst = (uint32_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::UINT64:
                        {
                            uint64_t src = round(fill);
                            uint64_t *dst = (uint64_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::FLOAT:
                        {
                            float src = fill;
                            float *dst = (float *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::DOUBLE:
                        {
                            double src = fill;
                            double *dst = (double *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i)
                                dst[i] = src;
                            break;
                        }
                    case ArrayShape::CINT8: 
                        {
                            int8_t src = round(fill);
                            int8_t *dst = (int8_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CINT16:
                        {
                            int16_t src = round(fill);
                            int16_t *dst = (int16_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CINT32:
                        {
                            int32_t src = round(fill);
                            int32_t *dst = (int32_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CINT64:
                        {
                            int64_t src = round(fill);
                            int64_t *dst = (int64_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CUINT8:
                        {
                            uint8_t src = round(fill);
                            uint8_t *dst = (uint8_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CUINT16:
                        {
                            uint16_t src = round(fill);
                            uint16_t *dst = (uint16_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CUINT32:
                        {
                            uint32_t src = round(fill);
                            uint32_t *dst = (uint32_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CUINT64:
                        {
                            uint64_t src = round(fill);
                            uint64_t *dst = (uint64_t *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CFLOAT:
                        {
                            float src = fill;
                            float *dst = (float *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    case ArrayShape::CDOUBLE:
                        {
                            double src = fill;
                            double *dst = (double *)localArray->getData();
                            for (uint64_t i = 0; i < arraySize; ++ i) {
                                dst[i << 1] = src;
                                dst[(i << 1) + 1] = 0.0;
                            }
                            break;
                        }
                    default:
                        break;
                }
                localArray->getData();
            }
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_FALSE;
        }
        return arrayAsNumpy(arrayPath);
    }

    // OK = bsp.delete(up-to-10-paths)
    static PyObject *bsp_delete(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        PyObject *obj[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
        int ok = PyArg_ParseTuple(args, "O|OOOOOOOOO:bsp.delete",
                obj + 0,
                obj + 1,
                obj + 2,
                obj + 3,
                obj + 4,
                obj + 5,
                obj + 6,
                obj + 7,
                obj + 8,
                obj + 9);
        if (!ok) {
            bsp_typeError("invalid dtype for bsp.delete(up-to-10-paths-or-indsets)");
            Py_RETURN_FALSE;
        }
        for (int i = 0; i < 10; ++i) {
            if (NULL == obj[i])
                break;
            try {
                if (PyUnicode_Check(obj[i])) {
                    runtime_->deleteObject(std::string(PyUnicode_AsUTF8(obj[i])));
                } else {
                    std::stringstream ssErr;
                    ssErr << "invalid argument " << i << " for bsp.delete" << std::endl;
                    bsp_typeError(ssErr.str().c_str());
                }
            } catch (const std::exception &e) {
                std::stringstream ssErr;
                ssErr << "failed to delete " << obj[i] << ": " << e.what();
                bsp_runtimeError(ssErr.str());
                Py_RETURN_FALSE;
            }
        }
        Py_RETURN_TRUE;
    }

    // OK = bsp.share(up-to-10-array-Paths)
    static PyObject *bsp_share(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        char *path[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
        int ok = PyArg_ParseTuple(args, "s|sssssssss:bsp.share",
                path + 0,
                path + 1,
                path + 2,
                path + 3,
                path + 4,
                path + 5,
                path + 6,
                path + 7,
                path + 8,
                path + 9);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.share(up-to-10-paths)");
            Py_RETURN_FALSE;
        }
        std::vector<std::string> arrayPaths;
        for (int i = 0; i < 10; ++i) {
            if (NULL == path[i])
                break;
            arrayPaths.push_back(std::string(path[i]));
        }
        try {
            runtime_->share(arrayPaths);
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    }

    // OK = bsp.globalize(procStart,gridShape,up-to-10-array-Paths)
    static PyObject *bsp_globalize(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        uint64_t procStart = 0;
        PyObject *objGridShape = NULL;
        uint64_t gridDimSize[7] = {0,0,0,0,0,0,0};
        unsigned nGridDims = 0;
        char *path[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
        int ok = PyArg_ParseTuple(args,"kOs|sssssssss:bsp.globalize",
                (unsigned long *)&procStart,
                &objGridShape,
                path + 0,
                path + 1,
                path + 2,
                path + 3,
                path + 4,
                path + 5,
                path + 6,
                path + 7,
                path + 8,
                path + 9);
        PyRef refGridShape(objGridShape);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.globalize(procStart,gridShape,up-to-10-array-paths)");
            Py_RETURN_FALSE;
        }
        ok = PyArg_ParseTuple(objGridShape,"k|kkkkkk:bsp.globalize.extractGridShape",
                (unsigned long *)(gridDimSize + 0),
                (unsigned long *)(gridDimSize + 1),
                (unsigned long *)(gridDimSize + 2),
                (unsigned long *)(gridDimSize + 3),
                (unsigned long *)(gridDimSize + 4),
                (unsigned long *)(gridDimSize + 5),
                (unsigned long *)(gridDimSize + 6));
        if (!ok) {
            bsp_typeError("invalid gridShape for bsp.globalize(procStart,gridShape,up-to-10-array-paths)");
            Py_RETURN_FALSE;
        }
        for (unsigned iDim = 0; iDim < 7; ++iDim) {
            if (gridDimSize[iDim] > 0)
                ++ nGridDims;
            else
                break;
        }
        std::vector<std::string> arrayPaths;
        for (int i = 0; i < 10; ++i) {
            if (NULL == path[i])
                break;
            arrayPaths.push_back(std::string(path[i]));
        }
        try {
            runtime_->globalize(arrayPaths,nGridDims,gridDimSize,procStart);
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    }

    // OK = bsp.privatize(up-to-10-array-paths)
    static PyObject *bsp_privatize(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        char *path[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
        int ok = PyArg_ParseTuple(args,"s|sssssssss:bsp.privatize",
                path + 0,
                path + 1,
                path + 2,
                path + 3,
                path + 4,
                path + 5,
                path + 6,
                path + 7,
                path + 8,
                path + 9);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.privatize(up-to-10-paths)");
            Py_RETURN_FALSE;
        }
        std::vector<std::string> arrayPaths;
        for (int i = 0; i < 10; ++i) {
            if (NULL == path[i])
                break;
            arrayPaths.push_back(std::string(path[i]));
        }
        try {
            runtime_->privatize(arrayPaths);
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    }

    // OK = bsp.toProc(procID,up-to-10-local-arrays)
    static PyObject *bsp_toProc(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        unsigned long procID = 0;
        char *path[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
        int ok = PyArg_ParseTuple(args, "ks|sssssssss:bsp.toProc",
                &procID,
                path + 0,
                path + 1,
                path + 2,
                path + 3,
                path + 4,
                path + 5,
                path + 6,
                path + 7,
                path + 8,
                path + 9);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.toProc(procID,up-to-10-local-arrays)");
            return PyRetVal(Py_BuildValue("O",Py_False));
        }
        try {
            for (int i = 0; i < 10; ++i) {
                if (NULL == path[i])
                    break;
                runtime_->exportUserDefined(procID,std::string(path[i]));
            }
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            return PyRetVal(Py_BuildValue("O",Py_False));
        }
        return PyRetVal(Py_BuildValue("O",Py_True));
    }

    void bsp_extractImported(NamedObject *nobj, std::string prefix, PyObject *pyDict) {
        if (nobj->getType() == ARRAY && !nobj->isGlobal()) {
            LocalArray *localArray = nobj->_localArray();
            std::string strKey;
            if (prefix == "")
                strKey = nobj->getName();
            else
                strKey = prefix + "." + nobj->getName();
            PyObject *value = Py_BuildValue("s",localArray->getPath().c_str());
            PyDict_SetItemString(pyDict,strKey.c_str(),value);
        } else if (nobj->getType() == NAMESPACE) {
            NameSpace *nsp = nobj->_namespace();
            std::string myPrefix; 
            if (prefix == "")
                myPrefix = nobj->getName();
            else
                myPrefix = prefix + "." + nobj->getName();
            for (NameSpaceIterator iter = nsp->begin(); iter != nsp->end(); ++iter) {
                bsp_extractImported(iter->second, myPrefix, pyDict);
            }
        }
    }

    // OK = bsp.sync(tag,optionalSendMatrix)
    static PyObject *bsp_sync(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        char *tag = NULL;
        PyObject *objSendMatrix = NULL;
        int ok = PyArg_ParseTuple(args, "s|O:bsp.sync", &tag, &objSendMatrix);
        PyRef refSendMatrix(objSendMatrix);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.sync(tag,optionalSendMatrix)");
            return PyRetVal(Py_BuildValue("O",Py_False));
        }
        try {
            if (NULL != objSendMatrix) {
                if (!PyArray_Check(objSendMatrix)) {
                    bsp_typeError("invalid sendMatrix for bsp.sync(tag,optionalSendMatrix)");
                    return PyRetVal(Py_BuildValue("O",Py_False));
                } else {
                    PyArrayObject *arrSendMatrix = (PyArrayObject *)objSendMatrix;
                    int nDims = PyArray_NDIM(arrSendMatrix);
                    if (2 != nDims || !PyArray_ISBOOL(arrSendMatrix)) {
                        bsp_typeError("invalid sendMatrix for bsp.sync(tag,optionalSendMatrix)");
                        return PyRetVal(Py_BuildValue("O",Py_False));
                    } else {
                        if (PyArray_DIM(arrSendMatrix,0) != nProcs_ ||
                                PyArray_DIM(arrSendMatrix,1) != nProcs_) {
                            bsp_typeError("invalid sendMatrix for bsp.sync(tag,optionalSendMatrix)");
                            return PyRetVal(Py_BuildValue("O",Py_False));
                        } else {
                            bool *sendMatrix = new bool[nProcs_ * nProcs_];
                            bool *data = (bool *)PyArray_DATA(arrSendMatrix);
                            uint64_t k = 0;
                            for (uint64_t i = 0; i < nProcs_; ++i) {
                                for (uint64_t j = 0; j < nProcs_; ++j) {
                                    sendMatrix[k] = data[k];
                                    ++k;
                                }
                            }
                            runtime_->exchange(sendMatrix, tag);
                            delete[] sendMatrix;
                        }
                    }
                }
            } else {
                bool *sendMatrix = new bool[nProcs_ * nProcs_];
                uint64_t k = 0;
                for (uint64_t i = 0; i < nProcs_; ++ i) {
                    for (uint64_t j = 0; j < nProcs_; ++ j) {
                        sendMatrix[k ++] = true;
                    }
                }
                runtime_->exchange(sendMatrix, tag);
                delete[] sendMatrix;
            }

            // update the fromProc lists
            for (uint64_t iProc = 0; iProc < nProcs_; ++iProc) {
                // clear the fromProc list
                PyObject *key, *value;
                Py_ssize_t pos = 0;
                while (PyDict_Next(fromProc_[iProc],&pos,&key,&value)){
                    Py_XDECREF(value);
                }
                PyDict_Clear(fromProc_[iProc]);

                // found all local arrays in the import path
                std::stringstream ss;
                ss << "_import.procID" << iProc;
                std::string pathImport = ss.str();
                if (runtime_->hasObject(pathImport)) {
                    NamedObject *nobjPath = runtime_->getObject(pathImport);
                    NameSpace *nspPath = nobjPath->_namespace();
                    for (NameSpaceIterator iter = nspPath->begin(); iter != nspPath->end(); ++iter)
                        bsp_extractImported(iter->second,"",fromProc_[iProc]);
                }
            }

            // clear updates
            for (std::map<std::string, PyObject *>::iterator iter = futures_.begin();
                    iter != futures_.end(); ++ iter) {
                Future_clearUpdates(iter->second);
            }

        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            return PyRetVal(Py_BuildValue("O",Py_False));
        }
        return PyRetVal(Py_BuildValue("O",Py_True));
    }

    // {partnerID1, ..., partnerIDk} = bsp.async(tag, optionalStopping)
    static PyObject *bsp_async(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        char *tag = NULL;
        PyObject *option = NULL;
        bool stopping = false;
        int ok = PyArg_ParseTuple(args, "s|O:bsp.async", &tag, &option);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.async(tag)");
            Py_RETURN_NONE;
        }
        if (option) {
            if (PyObject_IsTrue(option))
                stopping = true;
        }
	uint64_t procID = 0;
        try {
	    procID = runtime_->exchange(tag, stopping);

            // update the fromProc lists
            for (uint64_t iProc = 0; iProc < nProcs_; ++iProc) {
                // clear the fromProc list
                PyObject *key, *value;
                Py_ssize_t pos = 0;
                while (PyDict_Next(fromProc_[iProc],&pos,&key,&value)){
                    Py_XDECREF(value);
                }
                PyDict_Clear(fromProc_[iProc]);

                // found all local arrays in the import path
                std::stringstream ss;
                ss << "_import.procID" << iProc;
                std::string pathImport = ss.str();
                if (runtime_->hasObject(pathImport)) {
                    NamedObject *nobjPath = runtime_->getObject(pathImport);
                    NameSpace *nspPath = nobjPath->_namespace();
                    for (NameSpaceIterator iter = nspPath->begin(); iter != nspPath->end(); ++iter)
                        bsp_extractImported(iter->second,"",fromProc_[iProc]);
                }
            }

            // clear updates
            for (std::map<std::string, PyObject *>::iterator iter = futures_.begin();
                    iter != futures_.end(); ++ iter) {
                Future_clearUpdates(iter->second);
            }
        } catch (const std::exception &e) {
            bsp_runtimeError(e.what());
            Py_RETURN_NONE;
        }
        PyObject *result = PySet_New(NULL);
        if (procID == runtime_->getMyProcessID()) {
            uint64_t n = runtime_->sizeOfManifest();
            for (uint64_t i = 0; i < n; ++ i)
                PySet_Add(result, Py_BuildValue("I", (unsigned int)runtime_->itemOfManifest(i)));
        } else
            PySet_Add(result, Py_BuildValue("I", (unsigned int)procID));
        return result;
    }

    // bsp.addWorker(procID)
    static PyObject *bsp_addWorker(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        unsigned int procID = runtime_->getMyProcessID();
        int ok = PyArg_ParseTuple(args, "I:bsp.addWorker", &procID);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.addWorker(procID)");
            Py_RETURN_NONE;
        }
        try {
            runtime_->addWorker(procID);
        } catch (const std::exception & e) {
            bsp_runtimeError(e.what());
        }
        Py_RETURN_NONE;
    }

    // bsp.setScheduler(boundOfDelay, smallestBatch, largestBatch)
    static PyObject *bsp_setScheduler(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        unsigned int boundOfDelay = 0, smallestBatch = 0, largestBatch = 0;
        int ok = PyArg_ParseTuple(args, "I|II:bsp.setScheduler", &boundOfDelay, &smallestBatch, &largestBatch);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.setScheduler(boundOfDelay, smallestBatch, largestBatch)");
            Py_RETURN_NONE;
        }
        try {
            runtime_->setScheduler(boundOfDelay, smallestBatch, largestBatch);
        } catch (const std::exception & e) {
            bsp_runtimeError(e.what());
        }
        Py_RETURN_NONE;
    }

    // bsp.unsetScheduler()
    static PyObject *bsp_unsetScheduler(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        try {
            runtime_->unsetScheduler();
        } catch (const std::exception & e) {
            bsp_runtimeError(e.what());
        }
        Py_RETURN_NONE;
    }

    // bsp.enableScheduler()
    static PyObject *bsp_enableScheduler(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        try {
            runtime_->enableScheduler();
        } catch (const std::exception & e) {
            bsp_runtimeError(e.what());
        }
        Py_RETURN_NONE;
    }

    // bsp.disableScheduler()
    static PyObject *bsp_disableScheduler(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        try {
            runtime_->disableScheduler();
        } catch (const std::exception & e) {
            bsp_runtimeError(e.what());
        }
        Py_RETURN_NONE;
    }

    // bsp.toggleVerbose()
    static PyObject *bsp_toggleVerbose(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
	runtime_->setVerbose(!runtime_->isVerbose());
	Py_RETURN_NONE;
    }

    // bsp.tic()
    static PyObject *bsp_tic(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        gettimeofday(&tvStart_, NULL);
	Py_RETURN_NONE;
    }

    // bsp.toc()
    static PyObject *bsp_toc(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        gettimeofday(&tvStop_, NULL);
        double result = tvStop_.tv_sec - tvStart_.tv_sec + 1e-6 * (tvStop_.tv_usec - tvStart_.tv_usec);
        return PyRetVal(Py_BuildValue("d",result));
    }

    // bsp.minimize(params, funValue, funGradient, optMaxIter, optMLim, optStrPenalty, optMethod, optPenaltyLevel)
    static PyObject *bsp_minimize(PyObject *self, PyObject *args, PyObject *kwargs) {
        bsp_getScriptPos();
        static const char * kwlist[] = {"params", "funValue", "funGradient", "maxIter", "mLim", "penalty", "method", "penaltyLevel", "concensus", "concensusRange", "parallel", NULL};
        PyObject *objParam = NULL, *objFunValue = NULL, *objFunGradient = NULL, *objCoParams = NULL, *objCoMultipliers = NULL, *objParallel = NULL;
        unsigned long kMaxIter = 1000, kMLim = 20;
        char *strPenalty = NULL, *strMethod = NULL;
        double dPenaltyLevel = 1, dCoLevel = 1;
        int concensusRange = 0;
        int ok = PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Okkssd(dOO)iO: bsp.minimize", (char **)kwlist, 
                &objParam, &objFunValue, &objFunGradient,
                &kMaxIter, &kMLim, &strPenalty, &strMethod,&dPenaltyLevel,
                &dCoLevel, &objCoParams, &objCoMultipliers, &concensusRange, &objParallel);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.minimize()");
            Py_RETURN_NONE;
        }
        PyRef refParam(objParam), refFunValue(objFunValue), refFunGradient(objFunGradient), refCoParams(objCoParams), refCoMultipliers(objCoMultipliers);
        if (!PyArray_Check(objParam)) {
            bsp_typeError("invalid params for bsp.minimize()");
            Py_RETURN_NONE;
        }
        PyArrayObject *numpyArray = (PyArrayObject *)objParam;
        npy_intp *strides = PyArray_STRIDES(numpyArray);
        int nDims = PyArray_NDIM(numpyArray);
        int elemSize = 0;
        if (strides[nDims - 1] > strides[0]) {
            elemSize = strides[0];
        } else {
            elemSize = strides[nDims - 1];
        }
        if (!PyArray_ISFLOAT(numpyArray) || elemSize != 8) {
            bsp_typeError("invalid type of params for bsp.minimize()");
            Py_RETURN_NONE;
        }
        npy_intp *dimSize = PyArray_DIMS(numpyArray);
        unsigned long nParams = 1;
        for (int iDim = 0; iDim < nDims; ++ iDim) {
            nParams *= dimSize[iDim];
        }

        double *params = (double *)PyArray_BYTES(numpyArray);

        typedef double (* FunValue)(unsigned long, double *);
        typedef void (* FunGradient)(unsigned long, double *, double *);
        FunValue funValue = NULL;
        FunGradient funGradient = NULL;

        try {
            PyObject *myParam = Py_BuildValue("(O)", objFunValue);
            PyObject *objAddress = PyObject_CallObject(ctypes_addressof_, myParam);
            Py_XDECREF(myParam);
            if (objAddress == NULL) {
                bsp_typeError("invalid funValue for bsp.minimize()");
                Py_RETURN_NONE;
            }
            unsigned long long uAddress = 0;
            ok = PyArg_Parse(objAddress, "K", &uAddress);
            if (!ok) {
                bsp_typeError("invalid funValue for bsp.minimize()");
                Py_RETURN_NONE;
            }
            funValue = *(FunValue *) uAddress;
        } catch (const std::exception & e) {
            bsp_runtimeError(e.what());
        }

        if (objFunGradient) {
            try {
                PyObject *myParam = Py_BuildValue("(O)", objFunGradient);
                PyObject *objAddress = PyObject_CallObject(ctypes_addressof_, myParam);
                if (objAddress == NULL) {
                    bsp_typeError("invalid funGradient for bsp.minimize()");
                    Py_RETURN_NONE;
                }
                unsigned long long uAddress = 0;
                ok = PyArg_Parse(objAddress, "K", &uAddress);
                if (!ok) {
                    bsp_typeError("invalid funValue for bsp.minimize()");
                    Py_RETURN_NONE;
                }
                funGradient = *(FunGradient *) uAddress;
            } catch (const std::exception & e) {
                bsp_runtimeError(e.what());
            }
        }

        double *coParams = NULL;
        if (objCoParams) {
            if (!PyArray_Check(objCoParams)) {
                bsp_typeError("invalid coParams for bsp.minimize()");
                Py_RETURN_NONE;
            }
            PyArrayObject *coParamArray = (PyArrayObject *)objCoParams;
            npy_intp *stridesCoParams = PyArray_STRIDES(coParamArray);
            int nDimsCoParams = PyArray_NDIM(coParamArray);
            int elemSizeCoParams = 0;
            if (stridesCoParams[nDimsCoParams - 1] > stridesCoParams[0]) {
                elemSizeCoParams = stridesCoParams[0];
            } else {
                elemSizeCoParams = stridesCoParams[nDimsCoParams - 1];
            }
            if (!PyArray_ISFLOAT(coParamArray) || elemSizeCoParams != 8) {
                bsp_typeError("invalid type of coParams for bsp.minimize()");
                Py_RETURN_NONE;
            }
            npy_intp *dimSizeCoParams = PyArray_DIMS(coParamArray);
            unsigned long nCoParams = 1;
            for (int iDim = 0; iDim < nDimsCoParams; ++ iDim) {
                nCoParams *= dimSizeCoParams[iDim];
            }
            if (nCoParams != nParams) {
                bsp_typeError("invalid size of coParams for bsp.minimize()");
                Py_RETURN_NONE;
            }

            coParams = (double *)PyArray_BYTES(coParamArray);
        }

        double *coMultipliers = NULL;
        if (objCoMultipliers) {
            if (!PyArray_Check(objCoMultipliers)) {
                bsp_typeError("invalid coMultipliers for bsp.minimize()");
                Py_RETURN_NONE;
            }
            PyArrayObject *coMultiplierArray = (PyArrayObject *)objCoMultipliers;
            npy_intp *stridesCoMultipliers = PyArray_STRIDES(coMultiplierArray);
            int nDimsCoMultipliers = PyArray_NDIM(coMultiplierArray);
            int elemSizeCoMultipliers = 0;
            if (stridesCoMultipliers[nDimsCoMultipliers - 1] > stridesCoMultipliers[0]) {
                elemSizeCoMultipliers = stridesCoMultipliers[0];
            } else {
                elemSizeCoMultipliers = stridesCoMultipliers[nDimsCoMultipliers - 1];
            }
            if (!PyArray_ISFLOAT(coMultiplierArray) || elemSizeCoMultipliers != 8) {
                bsp_typeError("invalid type of coMultipliers for bsp.minimize()");
                Py_RETURN_NONE;
            }
            npy_intp *dimSizeCoMultipliers = PyArray_DIMS(coMultiplierArray);
            unsigned long nCoMultipliers = 1;
            for (int iDim = 0; iDim < nDimsCoMultipliers; ++ iDim) {
                nCoMultipliers *= dimSizeCoMultipliers[iDim];
            }
            if (nCoMultipliers != nParams) {
                bsp_typeError("invalid size of coMultipliers for bsp.minimize()");
                Py_RETURN_NONE;
            }

            coMultipliers = (double *)PyArray_BYTES(coMultiplierArray);
        }

        GradientBasedOptimization::Penalty penalty = GradientBasedOptimization::PENALTY_NONE;
        if (strPenalty) {
            if (0 == strcmp(strPenalty, "LogSum") || 0 == strcmp(strPenalty, "LOGSUM")) {
                penalty = GradientBasedOptimization::PENALTY_LOGSUM;
            } else if (0 == strcmp(strPenalty, "L1")) {
                penalty = GradientBasedOptimization::PENALTY_L1;
            } else if (0 == strcmp(strPenalty, "L2")) {
                penalty = GradientBasedOptimization::PENALTY_L2;
            } else {
                bsp_runtimeError("bsp.minimize: unknown penalty");
            }
        }

        ParallelOptimization parallel(funValue, funGradient);
        if (objParallel) {
            if (PyObject_IsTrue(objParallel)) {
                funValue = parallelFunValue;
                funGradient = parallelGradient;
            }
        }
        double result = 0.0;
        if (strMethod == NULL) {
            LBFGS lbfgs(nParams, funValue, kMaxIter, funGradient, kMLim, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                lbfgs.setPenalty(penalty);
                lbfgs.setPenaltyLevel(dPenaltyLevel, false);
            }
            if (coParams) {
                lbfgs.setCoLevel(dCoLevel, false);
                lbfgs.setCoParams(coParams);
                lbfgs.setCoMultipliers(coMultipliers);
                lbfgs.setCoRange(concensusRange);
            }
            lbfgs.minimize();
            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = lbfgs.param(i);
            }
            result = lbfgs.value();
        } else if (0 == strcmp(strMethod, "LBFGS")) {
            LBFGS lbfgs(nParams, funValue, kMaxIter, funGradient, kMLim, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                lbfgs.setPenalty(penalty);
                lbfgs.setPenaltyLevel(dPenaltyLevel, false);
            }
            if (coParams) {
                lbfgs.setCoLevel(dCoLevel, false);
                lbfgs.setCoParams(coParams);
                lbfgs.setCoMultipliers(coMultipliers);
                lbfgs.setCoRange(concensusRange);
            }
            lbfgs.minimize();
            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = lbfgs.param(i);
            }
            result = lbfgs.value();
        } else if (0 == strcmp(strMethod, "BFGS")) {
            BFGS bfgs(nParams, funValue, kMaxIter, funGradient, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                bfgs.setPenalty(penalty);
                bfgs.setPenaltyLevel(dPenaltyLevel, false);
            }
            if (coParams) {
                bfgs.setCoLevel(dCoLevel, false);
                bfgs.setCoParams(coParams);
                bfgs.setCoMultipliers(coMultipliers);
                bfgs.setCoRange(concensusRange);
            }
            bfgs.minimize();
            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = bfgs.param(i);
            }
            result = bfgs.value();
        } else if (0 == strcmp(strMethod, "CG")) {
            CG cg(nParams, funValue, kMaxIter, funGradient, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                cg.setPenalty(penalty);
                cg.setPenaltyLevel(dPenaltyLevel, false);
            }
            if (coParams) {
                cg.setCoLevel(dCoLevel, false);
                cg.setCoParams(coParams);
                cg.setCoMultipliers(coMultipliers);
                cg.setCoRange(concensusRange);
            }
            cg.minimize();
            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = cg.param(i);
            }
            result = cg.value();
        } else {
            bsp_runtimeError("unknown optMethod for bsp.minimize()");
            Py_RETURN_NONE;
        } 
        return PyRetVal(Py_BuildValue("d", result));
    }

    // bsp.maximize(params, funValue, funGradient, optMaxIter, optMLim, optStrPenalty, optMethod, optPenaltyLevel)
    static PyObject *bsp_maximize(PyObject *self, PyObject *args, PyObject *kwargs) {
        bsp_getScriptPos();
        static const char * kwlist[] = {"params", "funValue", "funGradient", "maxIter", "mLim", "penalty", "method", "penaltyLevel", "concensus", "concensusRange", "parallel", NULL};
        PyObject *objParam = NULL, *objFunValue = NULL, *objFunGradient = NULL, *objCoParams = NULL, *objCoMultipliers = NULL, *objParallel = NULL;
        unsigned long kMaxIter = 1000, kMLim = 20;
        char *strPenalty = NULL, *strMethod = NULL;
        double dPenaltyLevel = 1.0, dCoLevel = 1.0;
        int concensusRange = 0;
        int ok = PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Okkssd(dOO)ii: bsp.maximize", (char **)kwlist,
                &objParam, &objFunValue, &objFunGradient,
                &kMaxIter, &kMLim, &strPenalty, &strMethod, &dPenaltyLevel, 
                &dCoLevel, &objCoParams, &objCoMultipliers, &concensusRange, &objParallel);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.maximize()");
            Py_RETURN_NONE;
        }
        PyRef refParam(objParam), refFunValue(objFunValue), refFunGradient(objFunGradient), refCoParams(objCoParams), refCoMultipliers(objCoMultipliers);
        if (!PyArray_Check(objParam)) {
            bsp_typeError("invalid params for bsp.maximize()");
            Py_RETURN_NONE;
        }

        PyArrayObject *numpyArray = (PyArrayObject *)objParam;
        npy_intp *strides = PyArray_STRIDES(numpyArray);
        int nDims = PyArray_NDIM(numpyArray);
        int elemSize = 0;
        if (strides[nDims - 1] > strides[0]) {
            elemSize = strides[0];
        } else {
            elemSize = strides[nDims - 1];
        }
        if (!PyArray_ISFLOAT(numpyArray) || elemSize != 8) {
            bsp_typeError("invalid type of params for bsp.maximize()");
            Py_RETURN_NONE;
        }
        npy_intp *dimSize = PyArray_DIMS(numpyArray);
        unsigned long nParams = 1;
        for (int iDim = 0; iDim < nDims; ++ iDim) {
            nParams *= dimSize[iDim];
        }

        double *params = (double *)PyArray_BYTES(numpyArray);

        typedef double (* FunValue)(unsigned long, double *);
        typedef void (* FunGradient)(unsigned long, double *, double *);
        FunValue funValue = NULL;
        FunGradient funGradient = NULL;

        try {
            PyObject *myParam = Py_BuildValue("(O)", objFunValue);
            PyObject *objAddress = PyObject_CallObject(ctypes_addressof_, myParam);
            Py_XDECREF(myParam);
            if (objAddress == NULL) {
                bsp_typeError("invalid funValue for bsp.maximize()");
                Py_RETURN_NONE;
            }
            unsigned long long uAddress = 0;
            ok = PyArg_Parse(objAddress, "K", &uAddress);
            if (!ok) {
                bsp_typeError("invalid funValue for bsp.maximize()");
                Py_RETURN_NONE;
            }
            funValue = *(FunValue *) uAddress;
        } catch (const std::exception & e) {
            bsp_runtimeError(e.what());
        }

        if (objFunGradient) {
            try {
                PyObject *myParam = Py_BuildValue("(O)", objFunGradient);
                PyObject *objAddress = PyObject_CallObject(ctypes_addressof_, myParam);
                Py_XDECREF(myParam);
                if (objAddress == NULL) {
                    bsp_typeError("invalid funGradient for bsp.maximize()");
                    Py_RETURN_NONE;
                }
                unsigned long long uAddress = 0;
                ok = PyArg_Parse(objAddress, "K", &uAddress);
                if (!ok) {
                    bsp_typeError("invalid funValue for bsp.maximize()");
                    Py_RETURN_NONE;
                }
                funGradient = *(FunGradient *) uAddress;
            } catch (const std::exception & e) {
                bsp_runtimeError(e.what());
            }
        }

        double *coParams = NULL;
        if (objCoParams) {
            if (!PyArray_Check(objCoParams)) {
                bsp_typeError("invalid coParams for bsp.maximize()");
                Py_RETURN_NONE;
            }
            PyArrayObject *coParamArray = (PyArrayObject *)objCoParams;
            npy_intp *stridesCoParams = PyArray_STRIDES(coParamArray);
            int nDimsCoParams = PyArray_NDIM(coParamArray);
            int elemSizeCoParams = 0;
            if (stridesCoParams[nDimsCoParams - 1] > stridesCoParams[0]) {
                elemSizeCoParams = stridesCoParams[0];
            } else {
                elemSizeCoParams = stridesCoParams[nDimsCoParams - 1];
            }
            if (!PyArray_ISFLOAT(coParamArray) || elemSizeCoParams != 8) {
                bsp_typeError("invalid type of coParams for bsp.maximize()");
                Py_RETURN_NONE;
            }
            npy_intp *dimSizeCoParams = PyArray_DIMS(coParamArray);
            unsigned long nCoParams = 1;
            for (int iDim = 0; iDim < nDimsCoParams; ++ iDim) {
                nCoParams *= dimSizeCoParams[iDim];
            }
            if (nCoParams != nParams) {
                bsp_typeError("invalid size of coParams for bsp.maximize()");
                Py_RETURN_NONE;
            }

            coParams = (double *)PyArray_BYTES(coParamArray);
        }

        double *coMultipliers = NULL;
        if (objCoMultipliers) {
            if (!PyArray_Check(objCoMultipliers)) {
                bsp_typeError("invalid coMultipliers for bsp.maximize()");
                Py_RETURN_NONE;
            }
            PyArrayObject *coMultiplierArray = (PyArrayObject *)objCoMultipliers;
            npy_intp *stridesCoMultipliers = PyArray_STRIDES(coMultiplierArray);
            int nDimsCoMultipliers = PyArray_NDIM(coMultiplierArray);
            int elemSizeCoMultipliers = 0;
            if (stridesCoMultipliers[nDimsCoMultipliers - 1] > stridesCoMultipliers[0]) {
                elemSizeCoMultipliers = stridesCoMultipliers[0];
            } else {
                elemSizeCoMultipliers = stridesCoMultipliers[nDimsCoMultipliers - 1];
            }
            if (!PyArray_ISFLOAT(coMultiplierArray) || elemSizeCoMultipliers != 8) {
                bsp_typeError("invalid type of coMultipliers for bsp.maximize()");
                Py_RETURN_NONE;
            }
            npy_intp *dimSizeCoMultipliers = PyArray_DIMS(coMultiplierArray);
            unsigned long nCoMultipliers = 1;
            for (int iDim = 0; iDim < nDimsCoMultipliers; ++ iDim) {
                nCoMultipliers *= dimSizeCoMultipliers[iDim];
            }
            if (nCoMultipliers != nParams) {
                bsp_typeError("invalid size of coMultipliers for bsp.maximize()");
                Py_RETURN_NONE;
            }

            coMultipliers = (double *)PyArray_BYTES(coMultiplierArray);
        }

        GradientBasedOptimization::Penalty penalty = GradientBasedOptimization::PENALTY_NONE;
        if (strPenalty) {
            if (0 == strcmp(strPenalty, "LogSum") || 0 == strcmp(strPenalty, "LOGSUM")) {
                penalty = GradientBasedOptimization::PENALTY_LOGSUM;
            } else if (0 == strcmp(strPenalty, "L1")) {
                penalty = GradientBasedOptimization::PENALTY_L1;
            } else if (0 == strcmp(strPenalty, "L2")) {
                penalty = GradientBasedOptimization::PENALTY_L2;
            } else {
                bsp_runtimeError("bsp.maximize: unknown penalty");
            }
        }

        if (kMLim > nParams)
            kMLim = nParams;
        ParallelOptimization(funValue, funGradient);
        if (objParallel) {
            if (PyObject_IsTrue(objParallel)) {
                funValue = parallelFunValue;
                funGradient = parallelGradient;
            }
        }
        double result = 0.0;
        if (strMethod == NULL) {
            LBFGS lbfgs(nParams, funValue, kMaxIter, funGradient, kMLim, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                lbfgs.setPenalty(penalty);
                lbfgs.setPenaltyLevel(dPenaltyLevel, true);
            }
            if (coParams) {
                lbfgs.setCoLevel(dCoLevel, true);
                lbfgs.setCoParams(coParams);
                lbfgs.setCoMultipliers(coMultipliers);
                lbfgs.setCoRange(concensusRange);
            }
            lbfgs.maximize();

            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = lbfgs.param(i);
            }
            result = lbfgs.value();
        } else if (0 == strcmp(strMethod, "LBFGS")) {
            LBFGS lbfgs(nParams, funValue, kMaxIter, funGradient, kMLim, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                lbfgs.setPenalty(penalty);
                lbfgs.setPenaltyLevel(dPenaltyLevel, true);
            }
            if (coParams) {
                lbfgs.setCoLevel(dCoLevel, true);
                lbfgs.setCoParams(coParams);
                lbfgs.setCoMultipliers(coMultipliers);
                lbfgs.setCoRange(concensusRange);
            }
            lbfgs.maximize();
            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = lbfgs.param(i);
            }
            result = lbfgs.value();
        } else if (0 == strcmp(strMethod, "BFGS")) {
            BFGS bfgs(nParams, funValue, kMaxIter, funGradient, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                bfgs.setPenalty(penalty);
                bfgs.setPenaltyLevel(dPenaltyLevel, true);
            }
            if (coParams) {
                bfgs.setCoLevel(dCoLevel, true);
                bfgs.setCoParams(coParams);
                bfgs.setCoMultipliers(coMultipliers);
                bfgs.setCoRange(concensusRange);
            }
            bfgs.maximize();
            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = bfgs.param(i);
            }
            result = bfgs.value();
        } else if (0 == strcmp(strMethod, "CG")) {
            CG cg(nParams, funValue, kMaxIter, funGradient, params);
            if (penalty != GradientBasedOptimization::PENALTY_NONE) {
                cg.setPenalty(penalty);
                cg.setPenaltyLevel(dPenaltyLevel, true);
            }
            if (coParams) {
                cg.setCoLevel(dCoLevel, true);
                cg.setCoParams(coParams);
                cg.setCoMultipliers(coMultipliers);
                cg.setCoRange(concensusRange);
            }
            cg.maximize();
            for (unsigned long i = 0; i < nParams; ++ i) {
                params[i] = cg.param(i);
            }
            result = cg.value();
        } else {
            bsp_runtimeError("unknown optMethod for bsp.maximize()");
            Py_RETURN_NONE;
        } 
        return PyRetVal(Py_BuildValue("d", result));
    }

    static PyObject *bsp_concensus(PyObject *self, PyObject *args, PyObject *kwargs) {
        bsp_getScriptPos();
        static const char * kwlist[] = {"nParamsPerWorker", "nWorkers", "params", "multipliers", "center", "centerLevel", "proximityLevel", NULL};
        PyObject *objCoParams = NULL, *objCoMultipliers = NULL, *objParam = NULL;
        unsigned long nParamsPerWorker = 0, nWorkers = 0;
        double dCenterLevel = 1.0, dProximityLevel = 1.0;
        int ok = PyArg_ParseTupleAndKeywords(args, kwargs, "kkOOO|dd: bsp.concensus", (char **)kwlist,
                &nParamsPerWorker, &nWorkers, &objCoParams, &objCoMultipliers, &objParam, &dCenterLevel, &dProximityLevel);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.concensus()");
            Py_RETURN_NONE;
        }
        PyRef refCoParams(objCoParams), refCoMultipliers(objCoMultipliers), refParam(objParam);

        double *coParams = NULL;
        if (objCoParams) {
            if (!PyArray_Check(objCoParams)) {
                bsp_typeError("invalid coParams for bsp.concensus()");
                Py_RETURN_NONE;
            }
            PyArrayObject *coParamArray = (PyArrayObject *)objCoParams;
            npy_intp *stridesCoParams = PyArray_STRIDES(coParamArray);
            int nDimsCoParams = PyArray_NDIM(coParamArray);
            int elemSizeCoParams = 0;
            if (stridesCoParams[nDimsCoParams - 1] > stridesCoParams[0]) {
                elemSizeCoParams = stridesCoParams[0];
            } else {
                elemSizeCoParams = stridesCoParams[nDimsCoParams - 1];
            }
            if (!PyArray_ISFLOAT(coParamArray) || elemSizeCoParams != 8) {
                bsp_typeError("invalid type of coParams for bsp.concensus()");
                Py_RETURN_NONE;
            }
            npy_intp *dimSizeCoParams = PyArray_DIMS(coParamArray);
            unsigned long nCoParams = 1;
            for (int iDim = 0; iDim < nDimsCoParams; ++ iDim) {
                nCoParams *= dimSizeCoParams[iDim];
            }
            if (nCoParams != nParamsPerWorker * nWorkers) {
                bsp_typeError("invalid size of coParams for bsp.concensus()");
                Py_RETURN_NONE;
            }

            coParams = (double *)PyArray_BYTES(coParamArray);
        }

        double *coMultipliers = NULL;
        if (objCoMultipliers) {
            if (!PyArray_Check(objCoMultipliers)) {
                bsp_typeError("invalid coMultipliers for bsp.concensus()");
                Py_RETURN_NONE;
            }
            PyArrayObject *coMultiplierArray = (PyArrayObject *)objCoMultipliers;
            npy_intp *stridesCoMultipliers = PyArray_STRIDES(coMultiplierArray);
            int nDimsCoMultipliers = PyArray_NDIM(coMultiplierArray);
            int elemSizeCoMultipliers = 0;
            if (stridesCoMultipliers[nDimsCoMultipliers - 1] > stridesCoMultipliers[0]) {
                elemSizeCoMultipliers = stridesCoMultipliers[0];
            } else {
                elemSizeCoMultipliers = stridesCoMultipliers[nDimsCoMultipliers - 1];
            }
            if (!PyArray_ISFLOAT(coMultiplierArray) || elemSizeCoMultipliers != 8) {
                bsp_typeError("invalid type of coMultipliers for bsp.concensus()");
                Py_RETURN_NONE;
            }
            npy_intp *dimSizeCoMultipliers = PyArray_DIMS(coMultiplierArray);
            unsigned long nCoMultipliers = 1;
            for (int iDim = 0; iDim < nDimsCoMultipliers; ++ iDim) {
                nCoMultipliers *= dimSizeCoMultipliers[iDim];
            }
            if (nCoMultipliers != nParamsPerWorker * nWorkers) {
                bsp_typeError("invalid size of coMultipliers for bsp.concensus()");
                Py_RETURN_NONE;
            }

            coMultipliers = (double *)PyArray_BYTES(coMultiplierArray);
        }

        double *params = NULL;
        if (objParam) {
            if (!PyArray_Check(objParam)) {
                bsp_typeError("invalid params for bsp.minimize()");
                Py_RETURN_NONE;
            }
            PyArrayObject *numpyArray = (PyArrayObject *)objParam;
            npy_intp *strides = PyArray_STRIDES(numpyArray);
            int nDims = PyArray_NDIM(numpyArray);
            int elemSize = 0;
            if (strides[nDims - 1] > strides[0]) {
                elemSize = strides[0];
            } else {
                elemSize = strides[nDims - 1];
            }
            if (!PyArray_ISFLOAT(numpyArray) || elemSize != 8) {
                bsp_typeError("invalid type of params for bsp.minimize()");
                Py_RETURN_NONE;
            }
            npy_intp *dimSize = PyArray_DIMS(numpyArray);
            unsigned long nParams = 1;
            for (int iDim = 0; iDim < nDims; ++ iDim) {
                nParams *= dimSize[iDim];
            }
            if (nParams != nParamsPerWorker) {
                bsp_typeError("invalid size of center for bsp.maximize()");
                Py_RETURN_NONE;
            }

            params = (double *)PyArray_BYTES(numpyArray);
        }

        double retval = 
            concensus(dProximityLevel, dCenterLevel, nWorkers, nParamsPerWorker, coParams, coMultipliers, params);
        return PyRetVal(Py_BuildValue("d", retval));
    }

    // bsp.findFreqSet(sequence, fileName, optTemplate, optThreshold)
    static PyObject *bsp_findFreqSet(PyObject *self, PyObject *args, PyObject *kwargs) {
        bsp_getScriptPos();
        static const char * kwlist[] = {"sequence", "fileName", "threshold", "tmpl2", "tmpl3", "multiThread", NULL};
        PyObject *objSeq = NULL;
        char *strFileName = NULL;
        unsigned long threshold = 2;
        int t20 = 0, t21 = 0, t30 = 0, t31 = 0, t32 = 0, multiThread = 1;
        int ok = PyArg_ParseTupleAndKeywords(args, kwargs, "Os|k(ii)(iii)i: bsp.findFreqSet", (char **)kwlist,
                &objSeq, &strFileName, &threshold, &t20, &t21, &t30, &t31, &t32, &multiThread);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.findFreqSet(sequence, fileName, threshold, optTemplate)");
            Py_RETURN_FALSE;
        }

        PyRef refSeq(objSeq);
        bool useTmpl2 = (t20 != 0) || (t21 != 0);
        bool useTmpl3 = (t30 != 0) || (t31 != 0) || (t32 != 0);
        if (useTmpl2 && useTmpl3) 
        {
            bsp_typeError("tmpl2 and tmpl3 are not allowed to be used at the same time for bsp.findFreqSet(sequence, fileName, threshold, optTemplate)");
            Py_RETURN_FALSE;
        }

        if (!PyArray_Check(objSeq)) {
            bsp_typeError("invalid sequence for bsp.findFreqSet(sequence, fileName, threshold, optTemplate)");
            Py_RETURN_FALSE;
        }
        PyArrayObject *numpyArray = (PyArrayObject *)objSeq;
        npy_intp *strides = PyArray_STRIDES(numpyArray);
        int nDims = PyArray_NDIM(numpyArray);
        unsigned int elemSize = 0;
        if (strides[nDims - 1] > strides[0]) {
            elemSize = strides[0];
        } else {
            elemSize = strides[nDims - 1];
        }
        if (!PyArray_ISINTEGER(numpyArray) || elemSize != 2) {
            bsp_typeError("invalid type of sequence for bsp.findFreqSet(sequence, fileName, threshold, optTemplate)");
            Py_RETURN_FALSE;
        }

        npy_intp *dimSize = PyArray_DIMS(numpyArray);
        unsigned long nUnits = 1;
        for (int iDim = 0; iDim < nDims; ++ iDim) {
            nUnits *= dimSize[iDim];
        }
        unsigned short *x = (unsigned short *)PyArray_BYTES(numpyArray);
        Apriori apriori(threshold);
        if (useTmpl2) {
            apriori.scan(nUnits, t20, t21, x);
        } else if (useTmpl3) {
            apriori.scan(nUnits, t30, t31, t32, x);
        } else {
            apriori.scan(nUnits, x, multiThread != 0);
        }
        apriori.saveToFile(strFileName);
        Py_RETURN_TRUE;
    }

    static PyObject *bsp_getFreq(PyObject *self, PyObject *args, PyObject *kwargs) {
        bsp_getScriptPos();
        static const char * kwlist[] = {"freq", "be", "sequence", "fileName", NULL};
        PyObject *objSeq = NULL, *objFreq = NULL, *objBE = NULL;
        char *strFileName = NULL;
        int ok = PyArg_ParseTupleAndKeywords(args, kwargs, "OOOs: bsp.getFreq", (char **)kwlist,
                &objFreq, &objBE, &objSeq, &strFileName);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.getFreq(freq, be, sequence, fileName)");
            Py_RETURN_FALSE;
        }

        PyRef refSeq(objSeq), refFreq(objFreq), refBE(objBE);
        if (!PyArray_Check(objSeq)) {
            bsp_typeError("invalid sequence for bsp.getFreq(freq, be, sequence, fileName)");
            Py_RETURN_FALSE;
        }
        PyArrayObject *numpyArray = (PyArrayObject *)objSeq;
        npy_intp *strides = PyArray_STRIDES(numpyArray);
        int nDims = PyArray_NDIM(numpyArray);
        unsigned int elemSize = 0;
        if (strides[nDims - 1] > strides[0]) {
            elemSize = strides[0];
        } else {
            elemSize = strides[nDims - 1];
        }
        if (!PyArray_ISINTEGER(numpyArray) || elemSize != 2) {
            bsp_typeError("invalid type of sequence for bsp.getFreq(freq, be, sequence, fileName)");
            Py_RETURN_FALSE;
        }

        npy_intp *dimSize = PyArray_DIMS(numpyArray);
        unsigned long nUnits = 1;
        for (int iDim = 0; iDim < nDims; ++ iDim) {
            nUnits *= dimSize[iDim];
        }
        unsigned short *x = (unsigned short *)PyArray_BYTES(numpyArray);

        if (!PyArray_Check(objFreq)) {
            bsp_typeError("invalid freq for bsp.getFreq(freq, be, sequence, fileName)");
            Py_RETURN_FALSE;
        }
        PyArrayObject *freqArray = (PyArrayObject *)objFreq;
        npy_intp *stridesFreq = PyArray_STRIDES(freqArray);
        int nDimsFreq = PyArray_NDIM(freqArray);
        unsigned int elemSizeFreq = 0;
        if (stridesFreq[nDimsFreq - 1] > stridesFreq[0]) {
            elemSizeFreq = stridesFreq[0];
        } else {
            elemSizeFreq = stridesFreq[nDimsFreq - 1];
        }
        npy_intp *dimSizeFreq = PyArray_DIMS(freqArray);
        unsigned long nFreq = 1;
        for (int iDim = 0; iDim < nDimsFreq; ++ iDim) {
            nFreq *= dimSizeFreq[iDim];
        }
        if (!PyArray_ISINTEGER(freqArray) || elemSizeFreq != 4 || nFreq != nUnits * 4) {
            std::cout << PyArray_ISINTEGER(freqArray) << ", " << elemSizeFreq << ", " << nFreq << ", " << nUnits << std::endl;
            bsp_typeError("invalid type of freq for bsp.getFreq(freq, be, sequence, fileName)");
            Py_RETURN_FALSE;
        }
        int *freq = (int *)PyArray_BYTES(freqArray);

        if (!PyArray_Check(objBE)) {
            bsp_typeError("invalid freq for bsp.getFreq(freq, be, sequence, fileName)");
            Py_RETURN_FALSE;
        }
        PyArrayObject *beArray = (PyArrayObject *) objBE ;
        npy_intp *stridesBE = PyArray_STRIDES(beArray);
        int nDimsBE = PyArray_NDIM(beArray);
        unsigned int elemSizeBE = 0;
        if (stridesBE[nDimsBE - 1] > stridesBE[0]) {
            elemSizeBE = stridesBE[0];
        } else {
            elemSizeBE = stridesBE[nDimsBE - 1];
        }
        npy_intp *dimSizeBE = PyArray_DIMS(beArray);
        unsigned long nBE = 1;
        for (int iDim = 0; iDim < nDimsBE; ++ iDim) {
            nBE *= dimSizeBE[iDim];
        }
        if (!PyArray_ISFLOAT(beArray) || elemSizeBE != 8 || nBE != nUnits * 8) {
            bsp_typeError("invalid type of be for bsp.getFreq(freq, be, sequence, fileName)");
            Py_RETURN_FALSE;
        }
        double *be = (double *)PyArray_BYTES(beArray);

        Apriori apriori(0);
        apriori.loadFromFile(strFileName);
        apriori.getFreq(nUnits, x, freq, freq + nUnits, freq + 2 * nUnits, freq + 3 * nUnits,
                be, be + nUnits, be + 2 * nUnits, be + 3 * nUnits,
                be + 4 * nUnits, be + 5 * nUnits, be + 6 * nUnits, be + 7 * nUnits);
        Py_RETURN_TRUE;
    }

    static PyObject *bsp_getFreqIndex(PyObject *self, PyObject *args, PyObject *kwargs) {
        bsp_getScriptPos();
        static const char * kwlist[] = {"result", "sequence", "fileName", "tmpl2", "tmpl3", "start", NULL};
        PyObject *objSeq = NULL, *objResult = NULL;
        char *strFileName = NULL;
        int t20 = 0, t21 = 0, t30 = 0, t31 = 0, t32 = 0, start = 0;
        int ok = PyArg_ParseTupleAndKeywords(args, kwargs, "OOs|(ii)(iii)i: bsp.getFreqIndex", (char **)kwlist,
                &objResult, &objSeq, &strFileName, &t20, &t21, &t30, &t31, &t32, &start);
        if (!ok) {
            bsp_typeError("invalid arguments for bsp.getFreqIndex");
            Py_RETURN_NONE;
        }
        PyRef refResult(objResult), refSeq(objSeq);
        bool useTmpl2 = (t20 != 0) || (t21 != 0);
        bool useTmpl3 = (t30 != 0) || (t31 != 0) || (t32 != 0);
        if (useTmpl2 && useTmpl3) 
        {
            bsp_typeError("tmpl2 and tmpl3 are not allowed to be used at the same time for bsp.getFreqIndex");
            Py_RETURN_NONE;
        }

        if (!PyArray_Check(objResult)) {
            bsp_typeError("invalid result array for bsp.getFreqIndex");
            Py_RETURN_NONE;
        }
        PyArrayObject *resultArray = (PyArrayObject *)objResult;
        npy_intp *stridesOfResult = PyArray_STRIDES(resultArray);
        int nDimsOfResult = PyArray_NDIM(resultArray);
        unsigned int elemSizeOfResult = 0;
        if (stridesOfResult[nDimsOfResult - 1] > stridesOfResult[0]) {
            elemSizeOfResult = stridesOfResult[0];
        } else {
            elemSizeOfResult = stridesOfResult[nDimsOfResult - 1];
        }
        if (elemSizeOfResult != 4 || !PyArray_ISINTEGER(resultArray)) {
            bsp_typeError("Invalid element type of the result array for bsp.findFreqIndex");
            Py_RETURN_NONE;
        }
        npy_intp *dimSizeOfResult = PyArray_DIMS(resultArray);
        unsigned long n = 1;
        for (int iDim = 0; iDim < nDimsOfResult; ++ iDim) {
            n *= dimSizeOfResult[iDim];
        }
        int *result = (int *)PyArray_BYTES(resultArray);
        if (!PyArray_Check(objSeq)) {
            bsp_typeError("invalid sequence for bsp.getFreqIndex");
            Py_RETURN_NONE;
        }
        PyArrayObject *numpyArray = (PyArrayObject *)objSeq;
        npy_intp *strides = PyArray_STRIDES(numpyArray);
        int nDims = PyArray_NDIM(numpyArray);
        unsigned int elemSize = 0;
        if (strides[nDims - 1] > strides[0]) {
            elemSize = strides[0];
        } else {
            elemSize = strides[nDims - 1];
        }
        npy_intp *dimSize = PyArray_DIMS(numpyArray);
        unsigned long nUnits = 1;
        for (int iDim = 0; iDim < nDims; ++ iDim) {
            nUnits *= dimSize[iDim];
        }
        if (!PyArray_ISINTEGER(numpyArray) || elemSize != 2 || nUnits != n) {
            bsp_typeError("invalid element type/size of sequence for bsp.getFreqIndex");
            Py_RETURN_NONE;
        }
        unsigned short *x = (unsigned short *)PyArray_BYTES(numpyArray);
        Apriori apriori(0);
        apriori.loadFromFile(strFileName);
        int retval = 0;
        if (useTmpl2) {
            retval = apriori.getIndex2(n, t20, t21, x, start, result);
        } else if (useTmpl3) {
            retval = apriori.getIndex3(n, t30, t31, t32, x, start, result);
        }
        return PyRetVal(Py_BuildValue("i", retval));
    }

    static PyObject *bsp_mostFrequent(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        char *fileName = NULL;
        int ok = PyArg_ParseTuple(args, "s:bsp.mostFrequent", &fileName);
        if (!ok) {
            bsp_typeError("invalid file name for bsp.mostFrequent");
            Py_RETURN_NONE;
        }
        Apriori apriori(0);
        apriori.loadFromFile(fileName);
        unsigned long freq = 0;
        unsigned short word = 0;
        freq = apriori.mostFrequent(&word);
        //printf("largest be = %lf\n", apriori.largestBE());
        return PyRetVal(Py_BuildValue("(k,k)", freq, word));
    }

    static PyObject *bsp_wordFreq(PyObject *self, PyObject *args) {
        bsp_getScriptPos();
        char *fileName = NULL;
        int wi = 0;
        int ok = PyArg_ParseTuple(args, "si:bsp.wordFreq", &fileName, &wi);
        if (!ok) {
            bsp_typeError("invalid file name for bsp.wordFreq");
            Py_RETURN_NONE;
        }
        Apriori apriori(0);
        apriori.loadFromFile(fileName);
        return PyRetVal(Py_BuildValue("k", apriori.freq((unsigned short)wi)));
    }


    static PyMethodDef bspMethods_[] = {
        {"myProcID",bsp_myProcID,METH_VARARGS,"get the rank of current process"},
        {"procCount",bsp_procCount,METH_VARARGS,"get the number of processes"},
        {"fromProc",bsp_fromProc,METH_VARARGS,"get the imported objects from a given process"},
        {"toProc",bsp_toProc,METH_VARARGS,"export an object to a given process"},
        {"fromObject",bsp_fromObject,METH_VARARGS,"build local array from an object"},
        {"fromNumpy",bsp_fromNumpy,METH_VARARGS,"build local array from a numpy array"},
        {"toObject",bsp_toObject,METH_VARARGS,"build an object from a local array"},
        {"toNumpy",bsp_toNumpy,METH_VARARGS,"build a numpy array from a local aray"},
        {"asNumpy",bsp_asNumpy,METH_VARARGS,"open a view of numpy array for accessing a local array"},
        {"array",(PyCFunction)bsp_array,METH_VARARGS | METH_KEYWORDS,"create a new local array"},
        {"delete",bsp_delete,METH_VARARGS,"delete a local array or a path"},
        {"share",bsp_share,METH_VARARGS,"share local arrays"},
        {"globalize",bsp_globalize,METH_VARARGS,"globalize local arrays"},
        {"privatize",bsp_privatize,METH_VARARGS,"privatize share/global arrays"},
        {"sync",bsp_sync,METH_VARARGS,"sync data with optional send-matrix"},
	{"async", bsp_async, METH_VARARGS,"asynchronization"},
        {"addWorker", bsp_addWorker, METH_VARARGS, "add a worker for asynchronization"},
        {"setScheduler", bsp_setScheduler, METH_VARARGS,"set a scheduler for asynchronization"},
        {"unsetScheduler", bsp_unsetScheduler, METH_VARARGS, "unset the scheduler for asynchronization"},
        {"enableScheduler", bsp_enableScheduler, METH_VARARGS, "enable the scheduler for asynchronization"},
        {"disableScheduler", bsp_disableScheduler, METH_VARARGS, "disable the scheduler for asynchronization"},
	{"toggleVerbose", bsp_toggleVerbose, METH_VARARGS, "toggle verbose"},
        {"tic", bsp_tic, METH_VARARGS, "start timing"},
        {"toc", bsp_toc, METH_VARARGS, "stop timing"},
        {"minimize", (PyCFunction)bsp_minimize, METH_VARARGS | METH_KEYWORDS, "find the minimum of a given function"},
        {"maximize", (PyCFunction)bsp_maximize, METH_VARARGS | METH_KEYWORDS, "find the maximum of a given function"},
        {"concensus", (PyCFunction)bsp_concensus, METH_VARARGS | METH_KEYWORDS, "update center and multipliers for concensus"},
        {"findFreqSet", (PyCFunction)bsp_findFreqSet, METH_VARARGS | METH_KEYWORDS, "find the frequent sets of a sequence"},
        {"getFreq", (PyCFunction)bsp_getFreq, METH_VARARGS | METH_KEYWORDS, "get the frequency of the words in a sequence"},
        {"getFreqIndex", (PyCFunction)bsp_getFreqIndex, METH_VARARGS | METH_KEYWORDS, "get the indices of frequent sets of the words in a sequence"},
        {"mostFrequent", (PyCFunction)bsp_mostFrequent, METH_VARARGS, "get the most frequent word and its frequency"},
        {"wordFreq", (PyCFunction)bsp_wordFreq, METH_VARARGS, "get the frequency of a word"},
        //{"repeat", bsp_repeat, METH_VARARGS, "repeat an operation"},
        {NULL,NULL,0,NULL}
    };

    PyObject *importArray() {
        import_array();
        return NULL;
    }

    PyMODINIT_FUNC PyInit_bsp() {
        static struct PyModuleDef bspModule = {
            PyModuleDef_HEAD_INIT,
            "bsp",
            "A BSP extension to Python",
            -1,
            bspMethods_,
            NULL,
            NULL,
            NULL,
            NULL
        };
        return PyModule_Create(&bspModule);
    }

    void addType(const char *typeName, PyMethodDef *methods, PyObject *module) {
        PyObject *classDict = PyDict_New();
        int iMethod = 0;
        while (methods[iMethod].ml_name) {
            PyObject *closure = PyCFunction_New(methods + iMethod, NULL);
            PyObject *method = PyInstanceMethod_New(closure);
            PyDict_SetItemString(classDict, methods[iMethod].ml_name, method);
            Py_CLEAR(closure);
            Py_CLEAR(method);
            ++ iMethod;
        }
        PyObject *classBase = PyTuple_New(0);
        PyObject *className = PyUnicode_FromString(typeName);
        PyObject *myClass = PyObject_CallFunctionObjArgs((PyObject *) &PyType_Type, className, classBase, classDict, NULL);

        PyObject *moduleDict = PyModule_GetDict(module);
        PyDict_SetItemString(moduleDict, typeName, myClass);
        Py_CLEAR(classDict);
        Py_CLEAR(className);
        Py_CLEAR(myClass);
    }

    void initBSP(int *pArgc, char ***pArgv) {
        runtime_ = new Runtime(pArgc, pArgv);
        //runtime_->setVerbose(true);
        nProcs_ = runtime_->getNumberOfProcesses();
        fromProc_ = new PyObject *[nProcs_];
        for (unsigned i = 0; i < nProcs_; ++i) {
            fromProc_[i] = Py_BuildValue("{}");
        }

        PyImport_AppendInittab("bsp", PyInit_bsp);

        wchar_t progName[] = L"py3bsp";
        Py_SetProgramName(progName);
        Py_Initialize();
        importArray();

        pickle_ = PyImport_ImportModule("pickle");
        Py_XINCREF(pickle_);

        ctypes_ = PyImport_ImportModule("ctypes");
        Py_XINCREF(ctypes_);

        traceback_ = PyImport_ImportModule("traceback");
        Py_XINCREF(traceback_);

        pickle_dumps_ = PyObject_GetAttrString(pickle_, "dumps");
        assert(PyCallable_Check(pickle_dumps_));
        Py_XINCREF(pickle_dumps_);

        pickle_loads_ = PyObject_GetAttrString(pickle_, "loads");
        assert(PyCallable_Check(pickle_loads_));
        Py_XINCREF(pickle_loads_);

        ctypes_addressof_ = PyObject_GetAttrString(ctypes_, "addressof");
        assert(PyCallable_Check(ctypes_addressof_));
        Py_XINCREF(ctypes_addressof_);

        traceback_extractStack_ = PyObject_GetAttrString(traceback_, "extract_stack");
        Py_XINCREF(traceback_extractStack_);

        PyObject *module = PyImport_ImportModule("bsp");
        module_ = module;
        addType("Dim", DimMethods, module);
        addType("Future", FutureMethods, module);
    }

}
