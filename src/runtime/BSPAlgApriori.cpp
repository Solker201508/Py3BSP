#include "BSPAlgApriori.hpp"
#include <stdexcept>
#include <cassert>

using namespace BSP::Algorithm;

Apriori::Node::Node() {
    for (unsigned int i = 0; i < 256; ++ i) {
        _counter[i] = 0;
        _children[i] = NULL;
    }
}

Apriori::Node::~Node() {
    reset();
}

void Apriori::Node::reset() {
    for (unsigned int i = 0; i < 256; ++ i) {
        if (NULL != _children[i])
            delete _children[i];
        _counter[i] = 0;
        _children[i] = NULL;
    }
}

unsigned long Apriori::Node::countOf(unsigned int k, char *x) {
    if (k <= 0)
        return 0;
    unsigned char i = (unsigned char)x[0];
    if (_counter[i] == 0)
        return 0;
    if (k == 1)
        return _counter[i];
    if (_children[i] == NULL)
        return 0;
    return _children[i]->countOf(k - 1, x + 1);
}

void Apriori::Node::add(unsigned int k, char *x) {
    if (k <= 0)
        return;
    unsigned char i = (unsigned char)x[0];
    if (k == 1) {
        _counter[i] += 1;
        return;
    }
    if (_children[i] != NULL)
        _children[i]->add(k - 1, x + 1);
}

void Apriori::Node::spawn(unsigned int k, unsigned int threshold) {
    if (k <= 1)
        return;
    if (k == 2) {
        for (unsigned int iChild = 0; iChild < 256; ++ iChild) {
            if (_counter[iChild] > threshold)
                _children[iChild] = new Node();
        }
    } else {
        for (unsigned int iChild = 0; iChild < 256; ++ iChild) {
            if (NULL != _children[iChild])
                _children[iChild]->spawn(k - 1, threshold);
        }
    }
}

void Apriori::Node::saveToFile(FILE *file) {
    for (unsigned int i = 0; i < 256; ++ i) {
        long c = (long) _counter[i];
        if (_children[i] == NULL)
            c = -c;
        assert(1 == fwrite(&c, sizeof(c), 1, file));
        if (_children[i] != NULL) {
            _children[i]->saveToFile(file);
        }
    }
}

void Apriori::Node::loadFromFile(FILE *file) {
    reset();
    for (unsigned int i = 0; i < 256; ++ i) {
        long c = 0;
        assert(1 == fread(&c, sizeof(c), 1, file));
        if (c <= 0) {
            _counter[i] = -c;
        } else {
            _counter[i] = c;
            _children[i] = new Node();
            _children[i]->loadFromFile(file);
        }
    }
}

Apriori::Apriori(unsigned int unit) {
    _unit = unit;
    _depth = 0;
    _threshold = 2;
}

Apriori::~Apriori() {
}

void Apriori::setThreshold(unsigned int threshold) {
    _threshold = threshold;
}

void Apriori::scan(unsigned long nUnits, unsigned long unitDepth, char *x) {
    unsigned int n = nUnits - unitDepth + 1;
    unsigned long *posUnit = new unsigned long[n];
    unsigned long pos = 0;
    for (unsigned int i = 0; i < n; ++ i) {
        posUnit[i] = pos;
        pos += _unit;
    }
    for (unsigned long k = 0; k < unitDepth; ++ k) {
        n = scan(n, posUnit, x);
    }
    delete []posUnit;
}

void Apriori::scan(unsigned long nUnits, char *x, int tmplPos1) {
    unsigned int wordLen = 1;
    if (tmplPos1 < 0)
        wordLen -= tmplPos1;
    else if (tmplPos1 > 0)
        wordLen += tmplPos1;
    else {
        scan(nUnits, 1, x);
        return;
    }
    unsigned long n = nUnits - wordLen + 1;
    char *patterns = new char[_unit * n];
    unsigned long *posUnit = new unsigned long[n];
    unsigned int stride1 = _unit;
    if (tmplPos1 < 0) {
        stride1 *= -tmplPos1;
    } else {
        stride1 *= tmplPos1;
    }
    unsigned long pos = 0;
    unsigned long posSrc = 0;
    unsigned long posDst = 0;
    for (unsigned long i = 0; i < n; ++ i) {
        posUnit[i] = pos;
        pos += _unit;
        pos += _unit;

        for (unsigned int j = 0; j < _unit; ++ j) {
            patterns[posDst ++] = x[posSrc + j];
        }
        for (unsigned int j = 0; j < _unit; ++ j) {
            patterns[posDst ++] = x[posSrc + stride1 + j];
        }
        posSrc += _unit;
    }
    for (unsigned int i = 0; i < 2 * _unit; ++ i) {
        n = scan(n, posUnit, patterns);
    }
    delete[] patterns;
    delete[] posUnit;
}

void Apriori::scan(unsigned long nUnits, char *x, int tmplPos1, int tmplPos2) {
    if (tmplPos1 == tmplPos2) {
        scan(nUnits, x, tmplPos1);
        return;
    }
    int minTmplPos = tmplPos1;
    int maxTmplPos = tmplPos1;
    if (tmplPos2 < minTmplPos)
        minTmplPos = tmplPos2;
    else
        maxTmplPos = tmplPos2;

    unsigned int wordLen = 1;
    if (minTmplPos < 0)
        wordLen -= tmplPos1;
    else if (minTmplPos == 0) {
        scan(nUnits, x, maxTmplPos);
        return;
    }

    if (maxTmplPos > 0)
        wordLen += tmplPos1;
    else if (maxTmplPos == 0) {
        scan(nUnits, x, minTmplPos);
        return;
    }

    unsigned long n = nUnits - wordLen + 1;
    char *patterns = new char[_unit * n];
    unsigned long *posUnit = new unsigned long[n];
    unsigned int stride1 = _unit, stride2 = _unit;
    if (minTmplPos < 0) {
        if (maxTmplPos < 0) {
            stride1 *= maxTmplPos-minTmplPos;
            stride2 *= -minTmplPos;
        } else {
            stride1 *= -minTmplPos;
            stride2 *= maxTmplPos;
        }
    } else {
        stride1 *= minTmplPos;
        stride2 *= maxTmplPos;
    }

    unsigned long pos = 0;
    unsigned long posSrc = 0;
    unsigned long posDst = 0;
    for (unsigned long i = 0; i < n; ++ i) {
        posUnit[i] = pos;
        pos += _unit;
        pos += _unit;
        pos += _unit;

        for (unsigned int j = 0; j < _unit; ++ j) {
            patterns[posDst ++] = x[posSrc + j];
        }
        for (unsigned int j = 0; j < _unit; ++ j) {
            patterns[posDst ++] = x[posSrc + stride1 + j];
        }
        for (unsigned int j = 0; j < _unit; ++ j) {
            patterns[posDst ++] = x[posSrc + stride2 + j];
        }
        posSrc += _unit;
    }
    for (unsigned int i = 0; i < 3 * _unit; ++ i) {
        n = scan(n, posUnit, patterns);
    }
    delete[] patterns;
    delete[] posUnit;
}

unsigned long Apriori::scan(unsigned long nUnits, unsigned long *posUnit, char *x) {
    for (unsigned int iChar = 0; iChar < _unit; ++ iChar) {
        for (unsigned long iUnit = 0; iUnit < nUnits; ++ iUnit) {
            _root.add(_depth, x + posUnit[iUnit]);
        }
        ++ _depth;
        _root.spawn(_depth, _threshold);
    }

    unsigned long result = 0;
    for (unsigned long iUnit = 0; iUnit < nUnits; ++ iUnit) {
        if (_root.countOf(_depth, x + posUnit[iUnit]) > _threshold) {
            posUnit[result] = posUnit[iUnit];
            ++ result;
        }
    }
    return result;
}

void Apriori::saveToFile(char *fileName) {
    FILE *file = fopen(fileName, "w");
    assert(NULL != file);
    assert(1 == fwrite(&_unit, sizeof(_unit), 1, file));
    assert(1 == fwrite(&_depth, sizeof(_depth), 1, file));
    assert(1 == fwrite(&_threshold, sizeof(_threshold), 1, file));
    _root.saveToFile(file);
    fclose(file);
}

void Apriori::loadFromFile(char *fileName) {
    FILE *file = fopen(fileName, "r");
    assert(NULL != file);
    assert(1 == fread(&_unit, sizeof(_unit), 1, file));
    assert(1 == fread(&_depth, sizeof(_depth), 1, file));
    assert(1 == fread(&_threshold, sizeof(_threshold), 1, file));
    _root.loadFromFile(file);
    fclose(file);
}

