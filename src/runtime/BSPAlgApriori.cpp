#include "BSPAlgApriori.hpp"
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <pthread.h>
#include <unistd.h>

using namespace BSP::Algorithm;

Apriori::Apriori(unsigned int threshold) {
    _threshold = threshold;
}

Apriori::~Apriori() {
}

class ArgScan1: public std::map<unsigned short, unsigned long> {
    public:
        unsigned short *_x;
        unsigned long _i0, _i1;
};

void *threadScan1(void *pArgs) {
    ArgScan1 *myArg =  (ArgScan1 *)pArgs;
    std::map<unsigned short, unsigned long> &w1 = *myArg;
    unsigned short *x = myArg->_x;
    for (unsigned long i = myArg->_i0; i < myArg->_i1; ++ i) {
        std::map<unsigned short, unsigned long>::iterator loc = w1.find(x[i]);
        if (loc != w1.end()) {
            loc->second += 1;
        } else {
            w1.insert(std::pair<unsigned short, unsigned long>(x[i], 1));
        }
    }
    return NULL;
}

void Apriori::scan1(unsigned long n, unsigned short *x, bool multiThread) {
    if (n >= 1024 && multiThread) {
        int nThreads = sysconf(_SC_NPROCESSORS_ONLN);
        ArgScan1 *args = new ArgScan1[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            args[iThread]._x = x;
            args[iThread]._i0 = iThread * n / nThreads;
            args[iThread]._i1 = (iThread + 1) * n / nThreads;
            pthread_create(threads + iThread, NULL, threadScan1, args + iThread);
        }
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            for (std::map<unsigned short, unsigned long>::iterator loc = args[iThread].begin(); 
                    loc != args[iThread].end(); ++ loc) {
                std::map<unsigned short, unsigned long>::iterator loc1 = _w1.find(loc->first);
                if (loc1 == _w1.end()) {
                    _w1.insert(std::pair<unsigned short, unsigned long>(loc->first, loc->second));
                } else {
                    loc1->second += loc->second;
                }
            }
        }
        delete[] args;
        delete[] threads;
    } else {
        for (unsigned long i = 0; i < n; ++ i) {
            std::map<unsigned short, unsigned long>::iterator loc = _w1.find(x[i]);
            if (loc != _w1.end()) {
                loc->second += 1;
            } else {
                _w1.insert(std::pair<unsigned short, unsigned long>(x[i], 1));
            }
        }
    }
}

unsigned long key2(unsigned short k0, unsigned short k1) {
    return (((unsigned long)k1) << 16) | k0; 
}

unsigned long long key3(unsigned short k0, unsigned short k1, unsigned short k2) {
    return (((unsigned long long)k2) << 32) | (((unsigned long)k1) << 16) | k0; 
}

unsigned long long key3(unsigned long k0, unsigned long k1) {
    if ((k0 >> 16) == (k1 & ((1 << 16) - 1)))
        return (((unsigned long long)k1) << 16) | k0;
    else
        return (unsigned long long)-1;
}

unsigned long long key4(unsigned short k0, unsigned short k1, unsigned short k2, unsigned short k3) {
    return (((unsigned long long)k3) << 48) | (((unsigned long long)k2) << 32) | (((unsigned long)k1) << 16) | k0; 
}

unsigned long long key4(unsigned long long k0, unsigned long long k1) {
    if ((k0 >> 16) == (k1 & ((1ULL << 32) - 1)))
        return (k1 << 16) | k0;
    else
        return (unsigned long long)-1;
}

typedef std::pair< std::map<unsigned short, unsigned long>, std::map<unsigned short, unsigned long> > LR;

class ArgScan2: public std::map<unsigned long, unsigned long> {
    public:
        std::map<unsigned short, unsigned long> *_w1;
        unsigned short *_x;
        unsigned long _i0, _i1;
        unsigned long _threshold;
};

void *threadScan2(void *pArgs) {
    ArgScan2 *myArg =  (ArgScan2 *)pArgs;
    std::map<unsigned short, unsigned long> &w1 = *myArg->_w1;
    std::map<unsigned long, unsigned long> &w2 = *myArg;
    unsigned short *x = myArg->_x;
    unsigned long threshold = myArg->_threshold;
    for (unsigned long i = myArg->_i0; i < myArg->_i1; ++ i) {
        std::map<unsigned short, unsigned long>::iterator loc0 = w1.find(x[i]), loc1 = w1.find(x[i + 1]);
	if (loc0 == w1.end() || loc1 == w1.end())
	    continue;
        if (loc0->second > threshold && loc1->second > threshold) {
            unsigned long key = key2(x[i], x[i + 1]);
            std::map<unsigned long, unsigned long>::iterator loc = w2.find(key);
            if (loc == w2.end()) {
                w2.insert(std::pair<unsigned long, unsigned long>(key, 1));
            } else {
                loc->second += 1;
            }
        }
    }
    return NULL;
}

class ArgBE {
    public:
        unsigned long _kBegin, _kEnd;
        double *_leftEntropy;
        double *_rightEntropy;
        LR *_map;
};

void *threadBE(void *pArgs) {
    ArgBE *myArg = (ArgBE *)pArgs;
    for (unsigned long k = myArg->_kBegin; k < myArg->_kEnd; ++ k) {
        LR &loc = *(myArg->_map + k);
        double leftSum = 0.0;
        double leftEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = loc.first.begin(); 
                loc1 != loc.first.end(); ++ loc1) {
            leftEntropy -= loc1->second * log(loc1->second);
            leftSum += loc1->second;
        }
        if (leftSum > 0)
            leftEntropy = log(leftSum) - leftEntropy / leftSum;

        double rightSum = 0.0;
        double rightEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc2 = loc.second.begin(); 
                loc2 != loc.second.end(); ++ loc2) {
            rightEntropy -= loc2->second * log(loc2->second);
            rightSum += loc2->second;
        }
        if (rightSum)
            rightEntropy = log(rightSum) - rightEntropy / rightSum;
        myArg->_leftEntropy[k] = leftEntropy;
        myArg->_rightEntropy[k] = rightEntropy;
    }
    return NULL;
}

void Apriori::scan2(unsigned long n, unsigned short *x, bool multiThread) {
    int nThreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (n >= 1024 && multiThread) {
        ArgScan2 *args = new ArgScan2[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            args[iThread]._w1 = &_w1;
            args[iThread]._threshold = _threshold;
            args[iThread]._x = x;
            args[iThread]._i0 = iThread * (n - 1) / nThreads;
            args[iThread]._i1 = (iThread + 1) * (n - 1) / nThreads;
            pthread_create(threads + iThread, NULL, threadScan2, args + iThread);
        }
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            for (std::map<unsigned long, unsigned long>::iterator loc = args[iThread].begin(); 
                    loc != args[iThread].end(); ++ loc) {
                std::map<unsigned long, unsigned long>::iterator loc1 = _w2.find(loc->first);
                if (loc1 == _w2.end()) {
                    _w2.insert(std::pair<unsigned long, unsigned long>(loc->first, loc->second));
                } else {
                    loc1->second += loc->second;
                }
            }
        }
        delete[] args;
        delete[] threads;
    } else {
        for (unsigned long i = 0; i + 1 < n; ++ i) {
            std::map<unsigned short, unsigned long>::iterator loc0 = _w1.find(x[i]), loc1 = _w1.find(x[i + 1]);
            if (loc0->second > _threshold && loc1->second > _threshold) {
                unsigned long key = key2(x[i], x[i + 1]);
                std::map<unsigned long, unsigned long>::iterator loc = _w2.find(key);
                if (loc == _w2.end()) {
                    _w2.insert(std::pair<unsigned long, unsigned long>(key, 1));
                } else {
                    loc->second += 1;
                }
            }
        }
    }

    std::map< unsigned short, LR > be;
    for (std::map<unsigned short, unsigned long>::iterator loc = _w1.begin(); loc != _w1.end(); ++ loc) {
        if (loc->second > _threshold) {
            be.insert(std::pair< unsigned short, LR >(loc->first, LR()));
        }
    }
    unsigned long kBE = 0;
    for (std::map<unsigned long, unsigned long>::iterator loc = _w2.begin(); loc != _w2.end(); ++ loc, ++kBE) {
        unsigned short key1 = loc->first & ((1 << 16) - 1);
        unsigned short key2 = loc->first >> 16;
        be[key1].second[key2] = loc->second;
        be[key2].first[key1] = loc->second;
    }

    unsigned long nBE = be.size();
    if (nBE >= 1024 && multiThread) {
        unsigned short *key = new unsigned short[nBE];
        double *leftEntropy = new double[nBE];
        double *rightEntropy = new double[nBE];
        LR *myMap = new LR[nBE];
        std::map<unsigned short, LR>::iterator loc = be.begin();
        for (kBE = 0; kBE < nBE; ++ kBE, ++ loc) {
            key[kBE] = loc->first;
            myMap[kBE] = loc->second;
        }

        ArgBE *args = new ArgBE[nThreads];
        kBE = 0;
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            unsigned long kBegin = iThread * nBE / nThreads;
            unsigned long kEnd = (iThread + 1) * nBE / nThreads;
            args[iThread]._kBegin = kBegin;
            args[iThread]._kEnd = kEnd;
            args[iThread]._map = myMap;
            args[iThread]._leftEntropy = leftEntropy;
            args[iThread]._rightEntropy = rightEntropy;
            pthread_create(threads + iThread, NULL, threadBE, args + iThread);
        }

        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            unsigned long kBegin = args[iThread]._kBegin;
            unsigned long kEnd = args[iThread]._kEnd;
            for (kBE = kBegin; kBE < kEnd; ++kBE) {
                _bel1[key[kBE]] = leftEntropy[kBE];
                _ber1[key[kBE]] = rightEntropy[kBE];
            }
        }
        delete[] args;
        delete[] threads;
        delete[] key;
        delete[] leftEntropy;
        delete[] rightEntropy;
        delete[] myMap;
    } else {
        kBE = 0;
        for (std::map<unsigned short, LR>::iterator loc = be.begin(); loc != be.end(); ++ loc, ++ kBE) {
            double leftSum = 0.0;
            double leftEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc1 = loc->second.first.begin(); 
                    loc1 != loc->second.first.end(); ++ loc1) {
                leftEntropy -= loc1->second * log(loc1->second);
                leftSum += loc1->second;
            }
            if (leftSum > 0)
                leftEntropy = log(leftSum) - leftEntropy / leftSum;

            double rightSum = 0.0;
            double rightEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc2 = loc->second.second.begin(); 
                    loc2 != loc->second.second.end(); ++ loc2) {
                rightEntropy -= loc2->second * log(loc2->second);
                rightSum += loc2->second;
            }
            if (rightSum)
                rightEntropy = log(rightSum) - rightEntropy / rightSum;

            _bel1[loc->first] = leftEntropy;
            _ber1[loc->first] = rightEntropy;
        }
    }
}

void Apriori::scan2(unsigned long n, int pos1, int pos2, unsigned short *x) {
    for (long i = 0; i < n; ++ i) {
        if (i + pos1 < 0 || i + pos1 >= n || i + pos2 < 0 || i + pos2 >= n)
            continue;
        unsigned long key = key2(x[i + pos1], x[i + pos2]);
        std::map<unsigned long, unsigned long>::iterator loc = _w2.find(key);
        if (loc == _w2.end()) {
            _w2.insert(std::pair<unsigned long, unsigned long>(key, 1));
        } else {
            loc->second += 1;
        }
    }

    unsigned long index = 0;
    for (std::map<unsigned long, unsigned long>::iterator loc = _w2.begin(); loc != _w2.end(); ++ loc) {
        if (loc->second <= _threshold)
            continue;
        _i2[loc->first] = index;
        ++ index;
    }
}

class ArgScan3: public std::map<unsigned long long, unsigned long> {
    public:
        std::map<unsigned long, unsigned long> *_w2;
        unsigned short *_x;
        unsigned long _i0, _i1;
        unsigned long _threshold;
};

void *threadScan3(void *pArgs) {
    ArgScan3 *myArg =  (ArgScan3 *)pArgs;
    std::map<unsigned long, unsigned long> &w2 = *myArg->_w2;
    std::map<unsigned long long, unsigned long> &w3 = *myArg;
    unsigned short *x = myArg->_x;
    unsigned long threshold = myArg->_threshold;
    for (unsigned long i = myArg->_i0; i < myArg->_i1; ++ i) {
        std::map<unsigned long, unsigned long>::iterator loc0 = w2.find(key2(x[i],x[i + 1])), 
            loc1 = w2.find(key2(x[i + 1], x[i + 2]));
	if (loc0 == w2.end() || loc1 == w2.end())
	    continue;
        if (loc0->second > threshold && loc1->second > threshold) {
            unsigned long long key = key3(x[i], x[i + 1], x[i + 2]);
            std::map<unsigned long long, unsigned long>::iterator loc = w3.find(key);
            if (loc == w3.end()) {
                w3.insert(std::pair<unsigned long long, unsigned long>(key, 1));
            } else {
                loc->second += 1;
            }
        }
    }
    return NULL;
}

class ArgLR2: public std::map< unsigned long, LR > {
    public:
        unsigned long long *_key;
        unsigned long *_value;
        unsigned long _kBegin, _kEnd;
};

void *threadLR2(void *pArgs) {
    ArgLR2 *myArg = (ArgLR2 *)pArgs;
    unsigned long long *key = myArg->_key;
    unsigned long *value = myArg->_value;
    unsigned long kBegin = myArg->_kBegin;
    unsigned long kEnd = myArg->_kEnd;
    for (unsigned long k = kBegin; k < kEnd; ++ k) {
        unsigned long key1 = key[k] & ((1ULL << 32) - 1);
        unsigned long key2 = key[k] >> 16;

        std::map< unsigned long, LR >::iterator loc1 = myArg->find(key1);
        if (loc1 == myArg->end()) {
            myArg->insert(std::pair< unsigned long, LR >(key1, LR()));
            loc1 = myArg->find(key1);
        }
        loc1->second.first[key2 >> 16] = value[k];

        std::map< unsigned long, LR >::iterator loc2 = myArg->find(key2);
        if (loc2 == myArg->end()) {
            myArg->insert(std::pair< unsigned long, LR >(key2, LR()));
            loc2 = myArg->find(key1);
        }
        loc2->second.second[key1 & ((1 << 16) - 1)] = value[k];
    }
    return NULL;
}

void Apriori::scan3(unsigned long n, unsigned short *x, bool multiThread) {
    int nThreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (n >= 1024 && multiThread) {
        ArgScan3 *args = new ArgScan3[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            args[iThread]._w2 = &_w2;
            args[iThread]._threshold = _threshold;
            args[iThread]._x = x;
            args[iThread]._i0 = iThread * (n - 2) / nThreads;
            args[iThread]._i1 = (iThread + 1) * (n - 2) / nThreads;
            pthread_create(threads + iThread, NULL, threadScan3, args + iThread);
        }
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            for (std::map<unsigned long long, unsigned long>::iterator loc = args[iThread].begin(); 
                    loc != args[iThread].end(); ++ loc) {
                std::map<unsigned long long, unsigned long>::iterator loc1 = _w3.find(loc->first);
                if (loc1 == _w3.end()) {
                    _w3.insert(std::pair<unsigned long long, unsigned long>(loc->first, loc->second));
                } else {
                    loc1->second += loc->second;
                }
            }
        }
        delete[] args;
        delete[] threads;
    } else {
        for (unsigned long i = 0; i + 2 < n; ++ i) {
            std::map<unsigned long, unsigned long>::iterator loc0 = _w2.find(key2(x[i],x[i + 1])), 
                loc1 = _w2.find(key2(x[i + 1], x[i + 2]));
	    if (loc0 == _w2.end() || loc1 == _w2.end())
		continue;
            if (loc0->second > _threshold && loc1->second > _threshold) {
                unsigned long long key = key3(x[i], x[i + 1], x[i + 2]);
                std::map<unsigned long long, unsigned long>::iterator loc = _w3.find(key);
                if (loc == _w3.end()) {
                    _w3.insert(std::pair<unsigned long long, unsigned long>(key, 1));
                } else {
                    loc->second += 1;
                }
            }
        }
    }
    std::map< unsigned long, LR > be;
    for (std::map<unsigned long, unsigned long>::iterator loc = _w2.begin(); loc != _w2.end(); ++ loc) {
        if (loc->second > _threshold) {
            be.insert(std::pair< unsigned long, LR >(loc->first, LR()));
        }
    }
    unsigned long nW3 = _w3.size();
    if (nW3 >= 1024 && multiThread && false) {
        unsigned long long *key = new unsigned long long[nW3];
        unsigned long *value = new unsigned long[nW3];
        unsigned long k = 0;
        for (std::map<unsigned long long, unsigned long>::iterator loc = _w3.begin(); loc != _w3.end(); ++ loc, ++k) {
            key[k] = loc->first;
            value[k] = loc->second;
        }

        ArgLR2 *args = new ArgLR2[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            args[iThread]._key = key;
            args[iThread]._value = value;
            args[iThread]._kBegin = iThread * nW3 / nThreads;
            args[iThread]._kEnd = (iThread + 1) * nW3 / nThreads;
            pthread_create(threads + iThread, NULL, threadLR2, args + iThread);
        }
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            for (std::map<unsigned long, LR>::iterator loc = args[iThread].begin(); 
                    loc != args[iThread].end(); ++ loc) {
                std::map<unsigned long, LR>::iterator loc1 = be.find(loc->first);
                for (std::map<unsigned short, unsigned long>::iterator iter = loc->second.first.begin();
                        iter != loc->second.first.end(); ++ iter) {
                    loc1->second.first.insert(std::pair<unsigned short, unsigned long>(iter->first, iter->second));
                }
                for (std::map<unsigned short, unsigned long>::iterator iter = loc->second.second.begin();
                        iter != loc->second.second.end(); ++ iter) {
                    loc1->second.second.insert(std::pair<unsigned short, unsigned long>(iter->first, iter->second));
                }
            }
        }
        delete[] args;
        delete[] threads;
    } else {
        for (std::map<unsigned long long, unsigned long>::iterator loc = _w3.begin(); loc != _w3.end(); ++ loc) {
            unsigned long key1 = loc->first & ((1ULL << 32) - 1);
            unsigned long key2 = loc->first >> 16;
            be[key1].second[key2 >> 16] = loc->second;
            be[key2].first[key1 & ((1 << 16) - 1)] = loc->second;
        }
    }

    unsigned long nBE = be.size();
    if (nBE >= 1024 && multiThread) {
        unsigned long *key = new unsigned long[nBE];
        double *leftEntropy = new double[nBE];
        double *rightEntropy = new double[nBE];
        LR *myMap = new LR[nBE];
        std::map<unsigned long, LR>::iterator loc = be.begin();
        for (unsigned long kBE = 0; kBE < nBE; ++ kBE, ++ loc) {
            key[kBE] = loc->first;
            myMap[kBE] = loc->second;
        }

        ArgBE *args = new ArgBE[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            unsigned long kBegin = iThread * nBE / nThreads;
            unsigned long kEnd = (iThread + 1) * nBE / nThreads;
            args[iThread]._kBegin = kBegin;
            args[iThread]._kEnd = kEnd;
            args[iThread]._map = myMap;
            args[iThread]._leftEntropy = leftEntropy;
            args[iThread]._rightEntropy = rightEntropy;
            pthread_create(threads + iThread, NULL, threadBE, args + iThread);
        }

        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            unsigned long kBegin = args[iThread]._kBegin;
            unsigned long kEnd = args[iThread]._kEnd;
            for (unsigned long kBE = kBegin; kBE < kEnd; ++kBE) {
                _bel2[key[kBE]] = leftEntropy[kBE];
                _ber2[key[kBE]] = rightEntropy[kBE];
            }
        }
        delete[] args;
        delete[] threads;
        delete[] key;
        delete[] leftEntropy;
        delete[] rightEntropy;
        delete[] myMap;
    } else {
        for (std::map<unsigned long, LR>::iterator loc = be.begin(); loc != be.end(); ++ loc) {
            double leftSum = 0.0;
            double leftEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc1 = loc->second.first.begin(); 
                    loc1 != loc->second.first.end(); ++ loc1) {
                leftEntropy -= loc1->second * log(loc1->second);
                leftSum += loc1->second;
            }
            if (leftSum > 0)
                leftEntropy = log(leftSum) - leftEntropy / leftSum;

            double rightSum = 0.0;
            double rightEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc2 = loc->second.second.begin(); 
                    loc2 != loc->second.second.end(); ++ loc2) {
                rightEntropy -= loc2->second * log(loc2->second);
                rightSum += loc2->second;
            }
            if (rightSum > 0)
                rightEntropy = log(rightSum) - rightEntropy / rightSum;

            _bel2[loc->first] = leftEntropy;
            _ber2[loc->first] = rightEntropy;
        }
    }
}

void Apriori::scan3(unsigned long n, int pos1, int pos2, int pos3, unsigned short *x) {
    for (long i = 0; i < n; ++ i) {
        if (i + pos1 < 0 || i + pos1 >= n || i + pos2 < 0 || i + pos2 >= n || i + pos3 < 0 || i + pos3 >= n)
            continue;
        unsigned long long key = key3(x[i + pos1], x[i + pos2], x[i + pos3]);
        std::map<unsigned long long, unsigned long>::iterator loc = _w3.find(key);
        if (loc == _w3.end()) {
            _w3.insert(std::pair<unsigned long long, unsigned long>(key, 1));
        } else {
            loc->second += 1;
        }
    }

    unsigned long index = 0;
    for (std::map<unsigned long long, unsigned long>::iterator loc = _w3.begin(); loc != _w3.end(); ++ loc) {
        if (loc->second <= _threshold)
            continue;
        _i3[loc->first] = index;
        ++ index;
    }
}

class ArgScan4: public std::map<unsigned long long, unsigned long> {
    public:
        std::map<unsigned long long, unsigned long> *_w3;
        unsigned short *_x;
        unsigned long _i0, _i1;
        unsigned long _threshold;
};

void *threadScan4(void *pArgs) {
    ArgScan4 *myArg =  (ArgScan4 *)pArgs;
    std::map<unsigned long long, unsigned long> &w3 = *myArg->_w3;
    std::map<unsigned long long, unsigned long> &w4 = *myArg;
    unsigned short *x = myArg->_x;
    unsigned long threshold = myArg->_threshold;
    for (unsigned long i = myArg->_i0; i < myArg->_i1; ++ i) {
        std::map<unsigned long long, unsigned long>::iterator loc0 = w3.find(key3(x[i], x[i + 1], x[i + 2])), 
            loc1 = w3.find(key3(x[i + 1], x[i + 2], x[i + 3]));
	if (loc0 == w3.end() || loc1 == w3.end())
	    continue;
        if (loc0->second > threshold && loc1->second > threshold) {
            unsigned long long key = key4(x[i], x[i + 1], x[i + 2], x[i + 3]);
            std::map<unsigned long long, unsigned long>::iterator loc = w4.find(key);
            if (loc == w4.end()) {
                w4.insert(std::pair<unsigned long long, unsigned long>(key, 1));
            } else {
                loc->second += 1;
            }
        }
    }
    return NULL;
}

class ArgLR3: public std::map< unsigned long long, LR > {
    public:
        unsigned long long *_key;
        unsigned long *_value;
        unsigned long _kBegin, _kEnd;
};

void *threadLR3(void *pArgs) {
    ArgLR3 *myArg = (ArgLR3 *)pArgs;
    unsigned long long *key = myArg->_key;
    unsigned long *value = myArg->_value;
    unsigned long kBegin = myArg->_kBegin;
    unsigned long kEnd = myArg->_kEnd;
    for (unsigned long k = kBegin; k < kEnd; ++ k) {
        unsigned long long key1 = key[k] & ((1ULL << 48) - 1);
        unsigned long long key2 = key[k] >> 16;

        std::map< unsigned long long, LR >::iterator loc1 = myArg->find(key1);
        if (loc1 == myArg->end()) {
            myArg->insert(std::pair< unsigned long long, LR >(key1, LR()));
            loc1 = myArg->find(key1);
        }
        loc1->second.first[key2 >> 16] = value[k];

        std::map< unsigned long long, LR >::iterator loc2 = myArg->find(key2);
        if (loc2 == myArg->end()) {
            myArg->insert(std::pair< unsigned long long, LR >(key2, LR()));
            loc2 = myArg->find(key1);
        }
        loc2->second.second[key1 & ((1 << 16) - 1)] = value[k];
    }
    return NULL;
}

void Apriori::scan4(unsigned long n, unsigned short *x, bool multiThread) {
    int nThreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (n >= 1024 && multiThread) {
        ArgScan4 *args = new ArgScan4[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            args[iThread]._w3 = &_w3;
            args[iThread]._threshold = _threshold;
            args[iThread]._x = x;
            args[iThread]._i0 = iThread * (n - 3) / nThreads;
            args[iThread]._i1 = (iThread + 1) * (n - 3) / nThreads;
            pthread_create(threads + iThread, NULL, threadScan4, args + iThread);
        }
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            for (std::map<unsigned long long, unsigned long>::iterator loc = args[iThread].begin(); 
                    loc != args[iThread].end(); ++ loc) {
                std::map<unsigned long long, unsigned long>::iterator loc1 = _w4.find(loc->first);
                if (loc1 == _w4.end()) {
                    _w4.insert(std::pair<unsigned long, unsigned long>(loc->first, loc->second));
                } else {
                    loc1->second += loc->second;
                }
            }
        }
        delete[] args;
        delete[] threads;
    } else {
        for (unsigned long i = 0; i + 3 < n; ++ i) {
            std::map<unsigned long long, unsigned long>::iterator loc0 = _w3.find(key3(x[i], x[i + 1], x[i + 2])), 
                loc1 = _w3.find(key3(x[i + 1], x[i + 2], x[i + 3]));
	    if (loc0 == _w3.end() || loc1 == _w3.end())
		continue;
            if (loc0->second > _threshold && loc1->second > _threshold) {
                unsigned long long key = key4(x[i], x[i + 1], x[i + 2], x[i + 3]);
                std::map<unsigned long long, unsigned long>::iterator loc = _w4.find(key);
                if (loc == _w4.end()) {
                    _w4.insert(std::pair<unsigned long long, unsigned long>(key, 1));
                } else {
                    loc->second += 1;
                }
            }
        }
    }
    std::map< unsigned long long, LR > be;
    for (std::map<unsigned long long, unsigned long>::iterator loc = _w3.begin(); loc != _w3.end(); ++ loc) {
        if (loc->second > _threshold) {
            be.insert(std::pair< unsigned long long, LR >(loc->first, LR()));
        }
    }

    unsigned long nW4 = _w4.size();
    if (nW4 >= 1024 && multiThread && false) {
        unsigned long long *key = new unsigned long long[nW4];
        unsigned long *value = new unsigned long[nW4];
        unsigned long k = 0;
        for (std::map<unsigned long long, unsigned long>::iterator loc = _w4.begin(); loc != _w4.end(); ++ loc, ++k) {
            key[k] = loc->first;
            value[k] = loc->second;
        }

        ArgLR3 *args = new ArgLR3[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            args[iThread]._key = key;
            args[iThread]._value = value;
            args[iThread]._kBegin = iThread * nW4 / nThreads;
            args[iThread]._kEnd = (iThread + 1) * nW4 / nThreads;
            pthread_create(threads + iThread, NULL, threadLR3, args + iThread);
        }
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            for (std::map<unsigned long long, LR>::iterator loc = args[iThread].begin(); 
                    loc != args[iThread].end(); ++ loc) {
                std::map<unsigned long long, LR>::iterator loc1 = be.find(loc->first);
                for (std::map<unsigned short, unsigned long>::iterator iter = loc->second.first.begin();
                        iter != loc->second.first.end(); ++ iter) {
                    loc1->second.first.insert(std::pair<unsigned short, unsigned long>(iter->first, iter->second));
                }
                for (std::map<unsigned short, unsigned long>::iterator iter = loc->second.second.begin();
                        iter != loc->second.second.end(); ++ iter) {
                    loc1->second.second.insert(std::pair<unsigned short, unsigned long>(iter->first, iter->second));
                }
            }
        }
        delete[] args;
        delete[] threads;
    } else {
        for (std::map<unsigned long long, unsigned long>::iterator loc = _w4.begin(); loc != _w4.end(); ++ loc) {
            unsigned long key1 = loc->first & ((1ULL << 48) - 1);
            unsigned long key2 = loc->first >> 16;
            be[key1].second[key2 >> 32] = loc->second;
            be[key2].first[key1 & ((1 << 16) - 1)] = loc->second;
        }
    }

    unsigned long nBE = be.size();
    if (nBE >= 1024 && multiThread) {
        unsigned long long *key = new unsigned long long[nBE];
        double *leftEntropy = new double[nBE];
        double *rightEntropy = new double[nBE];
        LR *myMap = new LR[nBE];
        std::map<unsigned long long, LR>::iterator loc = be.begin();
        for (unsigned long kBE = 0; kBE < nBE; ++ kBE, ++ loc) {
            key[kBE] = loc->first;
            myMap[kBE] = loc->second;
        }

        ArgBE *args = new ArgBE[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            unsigned long kBegin = iThread * nBE / nThreads;
            unsigned long kEnd = (iThread + 1) * nBE / nThreads;
            args[iThread]._kBegin = kBegin;
            args[iThread]._kEnd = kEnd;
            args[iThread]._map = myMap;
            args[iThread]._leftEntropy = leftEntropy;
            args[iThread]._rightEntropy = rightEntropy;
            pthread_create(threads + iThread, NULL, threadBE, args + iThread);
        }

        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            unsigned long kBegin = args[iThread]._kBegin;
            unsigned long kEnd = args[iThread]._kEnd;
            for (unsigned long kBE = kBegin; kBE < kEnd; ++kBE) {
                _bel3[key[kBE]] = leftEntropy[kBE];
                _ber3[key[kBE]] = rightEntropy[kBE];
            }
        }
        delete[] args;
        delete[] threads;
        delete[] key;
        delete[] leftEntropy;
        delete[] rightEntropy;
        delete[] myMap;
    } else {
        for (std::map<unsigned long long, LR>::iterator loc = be.begin(); loc != be.end(); ++ loc) {
            double leftSum = 0.0;
            double leftEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc1 = loc->second.first.begin(); 
                    loc1 != loc->second.first.end(); ++ loc1) {
                leftEntropy -= loc1->second * log(loc1->second);
                leftSum += loc1->second;
            }
            if (leftSum > 0)
                leftEntropy = log(leftSum) - leftEntropy / leftSum;

            double rightSum = 0.0;
            double rightEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc2 = loc->second.second.begin(); 
                    loc2 != loc->second.second.end(); ++ loc2) {
                rightEntropy -= loc2->second * log(loc2->second);
                rightSum += loc2->second;
            }
            if (rightSum > 0)
                rightEntropy = log(rightSum) - rightEntropy / rightSum;

            _bel3[loc->first] = leftEntropy;
            _ber3[loc->first] = rightEntropy;

        }
    }

    // bel4, ber4
    be.clear();
    for (std::map<unsigned long long, unsigned long>::iterator loc = _w4.begin(); loc != _w4.end(); ++ loc) {
        be.insert(std::pair< unsigned long long, LR >(loc->first, LR()));
    }
    for (long i = 0; i + 3 < n; ++ i) {
        unsigned long long key = key4(x[i], x[i + 1], x[i + 2], x[i + 3]);
        std::map< unsigned long long, LR >::iterator loc = be.find(key);
        if (loc == be.end())
            continue;
        if (i > 0) {
            std::map< unsigned short, unsigned long >::iterator loc1 = loc->second.first.find(x[i - 1]);
            if (loc1 != loc->second.first.end()) {
                ++ loc1->second;
            } else {
                loc->second.first[x[i - 1]] = 1;
            }
        }
        if (i + 4 < n) {
            std::map< unsigned short, unsigned long >::iterator loc1 = loc->second.second.find(x[i + 4]);
            if (loc1 != loc->second.second.end()) {
                ++ loc1->second;
            } else {
                loc->second.second[x[i - 1]] = 1;
            }
        }
    }
    nBE = be.size();
    if (nBE >= 1024 && multiThread) {
        unsigned long long *key = new unsigned long long[nBE];
        double *leftEntropy = new double[nBE];
        double *rightEntropy = new double[nBE];
        LR *myMap = new LR[nBE];
        std::map<unsigned long long, LR>::iterator loc = be.begin();
        for (unsigned long kBE = 0; kBE < nBE; ++ kBE, ++ loc) {
            key[kBE] = loc->first;
            myMap[kBE] = loc->second;
        }

        ArgBE *args = new ArgBE[nThreads];
        pthread_t *threads = new pthread_t[nThreads];
        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            unsigned long kBegin = iThread * nBE / nThreads;
            unsigned long kEnd = (iThread + 1) * nBE / nThreads;
            args[iThread]._kBegin = kBegin;
            args[iThread]._kEnd = kEnd;
            args[iThread]._map = myMap;
            args[iThread]._leftEntropy = leftEntropy;
            args[iThread]._rightEntropy = rightEntropy;
            pthread_create(threads + iThread, NULL, threadBE, args + iThread);
        }

        for (int iThread = 0; iThread < nThreads; ++ iThread) {
            pthread_join(threads[iThread], NULL);
            unsigned long kBegin = args[iThread]._kBegin;
            unsigned long kEnd = args[iThread]._kEnd;
            for (unsigned long kBE = kBegin; kBE < kEnd; ++kBE) {
                _bel4[key[kBE]] = leftEntropy[kBE];
                _ber4[key[kBE]] = rightEntropy[kBE];
            }
        }
        delete[] args;
        delete[] threads;
        delete[] key;
        delete[] leftEntropy;
        delete[] rightEntropy;
        delete[] myMap;
    } else {
        for (std::map<unsigned long long, LR>::iterator loc = be.begin(); loc != be.end(); ++ loc) {
            double leftSum = 0.0;
            double leftEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc1 = loc->second.first.begin(); 
                    loc1 != loc->second.first.end(); ++ loc1) {
                leftEntropy -= loc1->second * log(loc1->second);
                leftSum += loc1->second;
            }
            leftEntropy = log(leftSum) - leftEntropy / leftSum;

            double rightSum = 0.0;
            double rightEntropy = 0.0;
            for (std::map<unsigned short, unsigned long>::iterator loc2 = loc->second.second.begin(); 
                    loc2 != loc->second.second.end(); ++ loc2) {
                rightEntropy -= loc2->second * log(loc2->second);
                rightSum += loc2->second;
            }
            rightEntropy = log(rightSum) - rightEntropy / rightSum;

            _bel4[loc->first] = leftEntropy;
            _ber4[loc->first] = rightEntropy;
        }
    }
}

void Apriori::scan4(unsigned long n, int pos1, int pos2, int pos3, int pos4, unsigned short *x) {
    for (long i = 0; i < n; ++ i) {
        if (i + pos1 < 0 || i + pos1 >= n || i + pos2 < 0 || i + pos2 >= n || i + pos3 < 0 || i + pos3 >= n || i + pos4 < 0 || i + pos4 >= n)
            continue;
        unsigned long long key = key4(x[i + pos1], x[i + pos2], x[i + pos3], x[i + pos4]);
        std::map<unsigned long long, unsigned long>::iterator loc = _w4.find(key);
        if (loc == _w4.end()) {
            _w4.insert(std::pair<unsigned long long, unsigned long>(key, 1));
        } else {
            loc->second += 1;
        }
    }

    unsigned long index = 0;
    for (std::map<unsigned long long, unsigned long>::iterator loc = _w4.begin(); loc != _w4.end(); ++ loc) {
        if (loc->second <= _threshold)
            continue;
        _i4[loc->first] = index;
        ++ index;
    }
}

void Apriori::scan(unsigned long n, unsigned short *x, bool multiThread) {
    scan1(n, x, multiThread);
    scan2(n, x, multiThread);
    scan3(n, x, multiThread);
    scan4(n, x, multiThread);
}

void Apriori::scan(unsigned long n, int pos1, int pos2, unsigned short *x) {
    scan2(n, pos1, pos2, x);
}

void Apriori::scan(unsigned long n, int pos1, int pos2, int pos3, unsigned short *x) {
    scan3(n, pos1, pos2, pos3, x);
}

void Apriori::scan(unsigned long n, int pos1, int pos2, int pos3, int pos4, unsigned short *x) {
    scan4(n, pos1, pos2, pos3, pos4, x);
}

void Apriori::getFreq(unsigned long n, unsigned short *x,
                        int *freq1, int *freq2, int *freq3, int *freq4,
                        double *bel1, double *ber1, double *bel2, double *ber2,
                        double *bel3, double *ber3, double *bel4, double *ber4) {
    for (long i = 0; i < n; ++ i) {
        std::map<unsigned short, unsigned long>::iterator loc1 = _w1.find(x[i]);
        if (loc1 != _w1.end()) {
            freq1[i] = (int) loc1->second;
        } else {
            freq1[i] = 0;
        }
        std::map<unsigned short, double>::iterator loc1L = _bel1.find(x[i]);
        if (loc1L != _bel1.end()) {
            bel1[i] = loc1L->second;
        } else {
            bel1[i] = 0.0;
        }
        std::map<unsigned short, double>::iterator loc1R = _ber1.find(x[i]);
        if (loc1R != _ber1.end()) {
            ber1[i] = loc1R->second;
        } else {
            ber1[i] = 0.0;
        }

        if (i + 1 >= n)
            continue;
        std::map<unsigned long, unsigned long>::iterator loc2 = _w2.find(key2(x[i], x[i + 1]));
        if (loc2 != _w2.end()) {
            freq2[i] = (int) loc2->second;
        } else {
            freq2[i] = 0;
        }
        std::map<unsigned long, double>::iterator loc2L = _bel2.find(key2(x[i], x[i + 1]));
        if (loc2L != _bel2.end()) {
            bel2[i] = loc2L->second;
        } else {
            bel2[i] = 0.0;
        }
        std::map<unsigned long, double>::iterator loc2R = _ber2.find(key2(x[i], x[i + 1]));
        if (loc2R != _ber2.end()) {
            ber2[i] = loc2R->second;
        } else {
            ber2[i] = 0.0;
        }

        if (i + 2 >= n)
            continue;
        std::map<unsigned long long, unsigned long>::iterator loc3 = _w3.find(key3(x[i], x[i + 1], x[i + 2]));
        if (loc3 != _w3.end()) {
            freq3[i] = (int) loc3->second;
        } else {
            freq3[i] = 0;
        }
        std::map<unsigned long long, double>::iterator loc3L = _bel3.find(key3(x[i], x[i + 1], x[i + 2]));
        if (loc3L != _bel3.end()) {
            bel3[i] = loc3L->second;
        } else {
            bel3[i] = 0.0;
        }
        std::map<unsigned long long, double>::iterator loc3R = _ber3.find(key3(x[i], x[i + 1], x[i + 2]));
        if (loc3R != _ber3.end()) {
            ber3[i] = loc3R->second;
        } else {
            ber3[i] = 0.0;
        }

        if (i + 3 >= n)
            continue;
        std::map<unsigned long long, unsigned long>::iterator loc4 = _w4.find(key4(x[i], x[i + 1], x[i + 2], x[i + 3]));
        if (loc4 != _w4.end()) {
            freq4[i] = (int) loc4->second;
        } else {
            freq4[i] = 0;
        }
        std::map<unsigned long long, double>::iterator loc4L = _bel4.find(key4(x[i], x[i + 1], x[i + 2], x[i + 3]));
        if (loc4L != _bel4.end()) {
            bel4[i] = loc4L->second;
        } else {
            bel4[i] = 0.0;
        }
        std::map<unsigned long long, double>::iterator loc4R = _ber4.find(key4(x[i], x[i + 1], x[i + 2], x[i + 3]));
        if (loc4R != _ber4.end()) {
            ber4[i] = loc4R->second;
        } else {
            ber4[i] = 0.0;
        }
    }
}

int Apriori::getIndex2(unsigned long n, int pos1, int pos2, unsigned short *x, int start, int *index) {
    for (long i = 0; i < n; ++ i) {
        if (i + pos1 < 0 || i + pos1 >= n || i + pos2 < 0 || i + pos2 >= n) {
            index[i] = 0;
            continue;
        }
        unsigned long key = key2(x[i + pos1], x[i + pos2]);
        std::map<unsigned long, unsigned long>::iterator loc = _i2.find(key);
        if (loc != _i2.end()) {
            index[i] = start + loc->second;
        } else {
            index[i] = 0;
        }
    }
    return start + _i2.size();
}

int Apriori::getIndex3(unsigned long n, int pos1, int pos2, int pos3, unsigned short *x, int start, int *index) {
    for (long i = 0; i < n; ++ i) {
        if (i + pos1 < 0 || i + pos1 >= n || i + pos2 < 0 || i + pos2 >= n || i + pos3 < 0 || i + pos3 >= n) {
            index[i] = 0;
            continue;
        }
        unsigned long long key = key3(x[i + pos1], x[i + pos2], x[i + pos3]);
        std::map<unsigned long long, unsigned long>::iterator loc = _i3.find(key);
        if (loc != _i3.end()) {
            index[i] = start + loc->second;
        } else {
            index[i] = 0;
        }
    }
    return start + _i3.size();
}

int Apriori::getIndex4(unsigned long n, int pos1, int pos2, int pos3, int pos4, unsigned short *x, int start, int *index) {
    for (long i = 0; i < n; ++ i) {
        if (i + pos1 < 0 || i + pos1 >= n || i + pos2 < 0 || i + pos2 >= n || i + pos3 < 0 || i + pos3 >= n || i + pos4 < 0 || i + pos4 >= n) {
            index[i] = 0;
            continue;
        }
        unsigned long long key = key4(x[i + pos1], x[i + pos2], x[i + pos3], x[i + pos4]);
        std::map<unsigned long long, unsigned long>::iterator loc = _i4.find(key);
        if (loc != _i4.end()) {
            index[i] = start + loc->second;
        } else {
            index[i] = 0;
        }
    }
    return start + _i4.size();
}

void Apriori::saveToFile(char *fileName) {
    //std::cout << "saveToFile begin" << std::endl;
    FILE *f = fopen(fileName, "w");
    assert(1 == fwrite(&_threshold, sizeof(_threshold), 1, f));

    unsigned long n[15]; 
    n[0] = _w1.size();
    n[1] = _w2.size();
    n[2] = _w3.size();
    n[3] = _w4.size();
    n[4] = _bel1.size();
    n[5] = _ber1.size();
    n[6] = _bel2.size();
    n[7] = _ber2.size();
    n[8] = _bel3.size();
    n[9] = _ber3.size();
    n[10] = _bel4.size();
    n[11] = _ber4.size();
    n[12] = _i2.size();
    n[13] = _i3.size();
    n[14] = _i4.size();
    assert(15 == fwrite(n, sizeof(n[0]), 15, f));
    //for (unsigned int i = 0; i < 15; ++ i) {
        //printf("%lu ", n[i]);
    //}
    //printf("\n");

    if (n[0] > 0) {
        unsigned short *keyW1 = new unsigned short[n[0]];
        unsigned long *valW1 = new unsigned long[n[0]];
        unsigned long k = 0;
        for (std::map<unsigned short, unsigned long>::iterator i1 = _w1.begin(); i1 != _w1.end(); ++ i1) {
            keyW1[k] = i1->first;
            valW1[k] = i1->second;
            ++ k;
        }
        assert(n[0] == fwrite(keyW1, sizeof(keyW1[0]), n[0], f));
        assert(n[0] == fwrite(valW1, sizeof(valW1[0]), n[0], f));
        delete[] keyW1;
        delete[] valW1;
    }

    if (n[1] > 0) {
        unsigned long *keyW2 = new unsigned long[n[1]];
        unsigned long *valW2 = new unsigned long[n[1]];
        unsigned long k = 0;
        for (std::map<unsigned long, unsigned long>::iterator i2 = _w2.begin(); i2 != _w2.end(); ++ i2) {
            keyW2[k] = i2->first;
            valW2[k] = i2->second;
            ++ k;
        }
        assert(n[1] == fwrite(keyW2, sizeof(keyW2[0]), n[1], f));
        assert(n[1] == fwrite(valW2, sizeof(valW2[0]), n[1], f));
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[2] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[2]];
        unsigned long *valW3 = new unsigned long[n[2]];
        unsigned long k = 0;
        for (std::map<unsigned long long, unsigned long>::iterator i3 = _w3.begin(); i3 != _w3.end(); ++ i3) {
            keyW3[k] = i3->first;
            valW3[k] = i3->second;
            ++ k;
        }
        assert(n[2] == fwrite(keyW3, sizeof(keyW3[0]), n[2], f));
        assert(n[2] == fwrite(valW3, sizeof(valW3[0]), n[2], f));
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[3] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[3]];
        unsigned long *valW4 = new unsigned long[n[3]];
        unsigned long k = 0;
        for (std::map<unsigned long long, unsigned long>::iterator i4 = _w4.begin(); i4 != _w4.end(); ++ i4) {
            keyW4[k] = i4->first;
            valW4[k] = i4->second;
            ++ k;
        }
        assert(n[3] == fwrite(keyW4, sizeof(keyW4[0]), n[3], f));
        assert(n[3] == fwrite(valW4, sizeof(valW4[0]), n[3], f));
        delete[] keyW4;
        delete[] valW4;
    }

    if (n[4] > 0) {
        unsigned short *keyW1 = new unsigned short[n[4]];
        double *valW1 = new double[n[4]];
        unsigned long k = 0;
        for (std::map<unsigned short, double>::iterator i1 = _bel1.begin(); i1 != _bel1.end(); ++ i1) {
            keyW1[k] = i1->first;
            valW1[k] = i1->second;
            ++ k;
        }
        assert(n[4] == fwrite(keyW1, sizeof(keyW1[0]), n[4], f));
        assert(n[4] == fwrite(valW1, sizeof(valW1[0]), n[4], f));
        delete[] keyW1;
        delete[] valW1;
    }

    if (n[5] > 0) {
        unsigned short *keyW1 = new unsigned short[n[5]];
        double *valW1 = new double[n[5]];
        unsigned long k = 0;
        for (std::map<unsigned short, double>::iterator i1 = _ber1.begin(); i1 != _ber1.end(); ++ i1) {
            keyW1[k] = i1->first;
            valW1[k] = i1->second;
            ++ k;
        }
        assert(n[5] == fwrite(keyW1, sizeof(keyW1[0]), n[5], f));
        assert(n[5] == fwrite(valW1, sizeof(valW1[0]), n[5], f));
        delete[] keyW1;
        delete[] valW1;
    }

    if (n[6] > 0) {
        unsigned long *keyW2 = new unsigned long[n[6]];
        double *valW2 = new double[n[6]];
        unsigned long k = 0;
        for (std::map<unsigned long, double>::iterator i2 = _bel2.begin(); i2 != _bel2.end(); ++ i2) {
            keyW2[k] = i2->first;
            valW2[k] = i2->second;
            ++ k;
        }
        assert(n[6] == fwrite(keyW2, sizeof(keyW2[0]), n[6], f));
        assert(n[6] == fwrite(valW2, sizeof(valW2[0]), n[6], f));
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[7] > 0) {
        unsigned long *keyW2 = new unsigned long[n[7]];
        double *valW2 = new double[n[7]];
        unsigned long k = 0;
        for (std::map<unsigned long, double>::iterator i2 = _ber2.begin(); i2 != _ber2.end(); ++ i2) {
            keyW2[k] = i2->first;
            valW2[k] = i2->second;
            ++ k;
        }
        assert(n[7] == fwrite(keyW2, sizeof(keyW2[0]), n[7], f));
        assert(n[7] == fwrite(valW2, sizeof(valW2[0]), n[7], f));
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[8] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[8]];
        double *valW3 = new double[n[8]];
        unsigned long k = 0;
        for (std::map<unsigned long long, double>::iterator i3 = _bel3.begin(); i3 != _bel3.end(); ++ i3) {
            keyW3[k] = i3->first;
            valW3[k] = i3->second;
            ++ k;
        }
        assert(n[8] == fwrite(keyW3, sizeof(keyW3[0]), n[8], f));
        assert(n[8] == fwrite(valW3, sizeof(valW3[0]), n[8], f));
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[9] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[9]];
        double *valW3 = new double[n[9]];
        unsigned long k = 0;
        for (std::map<unsigned long long, double>::iterator i3 = _ber3.begin(); i3 != _ber3.end(); ++ i3) {
            keyW3[k] = i3->first;
            valW3[k] = i3->second;
            ++ k;
        }
        assert(n[9] == fwrite(keyW3, sizeof(keyW3[0]), n[9], f));
        assert(n[9] == fwrite(valW3, sizeof(valW3[0]), n[9], f));
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[10] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[10]];
        double *valW4 = new double[n[10]];
        unsigned long k = 0;
        for (std::map<unsigned long long, double>::iterator i4 = _bel4.begin(); i4 != _bel4.end(); ++ i4) {
            keyW4[k] = i4->first;
            valW4[k] = i4->second;
            ++ k;
        }
        assert(n[10] == fwrite(keyW4, sizeof(keyW4[0]), n[10], f));
        assert(n[10] == fwrite(valW4, sizeof(valW4[0]), n[10], f));
        delete[] keyW4;
        delete[] valW4;
    }

    if (n[11] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[11]];
        double *valW4 = new double[n[11]];
        unsigned long k = 0;
        for (std::map<unsigned long long, double>::iterator i4 = _ber4.begin(); i4 != _ber4.end(); ++ i4) {
            keyW4[k] = i4->first;
            valW4[k] = i4->second;
            ++ k;
        }
        assert(n[11] == fwrite(keyW4, sizeof(keyW4[0]), n[11], f));
        assert(n[11] == fwrite(valW4, sizeof(valW4[0]), n[11], f));
        delete[] keyW4;
        delete[] valW4;
    }

    if (n[12] > 0) {
        unsigned long *keyW2 = new unsigned long[n[12]];
        unsigned long *valW2 = new unsigned long[n[12]];
        unsigned long k = 0;
        for (std::map<unsigned long, unsigned long>::iterator i2 = _i2.begin(); i2 != _i2.end(); ++ i2) {
            keyW2[k] = i2->first;
            valW2[k] = i2->second;
            ++ k;
        }
        assert(n[12] == fwrite(keyW2, sizeof(keyW2[0]), n[12], f));
        assert(n[12] == fwrite(valW2, sizeof(valW2[0]), n[12], f));
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[13] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[13]];
        unsigned long *valW3 = new unsigned long[n[13]];
        unsigned long k = 0;
        for (std::map<unsigned long long, unsigned long>::iterator i3 = _i3.begin(); i3 != _i3.end(); ++ i3) {
            keyW3[k] = i3->first;
            valW3[k] = i3->second;
            ++ k;
        }
        assert(n[13] == fwrite(keyW3, sizeof(keyW3[0]), n[13], f));
        assert(n[13] == fwrite(valW3, sizeof(valW3[0]), n[13], f));
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[14] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[14]];
        unsigned long *valW4 = new unsigned long[n[14]];
        unsigned long k = 0;
        for (std::map<unsigned long long, unsigned long>::iterator i4 = _i4.begin(); i4 != _i4.end(); ++ i4) {
            keyW4[k] = i4->first;
            valW4[k] = i4->second;
            ++ k;
        }
        assert(n[14] == fwrite(keyW4, sizeof(keyW4[0]), n[14], f));
        assert(n[14] == fwrite(valW4, sizeof(valW4[0]), n[14], f));
        delete[] keyW4;
        delete[] valW4;
    }

    fclose(f);
    //std::cout << "saveToFile end" << std::endl;
}

void Apriori::loadFromFile(char *fileName) {
    FILE *f = fopen(fileName, "r");
    assert(1 == fread(&_threshold, sizeof(_threshold), 1, f));

    unsigned long n[15]; 
    assert(15 == fread(n, sizeof(n[0]), 15, f));

    if (n[0] > 0) {
        unsigned short *keyW1 = new unsigned short[n[0]];
        unsigned long *valW1 = new unsigned long[n[0]];
        assert(n[0] == fread(keyW1, sizeof(keyW1[0]), n[0], f));
        assert(n[0] == fread(valW1, sizeof(valW1[0]), n[0], f));
        for (unsigned long k = 0; k < n[0]; ++ k) {
            _w1[keyW1[k]] = valW1[k];
        }
        delete[] keyW1;
        delete[] valW1;
    }

    if (n[1] > 0) {
        unsigned long *keyW2 = new unsigned long[n[1]];
        unsigned long *valW2 = new unsigned long[n[1]];
        assert(n[1] == fread(keyW2, sizeof(keyW2[0]), n[1], f));
        assert(n[1] == fread(valW2, sizeof(valW2[0]), n[1], f));
        for (unsigned long k = 0; k < n[1]; ++ k) {
            _w2[keyW2[k]] = valW2[k];
        }
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[2] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[2]];
        unsigned long *valW3 = new unsigned long[n[2]];
        assert(n[2] == fread(keyW3, sizeof(keyW3[0]), n[2], f));
        assert(n[2] == fread(valW3, sizeof(valW3[0]), n[2], f));
        for (unsigned long k = 0; k < n[2]; ++ k) {
            _w3[keyW3[k]] = valW3[k];
        }
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[3] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[3]];
        unsigned long *valW4 = new unsigned long[n[3]];
        assert(n[3] == fread(keyW4, sizeof(keyW4[0]), n[3], f));
        assert(n[3] == fread(valW4, sizeof(valW4[0]), n[3], f));
        for (unsigned long k = 0; k < n[3]; ++ k) {
            _w4[keyW4[k]] = valW4[k];
        }
        delete[] keyW4;
        delete[] valW4;
    }

    if (n[4] > 0) {
        unsigned short *keyW1 = new unsigned short[n[4]];
        double *valW1 = new double[n[4]];
        assert(n[4] == fread(keyW1, sizeof(keyW1[0]), n[4], f));
        assert(n[4] == fread(valW1, sizeof(valW1[0]), n[4], f));
        for (unsigned long k = 0; k < n[4]; ++ k) {
            _bel1[keyW1[k]] = valW1[k];
        }
        delete[] keyW1;
        delete[] valW1;
    }

    if (n[5] > 0) {
        unsigned short *keyW1 = new unsigned short[n[5]];
        double *valW1 = new double[n[5]];
        assert(n[5] == fread(keyW1, sizeof(keyW1[0]), n[5], f));
        assert(n[5] == fread(valW1, sizeof(valW1[0]), n[5], f));
        for (unsigned long k = 0; k < n[5]; ++ k) {
            _ber1[keyW1[k]] = valW1[k];
        }
        delete[] keyW1;
        delete[] valW1;
    }

    if (n[6] > 0) {
        unsigned long *keyW2 = new unsigned long[n[6]];
        double *valW2 = new double[n[6]];
        assert(n[6] == fread(keyW2, sizeof(keyW2[0]), n[6], f));
        assert(n[6] == fread(valW2, sizeof(valW2[0]), n[6], f));
        for (unsigned long k = 0; k < n[6]; ++ k) {
            _bel2[keyW2[k]] = valW2[k];
        }
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[7] > 0) {
        unsigned long *keyW2 = new unsigned long[n[7]];
        double *valW2 = new double[n[7]];
        assert(n[7] == fread(keyW2, sizeof(keyW2[0]), n[7], f));
        assert(n[7] == fread(valW2, sizeof(valW2[0]), n[7], f));
        for (unsigned long k = 0; k < n[7]; ++ k) {
            _ber2[keyW2[k]] = valW2[k];
        }
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[8] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[8]];
        double *valW3 = new double[n[8]];
        assert(n[8] == fread(keyW3, sizeof(keyW3[0]), n[8], f));
        assert(n[8] == fread(valW3, sizeof(valW3[0]), n[8], f));
        for (unsigned long k = 0; k < n[8]; ++ k) {
            _bel3[keyW3[k]] = valW3[k];
        }
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[9] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[9]];
        double *valW3 = new double[n[9]];
        assert(n[9] == fread(keyW3, sizeof(keyW3[0]), n[9], f));
        assert(n[9] == fread(valW3, sizeof(valW3[0]), n[9], f));
        for (unsigned long k = 0; k < n[9]; ++ k) {
            _ber3[keyW3[k]] = valW3[k];
        }
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[10] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[10]];
        double *valW4 = new double[n[10]];
        assert(n[10] == fread(keyW4, sizeof(keyW4[0]), n[10], f));
        assert(n[10] == fread(valW4, sizeof(valW4[0]), n[10], f));
        for (unsigned long k = 0; k < n[10]; ++ k) {
            _bel4[keyW4[k]] = valW4[k];
        }
        delete[] keyW4;
        delete[] valW4;
    }

    if (n[11] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[11]];
        double *valW4 = new double[n[11]];
        assert(n[11] == fread(keyW4, sizeof(keyW4[0]), n[11], f));
        assert(n[11] == fread(valW4, sizeof(valW4[0]), n[11], f));
        for (unsigned long k = 0; k < n[11]; ++ k) {
            _ber4[keyW4[k]] = valW4[k];
        }
        delete[] keyW4;
        delete[] valW4;
    }

    if (n[12] > 0) {
        unsigned long *keyW2 = new unsigned long[n[12]];
        unsigned long *valW2 = new unsigned long[n[12]];
        assert(n[12] == fread(keyW2, sizeof(keyW2[0]), n[12], f));
        assert(n[12] == fread(valW2, sizeof(valW2[0]), n[12], f));
        for (unsigned long k = 0; k < n[12]; ++ k) {
            _i2[keyW2[k]] = valW2[k];
        }
        delete[] keyW2;
        delete[] valW2;
    }

    if (n[13] > 0) {
        unsigned long long *keyW3 = new unsigned long long[n[13]];
        unsigned long *valW3 = new unsigned long[n[13]];
        assert(n[13] == fread(keyW3, sizeof(keyW3[0]), n[13], f));
        assert(n[13] == fread(valW3, sizeof(valW3[0]), n[13], f));
        for (unsigned long k = 0; k < n[13]; ++ k) {
            _i3[keyW3[k]] = valW3[k];
        }
        delete[] keyW3;
        delete[] valW3;
    }

    if (n[14] > 0) {
        unsigned long long *keyW4 = new unsigned long long[n[14]];
        unsigned long *valW4 = new unsigned long[n[14]];
        assert(n[14] == fread(keyW4, sizeof(keyW4[0]), n[14], f));
        assert(n[14] == fread(valW4, sizeof(valW4[0]), n[14], f));
        for (unsigned long k = 0; k < n[14]; ++ k) {
            _i4[keyW4[k]] = valW4[k];
        }
        delete[] keyW4;
        delete[] valW4;
    }

    fclose(f);
}

