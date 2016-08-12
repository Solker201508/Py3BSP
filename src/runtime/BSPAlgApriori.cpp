#include "BSPAlgApriori.hpp"
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdio>

using namespace BSP::Algorithm;

Apriori::Apriori(unsigned int threshold) {
    _threshold = threshold;
}

Apriori::~Apriori() {
}

void Apriori::scan1(unsigned long n, unsigned short *x) {
    std::cout << "scan pass 1 begin" << std::endl;
    for (unsigned long i = 0; i < n; ++ i) {
        std::map<unsigned short, unsigned long>::iterator loc = _w1.find(x[i]);
        if (loc != _w1.end()) {
            loc->second += 1;
        } else {
            _w1.insert(std::pair<unsigned short, unsigned long>(x[i], 1));
        }
        if ((i + 1) % 1000000 == 0) {
            std::cout << "scan pass 1: " << i + 1 << " completed" << std::endl;
        }
    }
    std::cout << "scan pass 1 done" << std::endl;
}

unsigned long key2(unsigned short k0, unsigned short k1) {
    return (((unsigned long)k1) << 16) | k0; 
}

unsigned long long key3(unsigned short k0, unsigned short k1, unsigned short k2) {
    return (((unsigned long)k2) << 32) | (((unsigned long)k1) << 16) | k0; 
}

unsigned long long key3(unsigned long k0, unsigned long k1) {
    if ((k0 >> 16) == (k1 & ((1 << 16) - 1)))
        return (((unsigned long long)k1) << 16) | k0;
    else
        return (unsigned long long)-1;
}

unsigned long long key4(unsigned short k0, unsigned short k1, unsigned short k2, unsigned short k3) {
    return (((unsigned long)k3) << 48) | (((unsigned long)k2) << 32) | (((unsigned long)k1) << 16) | k0; 
}

unsigned long long key4(unsigned long long k0, unsigned long long k1) {
    if ((k0 >> 16) == (k1 & ((1ULL << 32) - 1)))
        return (k1 << 16) | k0;
    else
        return (unsigned long long)-1;
}

void Apriori::scan2(unsigned long n, unsigned short *x) {
    std::cout << "scan pass 2 begin" << std::endl;
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
        if ((i + 1) % 1000000 == 0) {
            std::cout << "scan pass 2: " << i + 1 << " completed" << std::endl;
        }
    }
    std::cout << "scan pass 2 calculating boundary entropies" << std::endl;
    unsigned long kBE = 0;
    for (std::map<unsigned short, unsigned long>::iterator loc0 = _w1.begin(); loc0 != _w1.end(); ++ loc0, ++ kBE) {
        std::map<unsigned short, unsigned long> left, right;
        std::map<unsigned long, unsigned long>::iterator loc;
        unsigned long sumLeft = 0, sumRight = 0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = _w1.begin(); loc1 != _w1.end(); ++ loc1) {
            if (loc0->first == loc1->first)
                continue;
            if (loc0->second > _threshold && loc1->second > _threshold) {
                loc = _w2.find(key2(loc0->first, loc1->first));
                if (loc != _w2.end()) {
                    right[loc1->first] = loc->second;
                    sumRight += loc->second;
                }

                loc = _w2.find(key2(loc1->first, loc0->first));
                if (loc != _w2.end()) {
                    left[loc1->first] = loc->second;
                    sumLeft += loc->second;
                }
            }
        }
        double leftEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = left.begin(); loc1 != left.end(); ++ loc1) {
            double p = loc1->second / (double) sumLeft;
            leftEntropy -= p * log(p);
        }
        double rightEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = right.begin(); loc1 != right.end(); ++ loc1) {
            double p = loc1->second / (double) sumRight;
            rightEntropy -= p * log(p);
        }
        _bel1[loc0->first] = leftEntropy;
        _ber1[loc0->first] = rightEntropy;
        if ((kBE + 1) % 100 == 0) {
            std::cout << "scan pass 2 BE: " << kBE + 1 << " of " << _w1.size() << " completed" << std::endl;
        }
    }
    std::cout << "scan pass 2 done" << std::endl;
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

void Apriori::scan3(unsigned long n, unsigned short *x) {
    std::cout << "scan pass 3 begin" << std::endl;
    for (unsigned long i = 0; i + 2 < n; ++ i) {
        std::map<unsigned long, unsigned long>::iterator loc0 = _w2.find(key2(x[i],x[i + 1])), 
            loc1 = _w2.find(key2(x[i + 1], x[i + 2]));
        if (loc0->second > _threshold && loc1->second > _threshold) {
            unsigned long long key = key3(x[i], x[i + 1], x[i + 2]);
            std::map<unsigned long long, unsigned long>::iterator loc = _w3.find(key);
            if (loc == _w3.end()) {
                _w3.insert(std::pair<unsigned long long, unsigned long>(key, 1));
            } else {
                loc->second += 1;
            }
        }
        if ((i + 1) % 1000000 == 0) {
            std::cout << "scan pass 3: " << i + 1 << " completed" << std::endl;
        }
    }
    std::cout << "scan pass 3 calculating boundary entropies" << std::endl;
    unsigned long kBE = 0;
    for (std::map<unsigned long, unsigned long>::iterator loc0 = _w2.begin(); loc0 != _w2.end(); ++ loc0, ++ kBE) {
        std::map<unsigned short, unsigned long> left, right;
        std::map<unsigned long long, unsigned long>::iterator loc;
        unsigned long sumLeft = 0, sumRight = 0;
        for (std::map<unsigned long, unsigned long>::iterator loc1 = _w2.begin(); loc1 != _w2.end(); ++ loc1) {
            if (loc0->first == loc1->first)
                continue;
            if (loc0->second > _threshold && loc1->second > _threshold) {
                loc = _w3.find(key3(loc0->first, loc1->first));
                if (loc != _w3.end()) {
                    right[loc1->first >> 16] = loc->second;
                    sumRight += loc->second;
                }

                loc = _w3.find(key3(loc1->first, loc0->first));
                if (loc != _w3.end()) {
                    left[loc1->first & ((1 << 16) - 1)] = loc->second;
                    sumLeft += loc->second;
                }
            }
        }
        double leftEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = left.begin(); loc1 != left.end(); ++ loc1) {
            double p = loc1->second / (double) sumLeft;
            leftEntropy -= p * log(p);
        }
        double rightEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = right.begin(); loc1 != right.end(); ++ loc1) {
            double p = loc1->second / (double) sumRight;
            rightEntropy -= p * log(p);
        }
        _bel2[loc0->first] = leftEntropy;
        _ber2[loc0->first] = rightEntropy;
        if ((kBE + 1) % 100 == 0) {
            std::cout << "scan pass 3 BE: " << kBE + 1 << " of " << _w2.size() << " completed" << std::endl;
        }
    }
    std::cout << "scan pass 3 done" << std::endl;
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

void Apriori::scan4(unsigned long n, unsigned short *x) {
    std::cout << "scan pass 4 begin" << std::endl;
    for (unsigned long i = 0; i + 3 < n; ++ i) {
        std::map<unsigned long long, unsigned long>::iterator loc0 = _w3.find(key3(x[i], x[i + 1], x[i + 2])), 
            loc1 = _w3.find(key3(x[i + 1], x[i + 2], x[i + 3]));
        if (loc0->second > _threshold && loc1->second > _threshold) {
            unsigned long long key = key4(x[i], x[i + 1], x[i + 2], x[i + 3]);
            std::map<unsigned long long, unsigned long>::iterator loc = _w4.find(key);
            if (loc == _w4.end()) {
                _w4.insert(std::pair<unsigned long long, unsigned long>(key, 1));
            } else {
                loc->second += 1;
            }
        }
        if ((i + 1) % 1000000 == 0) {
            std::cout << "scan pass 4: " << i + 1 << " completed" << std::endl;
        }
    }
    std::cout << "scan pass 4 calculating boundary entropies" << std::endl;
    unsigned long kBE = 0;
    for (std::map<unsigned long long, unsigned long>::iterator loc0 = _w3.begin(); loc0 != _w3.end(); ++ loc0, ++ kBE) {
        std::map<unsigned short, unsigned long> left, right;
        std::map<unsigned long long, unsigned long>::iterator loc;
        unsigned long sumLeft = 0, sumRight = 0;
        for (std::map<unsigned long long, unsigned long>::iterator loc1 = _w3.begin(); loc1 != _w3.end(); ++ loc1) {
            if (loc0->first == loc1->first)
                continue;
            if (loc0->second > _threshold && loc1->second > _threshold) {
                loc = _w4.find(key4(loc0->first, loc1->first));
                if (loc != _w4.end()) {
                    right[loc1->first >> 32] = loc->second;
                    sumRight += loc->second;
                }

                loc = _w4.find(key4(loc1->first, loc0->first));
                if (loc != _w4.end()) {
                    left[loc1->first & ((1 << 16) - 1)] = loc->second;
                    sumLeft += loc->second;
                }
            }
        }
        double leftEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = left.begin(); loc1 != left.end(); ++ loc1) {
            double p = loc1->second / (double) sumLeft;
            leftEntropy -= p * log(p);
        }
        double rightEntropy = 0.0;
        for (std::map<unsigned short, unsigned long>::iterator loc1 = right.begin(); loc1 != right.end(); ++ loc1) {
            double p = loc1->second / (double) sumRight;
            rightEntropy -= p * log(p);
        }
        _bel3[loc0->first] = leftEntropy;
        _ber3[loc0->first] = rightEntropy;
        if ((kBE + 1) % 100 == 0) {
            std::cout << "scan pass 4 BE: " << kBE + 1 << " of " << _w3.size() << " completed" << std::endl;
        }
    }

    // bel4, ber4
    std::cout << "scan pass 4 finalizing ... " << std::endl;
    typedef std::pair< std::map<unsigned short, unsigned long>, std::map<unsigned short, unsigned long> > LR;
    std::map< unsigned long long, LR > be;
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
    std::cout << "scan pass 4 done" << std::endl;
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

void Apriori::scan(unsigned long n, unsigned short *x) {
    scan1(n, x);
    scan2(n, x);
    scan3(n, x);
    scan4(n, x);
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
        assert(n[0] == fwrite(valW3, sizeof(valW3[0]), n[9], f));
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
}

void Apriori::loadFromFile(char *fileName) {
    FILE *f = fopen(fileName, "r");
    fread(&_threshold, sizeof(_threshold), 1, f);

    unsigned long n[15]; 
    fread(n, sizeof(n[0]), 15, f);

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

