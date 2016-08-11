#ifndef __BSP_ALG_APRIORI_HPP__
#define __BSP_ALG_APRIORI_HPP__
#include <cstdio>

namespace BSP {
    namespace Algorithm {
        class Apriori {
            class Node {
                private:
                    Node *_children[256];
                    unsigned long _counter[256];
                public:
                    Node();
                    ~Node();
                    void reset();
                    unsigned long countOf(unsigned int k, char *x);
                    void add(unsigned int k, char *x);
                    void spawn(unsigned int k, unsigned int threshold);
                    void saveToFile(FILE *file);
                    void loadFromFile(FILE *file);

            };
            private:
                unsigned int _unit;
                unsigned int _depth;
                unsigned int _threshold;
                Node _root;
            public:
                Apriori(unsigned int unit);
                ~Apriori();
                void setThreshold(unsigned int threshold);
                void scan(unsigned long nUnits, unsigned long unitDepth, char *x);
                void scan(unsigned long nUnits, char *x, int tmplPos1);
                void scan(unsigned long nUnits, char *x, int tmplPos1, int tmplPos2);
                unsigned long scan(unsigned long nUnits, unsigned long *posUnit, char *x);
                void getFreq(unsigned long nUnits, unsigned long unitDepth, char *x, int *Freq);
                void getFreq(unsigned long nUnits, char *x, int tmplPos1, int *Freq);
                void getFreq(unsigned long nUnits, char *x, int tmplPos1, int tmplPos2, int *Freq);
                void getFreq(unsigned long nUnits, unsigned int depth, unsigned long *posUnit, char *x, int *Freq);
                void saveToFile(char *fileName);
                void loadFromFile(char *fileName);
        };
    }
}

#endif
