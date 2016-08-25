#ifndef __BSP_ALG_APRIORI_HPP__
#define __BSP_ALG_APRIORI_HPP__
#include <map>
namespace BSP {
    namespace Algorithm {
        class Apriori {
            private:
                std::map<unsigned short, unsigned long> _w1;
                std::map<unsigned long, unsigned long> _w2;
                std::map<unsigned long long, unsigned long> _w3, _w4;
                std::map<unsigned short, double> _bel1, _ber1;
                std::map<unsigned long, double> _bel2, _ber2;
                std::map<unsigned long long, double> _bel3, _ber3, _bel4, _ber4;
                std::map<unsigned long, unsigned long> _i2;
                std::map<unsigned long long, unsigned long> _i3;
                std::map<unsigned long long, unsigned long> _i4;
                unsigned int _threshold;
            protected:
                void scan1(unsigned long n, unsigned short *x, bool multiThread); 
                void scan2(unsigned long n, unsigned short *x, bool multiThread);
                void scan2(unsigned long n, int pos1, int pos2, unsigned short *x);
                void scan3(unsigned long n, unsigned short *x, bool multiThread);
                void scan3(unsigned long n, int pos1, int pos2, int pos3, unsigned short *x);
                void scan4(unsigned long n, unsigned short *x, bool multiThread);
                void scan4(unsigned long n, int pos1, int pos2, int pos3, int pos4, unsigned short *x);
            public:
                Apriori(unsigned int threshold);
                ~Apriori();
                void scan(unsigned long n, unsigned short *x, bool multiThread = true);
                void scan(unsigned long n, int pos1, int pos2, unsigned short *x);
                void scan(unsigned long n, int pos1, int pos2, int pos3, unsigned short *x);
                void scan(unsigned long n, int pos1, int pos2, int pos3, int pos4, unsigned short *x);
                void getFreq(unsigned long n, unsigned short *x, 
                        int *freq1, int *freq2, int *freq3, int *freq4,
                        double *bel1, double *ber1, double *bel2, double *ber2,
                        double *bel3, double *ber3, double *bel4, double *ber4);
                int getIndex2(unsigned long n, int pos1, int pos2, unsigned short *x, int start, int *index);
                int getIndex3(unsigned long n, int pos1, int pos2, int pos3, unsigned short *x, int start, int *index);
                int getIndex4(unsigned long n, int pos1, int pos2, int pos3, int pos4, unsigned short *x, int start, int *index);
                void saveToFile(char *fileName);
                void loadFromFile(char *fileName);
                unsigned long mostFrequent(unsigned short *word);
                double largestBE();
                unsigned long freq(unsigned short word);
        };
    }
}

#endif
