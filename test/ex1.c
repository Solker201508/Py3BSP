#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double center[106] = {1,0.6,4.6,0.5,0.87,0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};

double myFunValue1(unsigned long nParams, double *params) {
    double result = 0.0;
    unsigned long i;
    for (i = 0; i < nParams; ++i) {
        double d = fabs(params[i] - center[i]);
        result += d * sqrt(d);
    }
    return result;
}

void myGradient1(unsigned long nParams, double *params, double *result) {
    unsigned long k;
    for (k = 0; k < nParams; ++k) {
        int neg = params[k] < center[k];
        double d = fabs(params[k] - center[k]);
        if (neg)
            result[k] = -1.5 * sqrt(d);
        else
            result[k] = 1.5 * sqrt(d);
    }
}

double myFunValue2(unsigned long nParams, double *params) {
    double result = 0.0;
    unsigned long i;
    for (i = 0; i < nParams; ++i) {
        double d = fabs(params[i] - center[i]);
        result += -d * sqrt(d);
    }
    return result;
}

void myGradient2(unsigned long nParams, double *params, double *result) {
    unsigned long k;
    for (k = 0; k < nParams; ++k) {
        int neg = params[k] < center[k];
        double d = fabs(params[k] - center[k]);
        if (neg)
            result[k] = 1.5 * sqrt(d);
        else
            result[k] = -1.5 * sqrt(d);
    }
}

double myFunValue3(unsigned long nParams, double *params) {
    double result = 0.0;
    unsigned long i;
    for (i = 0; i < nParams; ++i) {
        double d = params[i] - center[i];
        result += d * d * d * d;
    }
    return result;
}

void myGradient3(unsigned long nParams, double *params, double *result) {
    unsigned long k;
    for (k = 0; k < nParams; ++k) {
        double d = params[k] - center[k];
        result[k] = 4 * d * d * d;
    }
}

double myFunValue4(unsigned long nParams, double *params) {
    double result = 0.0;
    unsigned long i;
    for (i = 0; i < nParams; ++i) {
        double d = params[i] - center[i];
        result += -d * d * d * d;
    }
    return result;
}

void myGradient4(unsigned long nParams, double *params, double *result) {
    unsigned long k;
    for (k = 0; k < nParams; ++k) {
        double d = params[k] - center[k];
        result[k] = -4 * d * d * d;
    }
}

void strReverse(char *str) {
    int n = strlen(str);
    char *rev = malloc(n + 1);
    rev[n] = 0;
    int i;
    for (i = n - 1; i >=0; -- i) {
        rev[i] = str[n - 1 - i];
    }
    printf("%s\n",rev);
}

void repeat(void fun(char *str), int t, char *str) {
    printf("%lx\n", (size_t) fun);
    int i;
    for (i = 0; i < t; ++ i) {
        fun(str);
    }
}
