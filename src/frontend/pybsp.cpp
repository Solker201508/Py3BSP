#include "module.hpp"
#include <iostream>
#include <cstdio>

void bsp_runtimeError(std::string strErr);
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s PYTHON_SCRIPT\n", argv[0]);
        return -1;
    }
    FILE *pyScript = fopen(argv[1],"r");
    if (pyScript == NULL) {
        fprintf(stderr, "ERROR: unable to open file '%s'\n", argv[1]);
        return -2;
    }

    initBSP(&argc, &argv);
    int err = PyRun_SimpleFile(pyScript,argv[1]);
    if (err != 0) {
        return -3;
    }
    fclose(pyScript);
    finiBSP();
    return 0;
}
