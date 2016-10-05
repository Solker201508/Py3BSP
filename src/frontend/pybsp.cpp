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
    char errBuf[] = "IF NOT KNOWING THE LOCATION OF THE ERROR, PLEASE PUT YOUR CODE IN THIS TRY EXCEPT BLOCK AND RUN AGAIN TO FIND OUT WHERE THE ERROR OCCURS:\n--------------------\nimport sys\nimport traceback\ntry:\n    #PUT YOURCODE HERE\n    ...\nexcept:\n    info = sys.exc_info()\n    print('Error: ', info[1])\n    traceback.print_tb(info[2])\n--------------------\nCHECK THE .ERR FILES IF THIS DOES NOT WORK\n";
    if (err != 0) {
        printf("%s",errBuf);
        return -3;
    }
    fclose(pyScript);
    finiBSP();
    return 0;
}
