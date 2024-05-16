#include <stdio.h>
#include <stdlib.h>

#include "util/argparse.h" 
#include "mtx/Mtx.h"
#include "reorder/reorder.h"
#include "kernel/eval.h"

int main(int argc, const char **argv)
{
    // parse arg
    std::string mtxfile;
    int maxiter, n;
    parseArgs(argc, argv, mtxfile, maxiter, n, false);

    // read original mtx
    Mtx<float> *spmat = new Mtx<float>(mtxfile, /*aligned=*/ 16); 
    // <-- set 16-aligned if run cuda lib (blockedell, cusparseLt)
    // spmat->print();

    // run kernel before reoder
    std::vector<double> res_before = eval(spmat, n, false);
    std::cout << res_before[0] << ", " << res_before[1] << ", ";

    delete spmat;
}