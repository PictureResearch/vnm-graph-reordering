#include <string>
#include <iostream>

using namespace std;

void parseArgs(const int argc, const char** argv, 
               std::string &mtxfile, std::string &outmtxfile, int &maxiter, 
               int &n, int &sched, int &v, bool verbose=false)
{
    std::string Usage =
        "\tRequired cmdline args:\n\
        --mtxfile: input .mtx file path in Matrix Market format. \n\
        --outmtxfile: output .mtx file path in Matrix Market format. \n\
        --maxiter: max number of iteration for reordering algorithm.\n\
        --n: B matrix col.\n\
        --sched: the schedule for nm reordering.\n\
        --v: the V for venom reordering.\n\
        \n";

    mtxfile = "";
    outmtxfile = "";
    maxiter = 0;
    n = 0;
    sched = 0;
    v = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mtxfile") && i!=argc-1) {
            mtxfile = argv[i+1];
        }
        else if (!strcmp(argv[i], "--outmtxfile") && i!=argc-1) {
            outmtxfile = argv[i+1];
        }
        else if (!strcmp(argv[i], "--maxiter") && i!=argc-1) {
            maxiter = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--n") && i!=argc-1) {
            n = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--sched") && i!=argc-1) {
            sched = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--v") && i!=argc-1) {
            v = atoi(argv[i+1]);
        }
    }



    if (verbose) {
        std::cout   << "\n==== arguments: ====\n"
                    << "\nmtxfile = " << mtxfile
                    << "\noutmtxfile = " << outmtxfile
                    << "\nmaxiter = " << maxiter
                    << "\nn = " << n
                    << "\nsched = " << sched
                    << "\nv = " << v;

        std::cout    <<   "\n" ;
    }
}