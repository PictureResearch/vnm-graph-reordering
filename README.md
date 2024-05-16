# vnm-graph-reordering

## General Instruction
To reorder a matrix:

1. Set V, N, M of interest
    * in `/mtx/Mtx.h`, modify line 15 and 16 for N and M
    * specify V in the flag `--v {V_val}`

2. `make clean && make spmm`

3. Specify param and I/O matrix path. Execute the following: `./spmm --mtxfile {path_to_input_matrix_file}.mtx --outmtxfile {path_to_output_matrix_file}.mtx --maxiter 10 --n 64 --sched 0 --v {V_val}`

4. The reordered matrix will be stored in `{path_to_output_matrix_file}.mtx`

## Artifect Evaluation
* A1. Reordering result quality

* A2. GNN performance improvement via the reordering
    * visit the `/gnn` folder

* A3. SpMM kernel speedups via the reordering
