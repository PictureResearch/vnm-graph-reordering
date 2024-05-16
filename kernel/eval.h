#ifndef EVAL_H
#define EVAL_H
#include "spmm.cu"
#include "../mtx/Mtx.h"

template<typename T>
std::vector<double> eval(Mtx<T> *spmat, int n, bool verbose=false)
{
    // convert A to dense for dense MM evaluation
    int m = spmat->nrows, k = spmat->ncols;

    __half *hA = (__half*)malloc(m * k * sizeof(__half));
    __half *hB = (__half*)malloc(k * n * sizeof(__half));
    __half *hC = (__half*)malloc(m * n * sizeof(__half));

    memset(hA, 0.0f, m * k * sizeof(__half));
    for (int i = 0; i < spmat->nnz; i++)
        hA[spmat->coo_rowind_h[i]*k+spmat->coo_colind_h[i]] = __float2half(spmat->coo_values_h[i]);

    // randomized B matrix
    srand(time(0));
    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<__half>(static_cast<float>((float)rand() / RAND_MAX));

    // init result C matrix
    memset(hC, 0.0f, m * n * sizeof(__half));

    /* run SpMM as dense MM (A Mtx as dense) with cuBLAS */
    // cuBLAS HGemm
    // double cublas_hgemm_time = evalCuBLASHGemm(hA, hB, hC, M, N, K);
    // printf("cublas_hgemm_time: %.3f\n", cublas_hgemm_time);
    // __half *result_cublas_hgemm = (__half*)malloc(M * N * sizeof(__half));
    // cudaMemcpy(result_cublas_hgemm, hC, M * N * sizeof(__half), cudaMemcpyHostToHost);
    // memset(hC, 0.0f, M * N * sizeof(__half));

    // cuBLAS GemmEX
    double cublas_gemmex_time = evaluCuBLASGemmex(hA, hB, hC, m, n, k);
    // printf("cublas_gemmex_time: %.3f\n", cublas_gemmex_time);
    __half *result_cublas_gemmex= (__half*)malloc(m * n * sizeof(__half));
    cudaMemcpy(result_cublas_gemmex, hC, m * n * sizeof(__half), cudaMemcpyHostToHost);
    memset(hC, 0.0f, m * n * sizeof(__half));

    /* run SpMM as N:M-sparse (A Mtx as sparse) with cuSPARSELt */
    /* !! row should be 16-aligned !!*/
    double cusparselt_matmul_time = evalCuSPARSELtMatmul(hA, hB, hC, m, n, k);
    // printf("cusparselt_matmul_time: %.3f\n", cusparselt_matmul_time);
    __half *result_cusparselt_matmul = (__half*)malloc(m * n * sizeof(__half));
    cudaMemcpy(result_cusparselt_matmul, hC, m * n * sizeof(__half), cudaMemcpyHostToHost);
    memset(hC, 0.0f, m * n * sizeof(__half));

    // /* run SpMM as block-sparse-16x16 (A Mtx as sparse) with cuSPARSE */
    // /* !! row should be 16-aligned !!*/
    // // convert bsr to blockedell
    // int *ell_columns;
    // __half *ell_values;
    // int *h_bsrRowPtr = (int *) malloc(sizeof(int) * (spmat->nblockrows + 1));
    // cudaMemcpy(h_bsrRowPtr, spmat->device_ref.bsr_indptr, sizeof(int) * (spmat->nblockrows + 1), cudaMemcpyDeviceToHost);
    // int ell_width = get_ell_width(h_bsrRowPtr, spmat->nblockrows) * spmat->blockdim;
    // int ell_nnz = ell_width * spmat->nrows;
    // SAFE_ALOC_GPU(ell_columns, ell_nnz / (spmat->blockdim * spmat->blockdim) * sizeof(int));
    // SAFE_ALOC_GPU(ell_values, ell_nnz * sizeof(__half));
    // BSR2BlockedELLhalf(ell_width, ell_columns, ell_values,
    //                    hA, m, k, 
    //                    spmat->nblockrows, spmat->nblocks, spmat->blockdim,
    //                    spmat->device_ref.bsr_indptr, spmat->device_ref.bsr_indices);
    
    // // kernel
    // double cusparse_blockedell_time = evalCuSPARSESpMMBlockedell(ell_columns, ell_values, ell_width,
    //                                                              hB, hC,
    //                                                              m, n, k, spmat->blockdim);
    // // printf("cusparse_blockedell_time: %.3f\n", cusparse_blockedell_time);
    // __half *result_cusparse_blockedell = (__half*)malloc(m * n * sizeof(__half));
    // cudaMemcpy(result_cusparse_blockedell, hC, m * n * sizeof(__half), cudaMemcpyHostToHost);
    // memset(hC, 0.0f, m * n * sizeof(__half));

    // /* run SpMM as block-sparse (A Mtx as sparse) with custom bsrwmma kernel */
    // // convert bsr values to half
    // __half *bsr_values_half;
    // int bsr_val_size = spmat->nblocks * spmat->blockdim * spmat->blockdim;
    // SAFE_ALOC_GPU(bsr_values_half, bsr_val_size * sizeof(__half));
    // convertFp32ToFp16<<< (bsr_val_size + 1023) / 1024, 1024 >>>
    //                 (bsr_values_half, spmat->device_ref.bsr_values, bsr_val_size);
    // // kernel
    // double custom_bsrwmma_time = evalCustomBsrwmma(spmat->device_ref.bsr_indptr, 
    //                                                spmat->device_ref.bsr_indices, 
    //                                                bsr_values_half,
    //                                                hB, hC,
    //                                                m, n, k, spmat->blockdim);
    // // printf("custom_bsrwmma_time: %.3f\n", custom_bsrwmma_time);
    // __half *result_custom_bsrwmma = (__half*)malloc(m * n * sizeof(__half));
    // cudaMemcpy(result_custom_bsrwmma, hC, m * n * sizeof(__half), cudaMemcpyHostToHost);
    // memset(hC, 0.0f, m * n * sizeof(__half));

    // // free bsr half
    // SAFE_FREE_GPU(bsr_values_half);

    // verify result
    // verifyResult(result_cublas_hgemm, result_custom_bsrwmma, true); // <-- hgemm fail when bcols !=32
    // bool pass1 = verifyResult(result_cublas_gemmex, result_cusparselt_matmul, m * n); // <-- need to check 
    // bool pass2 = verifyResult(result_cublas_gemmex, result_cusparse_blockedell, m * n); // pass
    // bool pass3 = verifyResult(result_cublas_gemmex, result_custom_bsrwmma, m * n); // pass

    if (verbose) {
        std::cout   << "\n==== kernel eval: ====\n"
                    << "\ncublas_gemmex_time = " << cublas_gemmex_time
                    // << "\ncusparselt_matmul_time = " << cusparselt_matmul_time
                    << "\ncusparse_blockedell_time = " << cusparse_blockedell_time
                    << "\ncustom_bsrwmma_time = " << custom_bsrwmma_time
                    // << "\npass1 = " << (pass1=0?0:1)
                    << "\npass2 = " << (pass2=0?0:1)
                    << "\npass3 = " << (pass3=0?0:1);
        std::cout    <<   "\n" ;
    }

    // SAFE_FREE_HOST(hA);
    // SAFE_FREE_HOST(hB);
    // SAFE_FREE_HOST(hC);
    // SAFE_FREE_HOST(ell_columns);
    // SAFE_FREE_HOST(ell_values);
    // SAFE_FREE_HOST(result_cublas_gemmex);
    // SAFE_FREE_HOST(result_cusparselt_matmul);
    // SAFE_FREE_HOST(result_cusparse_blockedell);
    // SAFE_FREE_HOST(result_custom_bsrwmma);


    return {cublas_gemmex_time, cusparselt_matmul_time};

    // return {cublas_gemmex_time, cusparselt_matmul_time, 
    //         cusparse_blockedell_time, custom_bsrwmma_time,
    //         static_cast<double>(pass1), static_cast<double>(pass2), static_cast<double>(pass3)};

    // return {cublas_gemmex_time, cusparse_blockedell_time, custom_bsrwmma_time,
    //         static_cast<double>(pass2=0?0:1), static_cast<double>(pass3=0?0:1)};

    // return {cublas_gemmex_time, custom_bsrwmma_time, static_cast<double>(pass3)};
}


#endif