#ifndef MTX_H
#define MTX_H

#include <vector>
#include "../util/utility.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <stdio.h>
#include "readMtx.hpp"

#define N 2
#define M 8

template<typename T>
class Mtx {
public:
    Mtx(std::string &mtxfile, int aligned) { this->init_from_file(mtxfile, aligned); }
    Mtx(Mtx<T> &cpymat) { this->init_from_copy(cpymat); }

    void init_from_file(std::string &mtxfile, int aligned);
    void init_from_copy(Mtx<T> &cpymat);
    void print();
    void convert_to_bsr_and_sync_device();
    void get_bsr_host();
    void get_coo_host();
    void write_to_file(std::string &destfile); // based on latest coo
    void reset(); // use coo format to update
    void __clear_device_ref() {
        SAFE_FREE_GPU(device_ref.coo_rowind);
        SAFE_FREE_GPU(device_ref.coo_colind);
        SAFE_FREE_GPU(device_ref.coo_values);

        SAFE_FREE_GPU(device_ref.csr_indptr);
        SAFE_FREE_GPU(device_ref.csr_indices);
        SAFE_FREE_GPU(device_ref.csr_values);

        SAFE_FREE_GPU(device_ref.bsr_indptr);
        SAFE_FREE_GPU(device_ref.bsr_indices);
        SAFE_FREE_GPU(device_ref.bsr_values);
    }

    ~Mtx() {
        if (device_synced) {
            __clear_device_ref();
        }
    }

public:
    // general info
    int nrows;
    int ncols;
    int nnz;

    // bsr info
    int blockdim = M;
    int nblockrows;
    int nblocks;

    // coo format on host (exist from initialize)
    std::vector<int> coo_rowind_h, coo_colind_h;
    std::vector<T> coo_values_h;

    // csr format on host (exist from initialize)
    std::vector<int> csr_indptr_h, csr_indices_h;
    std::vector<T> csr_values_h;

    // bsr format on host (for print() verify only)
    std::vector<int> bsr_indptr_h, bsr_indices_h;
    std::vector<T> bsr_values_h;

    // pointers to transformed format on device
    struct DeviceRef {
        // coo format -> to keep track of id permute
        int  *coo_rowind;
        int  *coo_colind;
        T    *coo_values;

        // csr format
        int  *csr_indptr;
        int  *csr_indices;
        T    *csr_values;

        // bsr format
        int  *bsr_indptr;
        int  *bsr_indices;
        T    *bsr_values;
    } device_ref;

    bool initialized = false;
    bool device_synced = false;
    bool bsr_converted = false;
};

template<typename T>
void Mtx<T>::init_from_file(std::string &mtxfile, int aligned)
{
    // load coo format host, general info
    readMtx(mtxfile.c_str(), &this->coo_rowind_h, &this->coo_colind_h, &this->coo_values_h,
    &this->nrows, &this->ncols, &this->nnz, 0, false);

    // alignment processing (for reorder algorithm)
    if (this->nrows % aligned != 0) {
        this->nrows = ((this->nrows + aligned - 1) / aligned) * aligned;
        this->ncols = ((this->ncols + aligned - 1) / aligned) * aligned;
    }

    // ensure nnz is correct
    if (this->nnz != coo_values_h.size()) this->nnz = coo_values_h.size();

    // load csr format host
    // int *csrindptr, *csrcolind;
    // T *csrval;
    // SAFE_ALOC_HOST(csrindptr, (this->nrows+1)*sizeof(int));
    // SAFE_ALOC_HOST(csrcolind, this->nnz*sizeof(int));
    // SAFE_ALOC_HOST(csrval, this->nnz*sizeof(T));
    int *csrindptr = (int*)malloc((this->nrows+1) * sizeof(int));
    int *csrcolind = (int*)malloc(this->nnz * sizeof(int));
    T *csrval = (T*)malloc(this->nnz * sizeof(T));

    coo2csr(csrindptr, csrcolind, csrval,
            this->coo_rowind_h, this->coo_colind_h, this->coo_values_h, 
            this->nrows, this->ncols);
    
    this->csr_indptr_h.resize(this->nrows+1);
    this->csr_indices_h.resize(this->nnz);
    this->csr_values_h.resize(this->nnz);
    memcpy(&this->csr_indptr_h[0], csrindptr, (this->nrows+1)*sizeof(int));
    memcpy(&this->csr_indices_h[0], csrcolind, this->nnz*sizeof(int));
    memcpy(&this->csr_values_h[0], csrval, this->nnz*sizeof(T));
    // SAFE_FREE_HOST(csrindptr);
    // SAFE_FREE_HOST(csrcolind);
    // SAFE_FREE_HOST(csrval);
    free(csrindptr);
    free(csrcolind);
    free(csrval);

    // set flag
    initialized = true;
    if (device_synced) {
        // clear any old version
        this->__clear_device_ref();
        device_synced = false;
        bsr_converted = false;
    }

    // convert to bsr and sync storage to device
    this->convert_to_bsr_and_sync_device();
}

template<typename T>
void Mtx<T>::init_from_copy(Mtx<T> &cpymat)
{
    // perform deep copy for all members
    // yet memory location should be new

    // general info
    nrows = cpymat.nrows;
    ncols = cpymat.ncols;
    nnz = cpymat.nnz;

    // bsr info
    blockdim = cpymat.blockdim;
    nblockrows = cpymat.nblockrows;
    nblocks = cpymat.nblocks;

    // coo format on host
    coo_rowind_h.resize(cpymat.coo_rowind_h.size());
    coo_colind_h.resize(cpymat.coo_colind_h.size());
    coo_values_h.resize(cpymat.coo_values_h.size());
    memcpy(&coo_rowind_h[0], &cpymat.coo_rowind_h[0], 
           cpymat.coo_rowind_h.size()*sizeof(int));
    memcpy(&coo_colind_h[0], &cpymat.coo_colind_h[0], 
        cpymat.coo_colind_h.size()*sizeof(int));
    memcpy(&coo_values_h[0], &cpymat.coo_values_h[0], 
        cpymat.coo_values_h.size()*sizeof(T));

    // csr format on host
    csr_indptr_h.resize(cpymat.csr_indptr_h.size());
    csr_indices_h.resize(cpymat.csr_indices_h.size());
    csr_values_h.resize(cpymat.csr_values_h.size());
    memcpy(&csr_indptr_h[0], &cpymat.csr_indptr_h[0], 
           cpymat.csr_indptr_h.size()*sizeof(int));
    memcpy(&csr_indices_h[0], &cpymat.csr_indices_h[0], 
        cpymat.csr_indices_h.size()*sizeof(int));
    memcpy(&csr_values_h[0], &cpymat.csr_values_h[0], 
        cpymat.csr_values_h.size()*sizeof(T));

    // bsr format on host
    if (cpymat.bsr_converted) cpymat.get_bsr_host(); // to ensure host is non-empty

    bsr_indptr_h.resize(cpymat.bsr_indptr_h.size());
    bsr_indices_h.resize(cpymat.bsr_indices_h.size());
    bsr_values_h.resize(cpymat.bsr_values_h.size());
    memcpy(&bsr_indptr_h[0], &cpymat.bsr_indptr_h[0], 
           cpymat.bsr_indptr_h.size()*sizeof(int));
    memcpy(&bsr_indices_h[0], &cpymat.bsr_indices_h[0], 
        cpymat.bsr_indices_h.size()*sizeof(int));
    memcpy(&bsr_values_h[0], &cpymat.bsr_values_h[0], 
        cpymat.bsr_values_h.size()*sizeof(T));

    // pointers to transformed format on device
    SAFE_ALOC_GPU(device_ref.coo_rowind, cpymat.coo_rowind_h.size()*sizeof(int));
    SAFE_ALOC_GPU(device_ref.coo_colind, cpymat.coo_colind_h.size()*sizeof(int));
    SAFE_ALOC_GPU(device_ref.coo_values, cpymat.coo_values_h.size()*sizeof(T));
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.coo_rowind, cpymat.device_ref.coo_rowind,
                    cpymat.coo_rowind_h.size()*sizeof(int), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.coo_colind, cpymat.device_ref.coo_colind,
                    cpymat.coo_colind_h.size()*sizeof(int), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.coo_values, cpymat.device_ref.coo_values,
                    cpymat.coo_values_h.size()*sizeof(T), cudaMemcpyDeviceToDevice) );

    SAFE_ALOC_GPU(device_ref.csr_indptr, cpymat.csr_indptr_h.size()*sizeof(int));
    SAFE_ALOC_GPU(device_ref.csr_indices, cpymat.csr_indices_h.size()*sizeof(int));
    SAFE_ALOC_GPU(device_ref.csr_values, cpymat.csr_values_h.size()*sizeof(T));
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.csr_indptr, cpymat.device_ref.csr_indptr,
                    cpymat.csr_indptr_h.size()*sizeof(int), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.csr_indices, cpymat.device_ref.csr_indices,
                    cpymat.csr_indices_h.size()*sizeof(int), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.csr_values, cpymat.device_ref.csr_values,
                    cpymat.csr_values_h.size()*sizeof(T), cudaMemcpyDeviceToDevice) );
    
    SAFE_ALOC_GPU(device_ref.bsr_indptr, cpymat.bsr_indptr_h.size()*sizeof(int));
    SAFE_ALOC_GPU(device_ref.bsr_indices, cpymat.bsr_indices_h.size()*sizeof(int));
    SAFE_ALOC_GPU(device_ref.bsr_values, cpymat.bsr_values_h.size()*sizeof(T));
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.bsr_indptr, cpymat.device_ref.bsr_indptr,
                    cpymat.bsr_indptr_h.size()*sizeof(int), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.bsr_indices, cpymat.device_ref.bsr_indices,
                    cpymat.bsr_indices_h.size()*sizeof(int), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_ref.bsr_values, cpymat.device_ref.bsr_values,
                    cpymat.bsr_values_h.size()*sizeof(T), cudaMemcpyDeviceToDevice) );

    initialized = cpymat.initialized;
    device_synced = cpymat.device_synced;
    bsr_converted = cpymat.bsr_converted;
}

template<typename T>
void Mtx<T>::print()
{
    // general info
    std::cout   << "\n### Mtx ###\n"
                << "\nrows = " << nrows
                << "\ncols = " << ncols
                << "\nnnz = " << nnz;
    std::cout << std::endl;

    // bsr info
    std::cout   << "\nblockdim = " << blockdim
                << "\nnblockrows = " << nblockrows
                << "\nnblocks = " << nblocks;
    std::cout << std::endl;

    // coo format on host 
    std::cout    << "\n=== COO ===\n";
    std::cout    <<   "\ncoo_rowind_h: [ " ;
    for (int i=0; i<nnz; i++) std::cout << coo_rowind_h[i] << " ";
    std::cout    <<   "]\n" ;

    std::cout    <<   "\ncoo_colind_h: [ " ;
    for (int i=0; i<nnz; i++) std::cout << coo_colind_h[i] << " ";
    std::cout    <<   "]\n" ;

    std::cout    <<   "\ncoo_values_h: [ " ;
    for (int i=0; i<nnz; i++) std::cout << coo_values_h[i] << " ";
    std::cout    <<   "]\n" ;

    // csr format on host
    std::cout    << "\n=== CSR ===\n"; 
    std::cout    <<   "\ncsr_indptr_h: [ " ;
    for (int i=0; i<(nrows+1); i++) std::cout << csr_indptr_h[i] << " ";
    std::cout    <<   "]\n" ;

    std::cout    <<   "\ncsr_indices_h: [ " ;
    for (int i=0; i<nnz; i++) std::cout << csr_indices_h[i] << " ";
    std::cout    <<   "]\n" ;

    std::cout    <<   "\ncsr_values_h: [ " ;
    for (int i=0; i<nnz; i++) std::cout << csr_values_h[i] << " ";
    std::cout    <<   "]\n" ;

    // bsr
    this->get_bsr_host();
    std::cout    << "\n=== BSR ===\n";
    std::cout    <<   "\nbsr_indptr_h: [ " ;
    for (int i=0; i<(nblockrows+1); i++) std::cout << bsr_indptr_h[i] << " ";
    std::cout    <<   "]\n" ;

    std::cout    <<   "\nbsr_indices_h: [ " ;
    for (int i=0; i<nblocks; i++) std::cout << bsr_indices_h[i] << " ";
    std::cout    <<   "]\n" ;

    std::cout    <<   "\nbsr_values_h:\n" ;
    for (int i=0; i<nblocks; i++) {
        std::cout << "[" << i << "]"<< std::endl;
        for (int j=0; j<blockdim; j++) {
            for (int k=0; k<blockdim; k++) {
                std::cout << bsr_values_h[i*blockdim*blockdim+j*blockdim+k] << " ";
            }
            std::cout << std::endl;
        }
    }
}

template<typename T>
void Mtx<T>::convert_to_bsr_and_sync_device()
{
    // availability check
    if (!initialized || device_synced || bsr_converted) { 
        std::cout << "convert_to_bsr_and_sync_device() failed!" << std::endl; 
        return;
    }

    // sync coo to device
    SAFE_ALOC_GPU(this->device_ref.coo_rowind, this->nnz*sizeof(int));
    SAFE_ALOC_GPU(this->device_ref.coo_colind, this->nnz*sizeof(int));
    SAFE_ALOC_GPU(this->device_ref.coo_values, this->nnz*sizeof(T));
    CUDA_SAFE_CALL( cudaMemcpy(this->device_ref.coo_rowind, &this->coo_rowind_h[0],
                    this->nnz*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(this->device_ref.coo_colind, &this->coo_colind_h[0],
                    this->nnz*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(this->device_ref.coo_values, &this->coo_values_h[0],  
                    this->nnz*sizeof(T), cudaMemcpyHostToDevice) );

    // sync csr to device
    SAFE_ALOC_GPU(this->device_ref.csr_indptr, (this->nrows+1)*sizeof(int));
    SAFE_ALOC_GPU(this->device_ref.csr_indices, this->nnz*sizeof(int));
    SAFE_ALOC_GPU(this->device_ref.csr_values, this->nnz*sizeof(T));
    CUDA_SAFE_CALL( cudaMemcpy(this->device_ref.csr_indptr, &this->csr_indptr_h[0],
                    (this->nrows+1)*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(this->device_ref.csr_indices, &this->csr_indices_h[0],
                    this->nnz*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(this->device_ref.csr_values, &this->csr_values_h[0],  
                    this->nnz*sizeof(T), cudaMemcpyHostToDevice) );

    // convert csr to bsr in device
    cusparseMatDescr_t csr_descr = 0;
    cusparseMatDescr_t bsr_descr = 0;
    cudaStream_t streamId = 0;
    cusparseHandle_t handle = 0;

    this->nblockrows = (this->nrows + blockdim - 1) / blockdim;

    cusparseCreateMatDescr(&csr_descr);
    cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&bsr_descr);
    cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreate(&handle);
    cusparseSetStream(handle, streamId);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    SAFE_ALOC_GPU(this->device_ref.bsr_indptr, (this->nblockrows+1)*sizeof(int));
    cusparseXcsr2bsrNnz(handle, dirA, this->nrows, this->ncols, csr_descr,
                        this->device_ref.csr_indptr, this->device_ref.csr_indices, 
                        this->blockdim, bsr_descr, this->device_ref.bsr_indptr, &this->nblocks);
    
    SAFE_ALOC_GPU(this->device_ref.bsr_indices, this->nblocks*sizeof(int));
    SAFE_ALOC_GPU(this->device_ref.bsr_values, this->nblocks*this->blockdim*this->blockdim*sizeof(T));
    cusparseScsr2bsr(handle, dirA, this->nrows, this->ncols, 
                     csr_descr, 
                     this->device_ref.csr_values, this->device_ref.csr_indptr, this->device_ref.csr_indices, 
                     this->blockdim, bsr_descr, 
                     this->device_ref.bsr_values, this->device_ref.bsr_indptr, this->device_ref.bsr_indices);

    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);
    
    // set flag 
    device_synced = true;
    bsr_converted = true;
}

template<typename T>
void Mtx<T>::get_bsr_host() 
{
    if (!bsr_converted) { 
        std::cout << "get_bsr_host() failed: bsr not yet converted!" << std::endl; 
        return;
    }

    this->bsr_indptr_h.resize(this->nblockrows+1);
    this->bsr_indices_h.resize(this->nblocks);
    this->bsr_values_h.resize(this->nblocks*this->blockdim*this->blockdim);

    CUDA_SAFE_CALL( cudaMemcpy(&this->bsr_indptr_h[0], this->device_ref.bsr_indptr, 
                    (this->nblockrows+1)*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(&this->bsr_indices_h[0], this->device_ref.bsr_indices, 
                    this->nblocks*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(&this->bsr_values_h[0], this->device_ref.bsr_values, 
                    this->nblocks*this->blockdim*this->blockdim*sizeof(T), cudaMemcpyDeviceToHost) );
}

template<typename T>
void Mtx<T>::get_coo_host() 
{
    CUDA_SAFE_CALL( cudaMemcpy(&this->coo_rowind_h[0], this->device_ref.coo_rowind, 
                    this->nnz*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(&this->coo_colind_h[0], this->device_ref.coo_colind, 
                    this->nnz*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(&this->coo_values_h[0], this->device_ref.coo_values, 
                    this->nnz*sizeof(T), cudaMemcpyDeviceToHost) );
}

template<typename T>
void Mtx<T>::write_to_file(std::string &destfile)
{
    FILE *fptr = fopen(destfile.c_str(), "w");

    MM_typecode matcode;                        

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    mm_write_banner(fptr, matcode); 
    mm_write_mtx_crd_size(fptr, this->nrows, this->ncols, this->nnz);

    /* NOTE: matrix market files use 1-based indices, i.e. first element
      of a vector has index 1, not 0.  */
    this->get_coo_host();
    for (int i=0; i<this->nnz; i++)
        fprintf(fptr, "%d %d %.2f\n", this->coo_rowind_h[i]+1, this->coo_colind_h[i]+1, this->coo_values_h[i]);
        // fprintf(fptr, "%d %d %10.3g\n", this->coo_rowind_h[i]+1, this->coo_colind_h[i]+1, this->coo_values_h[i]);
    
    fclose(fptr);
}

template<typename T>
void Mtx<T>::reset()
{
    // get the latest coo
    this->get_coo_host();

    // redo the initialization for all
    // int *csrindptr, *csrcolind;
    // T *csrval;
    // SAFE_ALOC_HOST(csrindptr, (this->nrows+1)*sizeof(int));
    // SAFE_ALOC_HOST(csrcolind, this->nnz*sizeof(int));
    // SAFE_ALOC_HOST(csrval, this->nnz*sizeof(T));
    int *csrindptr = (int*)malloc((this->nrows+1) * sizeof(int));
    int *csrcolind = (int*)malloc(this->nnz * sizeof(int));
    T *csrval = (T*)malloc(this->nnz * sizeof(T));

    coo2csr(csrindptr, csrcolind, csrval,
            this->coo_rowind_h, this->coo_colind_h, this->coo_values_h, 
            this->nrows, this->ncols);
    
    this->csr_indptr_h.resize(this->nrows+1);
    this->csr_indices_h.resize(this->nnz);
    this->csr_values_h.resize(this->nnz);
    memcpy(&this->csr_indptr_h[0], csrindptr, (this->nrows+1)*sizeof(int));
    memcpy(&this->csr_indices_h[0], csrcolind, this->nnz*sizeof(int));
    memcpy(&this->csr_values_h[0], csrval, this->nnz*sizeof(T));
    free(csrindptr);
    free(csrcolind);
    free(csrval);

    // set flag
    initialized = true;
    if (device_synced) {
        // clear any old version
        this->__clear_device_ref();
        device_synced = false;
        bsr_converted = false;
    }

    // convert to bsr and sync storage to device
    this->convert_to_bsr_and_sync_device();
}

#endif