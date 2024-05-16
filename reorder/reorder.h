#ifndef REORDER_H
#define REORDER_H

#include "../mtx/Mtx.h"

#include <queue>
#include <unordered_map>
#include <map>
#include "swappair.h"

#define MAXTHRD 1024

typedef std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::function<bool(std::pair<int, int>&, std::pair<int, int>&)>> InvalSegList;
typedef std::priority_queue<SwapPair*, std::vector<SwapPair*>, std::function<bool(SwapPair*, SwapPair*)>> SwapPairList;
typedef std::priority_queue<SwapPairs*, std::vector<SwapPairs*>, std::function<bool(SwapPairs*, SwapPairs*)>> SwapPairsList;

// A faster way to obtain lane id in a warp
#define GET_LANEID              \
    unsigned laneid;            \
    asm("mov.u32 %0, %%laneid;" \
        : "=r"(laneid));

__global__ void Reorder(int *row, int *col, int *newid, const int nnz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz)
    {
        int u = row[idx];
        int v = col[idx];
        if (newid[u] != -1) 
            row[idx] = newid[u];
        if (newid[v] != -1) 
            col[idx] = newid[v];
    }
}

// binary search
template <typename Index>
__device__ Index binarySearchInd(const Index *array,
                                 Index target,
                                 Index begin,
                                 Index end)
{
    int mid;
    while (begin < end)
    {
        int mid = begin + (end - begin) / 2;
        int item = array[mid];
        if (item == target)
            return mid;
        bool larger = (item > target);
        if (larger)
            end = mid;
        else
            begin = mid + 1;
    }

    return -1;
}

__device__ void printBits(unsigned char num)
{
    unsigned char j;
    for (j = 1 << 7; j > 0; j = j / 2)
        (num & j) ? printf("1") : printf("0");
    printf("\n");
}

__device__ unsigned setBit(unsigned val, unsigned bit, int pos)
{
    return (val & ~(1 << (M-1-pos))) | (bit << (M-1-pos));
}

// define profitable swap
__global__ void GetSegSwapScore(int *score, 
                                const int* __restrict__ bsrrowptr, 
                                const int* __restrict__ bsrcolind, 
                                const float* __restrict__ bsrval, 
                                const int nrows, const int seg1, const int seg2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    GET_LANEID;

    if (idx < nrows)
    {
        // retrieve bitmap
        int id1 = binarySearchInd<int>(bsrcolind, seg1, bsrrowptr[idx/M], bsrrowptr[idx/M+1]);
        
        register unsigned lval = 0x00000000;
        if (id1 != -1)
        {
            for (int i=0; i<M; i++) 
                lval = (lval << 1) + (bsrval[id1*M*M+(laneid%M)*M+i] != 0.0f);
        }

        int id2 = binarySearchInd<int>(bsrcolind, seg2, bsrrowptr[idx/M], bsrrowptr[idx/M+1]);
        register unsigned rval = 0x00000000;
        if (id2 != -1)
        {
            for (int i=0; i<M; i++) 
                rval = (rval << 1) + (bsrval[id2*M*M+(laneid%M)*M+i] != 0.0f);
        }

        // swapping 1 bit between lval and rval
        register int lgain[M*M] = {0};
        register int rgain[M*M] = {0};

        for (int bitpos1=0; bitpos1<M; bitpos1++)
        {
            for (int bitpos2=0; bitpos2<M; bitpos2++)
            {
                unsigned bit1 = (lval >> (M-1-bitpos1)) & 0x1;
                unsigned bit2 = (rval >> (M-1-bitpos2)) & 0x1; 

                register unsigned lval_new = (lval & ~(1 << (M-1-bitpos1))) | (bit2 << (M-1-bitpos1));
                register unsigned rval_new = (rval & ~(1 << (M-1-bitpos2))) | (bit1 << (M-1-bitpos2));
                // if (idx == 3 && bitpos1==0 && bitpos2==0) printBits(rval_new);

                lgain[bitpos1*M+bitpos2] = ((__popc(lval) > N)?1:0) - ((__popc(lval_new) > N)?1:0);  // 1, 0, -1
                rgain[bitpos1*M+bitpos2] = ((__popc(rval) > N)?1:0) - ((__popc(rval_new) > N)?1:0);  // 1, 0, -1
            }
        }

        // store score
        for (int i=0; i<M*M; i++)
        {
            atomicAdd(&score[i*2+0], lgain[i]);
            atomicAdd(&score[i*2+1], rgain[i]);
        }
    }
}

__global__ void GetSegSwapScoreAll(int *score, 
                                   const int* __restrict__ bsrrowptr, 
                                   const int* __restrict__ bsrcolind, 
                                   const float* __restrict__ bsrval, 
                                   const int nrows, const int seg1, const int seg2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    GET_LANEID;

    if (idx < nrows)
    {
        // retrieve bitmap
        int id1 = binarySearchInd<int>(bsrcolind, seg1, bsrrowptr[idx/M], bsrrowptr[idx/M+1]);
        
        register unsigned lval = 0x00000000;
        if (id1 != -1)
        {
            for (int i=0; i<M; i++) 
                lval = (lval << 1) + (bsrval[id1*M*M+(laneid%M)*M+i] != 0.0f);
        }

        int id2 = binarySearchInd<int>(bsrcolind, seg2, bsrrowptr[idx/M], bsrrowptr[idx/M+1]);
        register unsigned rval = 0x00000000;
        if (id2 != -1)
        {
            for (int i=0; i<M; i++) 
                rval = (rval << 1) + (bsrval[id2*M*M+(laneid%M)*M+i] != 0.0f);
        }

        // swapping 1 bit between lval and rval
        register int lgain[52] = {0};
        register int rgain[52] = {0};

        // 0-15 for one swap
        for (int bitpos1=0; bitpos1<M; bitpos1++)
        {
            for (int bitpos2=0; bitpos2<M; bitpos2++)
            {
                unsigned bit1 = (lval >> (M-1-bitpos1)) & 0x1;
                unsigned bit2 = (rval >> (M-1-bitpos2)) & 0x1; 

                register unsigned lval_new = (lval & ~(1 << (M-1-bitpos1))) | (bit2 << (M-1-bitpos1));
                register unsigned rval_new = (rval & ~(1 << (M-1-bitpos2))) | (bit1 << (M-1-bitpos2));
                // if (idx == 3 && bitpos1==0 && bitpos2==0) printBits(rval_new);

                lgain[bitpos1*M+bitpos2] = ((__popc(lval) > N)?1:0) - ((__popc(lval_new) > N)?1:0);  // 1, 0, -1
                rgain[bitpos1*M+bitpos2] = ((__popc(rval) > N)?1:0) - ((__popc(rval_new) > N)?1:0);  // 1, 0, -1
            }
        }

        // 16-33 for two swaps
        register unsigned bit[8] = {0};
        register int pos[36][8] = {
            {1, 2, 6, 7, 0, 3, 4, 5}, // {(0, 6), (3, 7)}
            {1, 2, 5, 7, 0, 3, 4, 6}, // {(0, 5), (3, 7)}
            {1, 2, 5, 6, 0, 3, 4, 7}, // {(0, 5), (3, 6)}
            {1, 2, 4, 7, 0, 3, 5, 6}, // {(0, 4), (3, 7)}
            {1, 2, 4, 6, 0, 3, 5, 7}, // {(0, 4), (3, 6)}
            {1, 2, 4, 5, 0, 3, 6, 7}, // {(0, 4), (3, 5)}

            {0, 1, 4, 5, 2, 3, 6, 7}, // {(2, 4), (3, 5)}
            {0, 1, 4, 6, 2, 3, 5, 7}, // {(2, 4), (3, 6)}
            {0, 1, 4, 7, 2, 3, 5, 6}, // {(2, 4), (3, 7)}
            {0, 1, 5, 6, 2, 3, 4, 7}, // {(2, 5), (3, 6)}
            {0, 1, 5, 7, 2, 3, 4, 6}, // {(2, 5), (3, 7)}
            {0, 1, 6, 7, 2, 3, 4, 5}, // {(2, 6), (3, 7)}

            {0, 2, 4, 5, 1, 3, 6, 7}, // {(1, 4), (3, 5)}
            {0, 2, 4, 6, 1, 3, 5, 7}, // {(1, 4), (3, 6)}
            {0, 2, 4, 7, 1, 3, 5, 6}, // {(1, 4), (3, 7)}
            {0, 2, 5, 6, 1, 3, 4, 7}, // {(1, 5), (3, 6)}
            {0, 2, 5, 7, 1, 3, 4, 6}, // {(1, 5), (3, 7)}
            {0, 2, 6, 7, 1, 3, 4, 5}, // {(1, 6), (3, 7)}

            {0, 3, 4, 5, 1, 2, 6, 7}, // {(1, 4), (2, 5)}
            {0, 3, 4, 6, 1, 2, 5, 7}, // {(1, 4), (2, 6)}
            {0, 3, 4, 7, 1, 2, 5, 6}, // {(1, 4), (2, 7)}
            {0, 3, 5, 6, 1, 2, 4, 7}, // {(1, 5), (2, 6)}
            {0, 3, 5, 7, 1, 2, 4, 6}, // {(1, 5), (2, 7)}
            {0, 3, 6, 7, 1, 2, 4, 5}, // {(1, 6), (2, 7)}

            {1, 3, 6, 7, 0, 2, 4, 5}, // {(0, 6), (2, 7)}
            {1, 3, 5, 7, 0, 2, 4, 6}, // {(0, 5), (2, 7)}
            {1, 3, 5, 6, 0, 2, 4, 7}, // {(0, 5), (2, 6)}
            {1, 3, 4, 7, 0, 2, 5, 6}, // {(0, 4), (2, 7)}
            {1, 3, 4, 6, 0, 2, 5, 7}, // {(0, 4), (2, 6)}
            {1, 3, 4, 5, 0, 2, 6, 7}, // {(0, 4), (2, 5)}

            {2, 3, 4, 5, 0, 1, 6, 7}, // {(0, 4), (1, 5)}
            {2, 3, 4, 6, 0, 1, 5, 7}, // {(0, 4), (1, 6)}
            {2, 3, 4, 7, 0, 1, 5, 6}, // {(0, 4), (1, 7)}
            {2, 3, 5, 6, 0, 1, 4, 7}, // {(0, 5), (1, 7)}
            {2, 3, 5, 7, 0, 1, 4, 6}, // {(0, 5), (1, 6)}
            {2, 3, 6, 7, 0, 1, 4, 5}, // {(0, 6), (1, 5)}
        };

        // get bit at bit pos
        for (int i=0; i<M; i++)
        {
            bit[i] = (lval >> (M-1-i)) & 0x1; // lbit
            bit[M+i] = (rval >> (M-1-i)) & 0x1; // rbit
        }

        // get gain
        for (int i=0; i<36; i++)
        {
            register unsigned lval_new = 0x00000000;
            register unsigned rval_new = 0x00000000;

            // set bit
            for(int j=0; j<M; j++)
            {
                lval_new = setBit(lval_new, bit[pos[i][j]], j);
                rval_new = setBit(rval_new, bit[pos[i][M+j]], j);
            }

            lgain[16+i] = ((__popc(lval) > N)?1:0) - ((__popc(lval_new) > N)?1:0);
            rgain[16+i] = ((__popc(rval) > N)?1:0) - ((__popc(rval_new) > N)?1:0);
        }

        // store score
        for (int i=0; i<52; i++)
        {
            atomicAdd(&score[i*2+0], lgain[i]);
            atomicAdd(&score[i*2+1], rgain[i]);
        }
    }
}

__global__ void LocateinvalidSegment(int *result, 
                                     const int* __restrict__ bsrrowptr, 
                                     const int* __restrict__ bsrcolind, 
                                     const float* __restrict__ bsrval, 
                                     const int nrows, const int num_of_segment)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    GET_LANEID;

    if (idx < nrows && idy < num_of_segment)
    {  
        // retrieve bitmap
        int id = binarySearchInd<int>(bsrcolind, idy, bsrrowptr[idx/M], bsrrowptr[idx/M+1]);

        register unsigned val = 0x00000000;
        if (id != -1)
        {
            for (int i=0; i<M; i++) 
                val = (val << 1) + (bsrval[id*M*M+(laneid%M)*M+i] != 0.0f);
        }

        // warp vote to count
        register unsigned ivcnt = __popc((__ballot_sync(0xFFFFFFFF, __popc(val) > N)));
        if (laneid == 0) atomicAdd(&result[idy], ivcnt);
    }

}

__global__ void LocateAllZeroSegment(int *result, 
                                     const int* __restrict__ bsrrowptr, 
                                     const int* __restrict__ bsrcolind, 
                                     const float* __restrict__ bsrval, 
                                     const int nrows, const int num_of_segment)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    GET_LANEID;

    if (idx < nrows && idy < num_of_segment)
    {  
        // retrieve bitmap
        int id = binarySearchInd<int>(bsrcolind, idy, bsrrowptr[idx/M], bsrrowptr[idx/M+1]);

        register unsigned val = 0x00000000;
        if (id != -1)
        {
            for (int i=0; i<M; i++) 
                val = (val << 1) + (bsrval[id*M*M+(laneid%M)*M+i] != 0.0f);
        }

        // warp vote to count
        register unsigned ivcnt = __popc((__ballot_sync(0xFFFFFFFF, __popc(val) == 0)));
        if (laneid == 0) atomicAdd(&result[idy], ivcnt);
    }
}

template <typename T>
class reorderUtil {
public: 
    reorderUtil(Mtx<T> *_mat, int _maxiter): 
                mat(_mat), MAXITER(_maxiter) {
        
        // init num_of_segment
        num_of_segment = (((mat->nrows) + M - 1) / M);

        // init permute id list
        newid.resize(mat->nrows); 
        memset(&newid[0], -1, (this->mat)->nrows*sizeof(int));

        // init backup coo storage
        SAFE_ALOC_GPU(bckp_coo_row, (this->mat)->nnz*sizeof(int));
        SAFE_ALOC_GPU(bckp_coo_col, (this->mat)->nnz*sizeof(int));
    }
    ~reorderUtil() { 
    }

    // one step commit to newid 
    void swap(const int v1, const int v2) { newid[v1] = v2; newid[v2] = v1;}
    
    // total commit per iteration
    void reorder();

    // new heuristic
    InvalSegList locate_invalid_segment();
    InvalSegList locate_invalid_segment_reverse();
    SwapPairList get_segment_swap_score(const int seg1, const int seg2);
    SwapPairsList get_segment_swap_score_all(const int seg1, const int seg2);

    // main reorder function
    bool visited(const int v) { return this->newid[v] != -1; }
    bool seg_visited(const int seg);
    std::vector<int> run_old(bool verbose);
    std::vector<int> run(bool verbose, bool stepmsg);

    std::vector<int> run_allpairs(bool verbose, bool stepmsg);
    std::vector<int> run_allpairs_double(bool verbose, bool stepmsg);
    std::vector<int> run_allpairs_reverse(bool verbose, bool stepmsg);
    std::vector<int> run_allpairs_reverse_double(bool verbose, bool stepmsg); 
    
    std::vector<int> run_onepair(bool verbose, bool stepmsg);
    std::vector<int> run_onepair_double(bool verbose, bool stepmsg);
    std::vector<int> run_onepair_reverse(bool verbose, bool stepmsg);
    std::vector<int> run_onepair_reverse_double(bool verbose, bool stepmsg); 

    // run to completion
    std::vector<int> run_complete(bool verbose, bool stepmsg); 

    // stats
    int get_total_invalid_segment_cnt(InvalSegList isl);
    std::vector<int> locate_allzero_segment();
    std::vector<int> locate_zero_column_in_valid_segment(InvalSegList isl);
    int locate_min_outdeg_seg();

public:
    int MAXITER;
    int EXPLORE_RATE = 1;
    int EXPLOIT_RATE = 1;
    int topN = 1;

    int num_of_segment;

    Mtx<T> *mat;
    std::vector<int> newid;

    // for reorder backtrace
    int *bckp_coo_row;
    int *bckp_coo_col;
};

template<typename T>
void reorderUtil<T>::reorder() 
{
    // backup current coo for backtracing
    CUDA_SAFE_CALL( cudaMemcpy(this->bckp_coo_row, (this->mat)->device_ref.coo_rowind, 
                    (this->mat)->nnz*sizeof(int), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(this->bckp_coo_col, (this->mat)->device_ref.coo_colind, 
                    (this->mat)->nnz*sizeof(int), cudaMemcpyDeviceToDevice) );  

    // parallelize to gpu, O(nnz) -> O(1)
    int *newid_gpu;
    SAFE_ALOC_GPU(newid_gpu, (this->mat)->nrows*sizeof(int));
    CUDA_SAFE_CALL( cudaMemcpy(newid_gpu, &this->newid[0], 
                    (this->mat)->nrows*sizeof(int), cudaMemcpyHostToDevice) );  
    dim3 GRID(((this->mat)->nnz+MAXTHRD-1)/MAXTHRD);
    dim3 BLOCK(MAXTHRD);
    Reorder<<<GRID, BLOCK>>>((this->mat)->device_ref.coo_rowind, 
                             (this->mat)->device_ref.coo_colind, 
                             newid_gpu, (this->mat)->nnz);
    SAFE_FREE_GPU(newid_gpu);

    // reset newid
    memset(&this->newid[0], -1, (this->mat)->nrows*sizeof(int));

    // reset matrix
    (this->mat)->reset(); // use coo format to update
}


// sort by decreasing cnt
bool cmp(std::pair<int, int>& a, std::pair<int, int>& b) 
{
    // if cnt are the same, sort by seg ID
    if (a.second == b.second) return a.first > b.first;
    return a.second < b.second; 
} 

bool cmp_reverse(std::pair<int, int>& a, std::pair<int, int>& b) 
{
    // if cnt are the same, sort by seg ID
    if (a.second == b.second) return a.first < b.first;
    return a.second > b.second; 
} 

template<typename T>
InvalSegList reorderUtil<T>::locate_invalid_segment()
{
    int *result_gpu;
    SAFE_ALOC_GPU(result_gpu, this->num_of_segment*sizeof(int));
    CUDA_SAFE_CALL( cudaMemset(result_gpu, 0, this->num_of_segment*sizeof(int)) );
    
    dim3 GRID(((this->mat)->nrows+MAXTHRD-1)/MAXTHRD, this->num_of_segment);
    dim3 BLOCK(MAXTHRD, 1);
    LocateinvalidSegment<<<GRID, BLOCK>>>(result_gpu, 
                                         (this->mat)->device_ref.bsr_indptr, 
                                         (this->mat)->device_ref.bsr_indices,
                                         (this->mat)->device_ref.bsr_values,
                                         (this->mat)->nrows, this->num_of_segment);

    std::vector<int> result(this->num_of_segment);
    CUDA_SAFE_CALL( cudaMemcpy(&result[0], result_gpu,
                    this->num_of_segment*sizeof(int), cudaMemcpyDeviceToHost) );  
    SAFE_FREE_GPU(result_gpu);

    InvalSegList invresult(cmp);
    for(int i=0; i<this->num_of_segment; i++) if (result[i] != 0) invresult.push({i, result[i]});

    // showpq(invresult);

    return invresult;
}

template<typename T>
InvalSegList reorderUtil<T>::locate_invalid_segment_reverse()
{
    int *result_gpu;
    SAFE_ALOC_GPU(result_gpu, this->num_of_segment*sizeof(int));
    CUDA_SAFE_CALL( cudaMemset(result_gpu, 0, this->num_of_segment*sizeof(int)) );
    
    dim3 GRID(((this->mat)->nrows+MAXTHRD-1)/MAXTHRD, this->num_of_segment);
    dim3 BLOCK(MAXTHRD, 1);
    LocateinvalidSegment<<<GRID, BLOCK>>>(result_gpu, 
                                         (this->mat)->device_ref.bsr_indptr, 
                                         (this->mat)->device_ref.bsr_indices,
                                         (this->mat)->device_ref.bsr_values,
                                         (this->mat)->nrows, this->num_of_segment);

    std::vector<int> result(this->num_of_segment);
    CUDA_SAFE_CALL( cudaMemcpy(&result[0], result_gpu,
                    this->num_of_segment*sizeof(int), cudaMemcpyDeviceToHost) );  
    SAFE_FREE_GPU(result_gpu);

    // std::vector<std::pair<int, int>> invresult;
    // for(int i=0; i<this->num_of_segment; i++) if (result[i] != 0) invresult.push_back({i, result[i]});

    // sort by invalid count
    // std::sort(invresult.begin(), invresult.end(), cmp); 

    InvalSegList invresult(cmp_reverse);
    for(int i=0; i<this->num_of_segment; i++) if (result[i] != 0) invresult.push({i, result[i]});

    // showpq(invresult);

    return invresult;
}

template<typename T>
SwapPairList reorderUtil<T>::get_segment_swap_score(const int seg1, const int seg2)
{
    int *result_gpu;
    SAFE_ALOC_GPU(result_gpu, M*M*2*sizeof(int));
    CUDA_SAFE_CALL( cudaMemset(result_gpu, 0, M*M*2*sizeof(int)) );
    
    dim3 GRID(((this->mat)->nrows+MAXTHRD-1)/MAXTHRD);
    dim3 BLOCK(MAXTHRD);
    GetSegSwapScore<<<GRID, BLOCK>>>(result_gpu, 
                                    (this->mat)->device_ref.bsr_indptr, 
                                    (this->mat)->device_ref.bsr_indices,
                                    (this->mat)->device_ref.bsr_values,
                                    (this->mat)->nrows, seg1, seg2);

    std::vector<int> result(M*M*2);
    CUDA_SAFE_CALL( cudaMemcpy(&result[0], result_gpu,
                    M*M*2*sizeof(int), cudaMemcpyDeviceToHost) );  
    SAFE_FREE_GPU(result_gpu);

    // get a 16-by-16 {lgain, rgain}
    // ideally, both linvcnt-lgain & rinvcnt-rgain eqauls to 0
    // if not, we get at best rinvcnt-rgain = 0, then remove R in set
    SwapPairList pq(Comparator);
    for (int i=0; i<M*M; i++)
    {
        int lgain = result[i*2+0], rgain = result[i*2+1];
        // if ((rgain > 0) && (lgain + rgain > 0)) // option1: ensure rgain is positive (target seg's cnt is decreasing)
        // if ((lgain >= 0) && (rgain >= 0) && (lgain + rgain > 0)) // option2: filter out the negative gains
        // option3: don't do any filtering at all
        // {
            SwapPair *p = new SwapPair(seg1*M+(i/M), seg2*M+(i%M), lgain, rgain);
            pq.push(p);
        // }
    }

    return pq;
}

template<typename T>
SwapPairsList reorderUtil<T>::get_segment_swap_score_all(const int seg1, const int seg2)
{
    int *result_gpu;
    SAFE_ALOC_GPU(result_gpu, 52*2*sizeof(int));
    CUDA_SAFE_CALL( cudaMemset(result_gpu, 0, 52*2*sizeof(int)) );

    dim3 GRID(((this->mat)->nrows+MAXTHRD-1)/MAXTHRD);
    dim3 BLOCK(MAXTHRD);
    GetSegSwapScoreAll<<<GRID, BLOCK>>>(result_gpu, 
                                        (this->mat)->device_ref.bsr_indptr, 
                                        (this->mat)->device_ref.bsr_indices,
                                        (this->mat)->device_ref.bsr_values,
                                        (this->mat)->nrows, seg1, seg2);

    std::vector<int> result(52*2);
    CUDA_SAFE_CALL( cudaMemcpy(&result[0], result_gpu,
                    52*2*sizeof(int), cudaMemcpyDeviceToHost) );  
    SAFE_FREE_GPU(result_gpu);

    // get a 16-by-16 {lgain, rgain}
    // ideally, both linvcnt-lgain & rinvcnt-rgain eqauls to 0
    // if not, we get at best rinvcnt-rgain = 0, then remove R in set
    SwapPairsList pq(Comparators);
    for (int i=0; i<M*M; i++)
    {
        int lgain = result[i*2+0], rgain = result[i*2+1];
        // if ((rgain > 0) && (lgain + rgain > 0)) // option1: ensure rgain is positive (target seg's cnt is decreasing)
        // if ((lgain >= 0) && (rgain >= 0) && (lgain + rgain > 0)) // option2: filter out the negative gains
        // option3: don't do any filtering at all
        // {
            SwapPairs *p = new SwapPairs(seg1*M+(i/M), seg2*M+(i%M), lgain, rgain);
            pq.push(p);
        // }
    }

    for (int i=M*M; i<52; i++)
    {
        int lgain = result[i*2+0], rgain = result[i*2+1];
        // if ((rgain > 0) && (lgain + rgain > 0)) // option1: ensure rgain is positive (target seg's cnt is decreasing)
        // if ((lgain >= 0) && (rgain >= 0) && (lgain + rgain > 0)) // option2: filter out the negative gains
        // option3: don't do any filtering at all
        // {
            SwapPairs *p = new SwapPairs(i-M*M, seg1, seg2, lgain, rgain);
            pq.push(p);
        // }
    }

    return pq;
}

template<typename T>
int reorderUtil<T>::get_total_invalid_segment_cnt(InvalSegList isl)
{
    InvalSegList g = isl;
    int cnt = 0;
    while (!g.empty()) {
        cnt += (g.top()).second;
        g.pop();
    }

    return cnt;
}

template<typename T>
std::vector<int> reorderUtil<T>::locate_allzero_segment()
{
    int *result_gpu;
    SAFE_ALOC_GPU(result_gpu, this->num_of_segment*sizeof(int));
    CUDA_SAFE_CALL( cudaMemset(result_gpu, 0, this->num_of_segment*sizeof(int)) );
    
    dim3 GRID(((this->mat)->nrows+MAXTHRD-1)/MAXTHRD, this->num_of_segment);
    dim3 BLOCK(MAXTHRD, 1);
    LocateAllZeroSegment<<<GRID, BLOCK>>>(result_gpu, 
                                         (this->mat)->device_ref.bsr_indptr, 
                                         (this->mat)->device_ref.bsr_indices,
                                         (this->mat)->device_ref.bsr_values,
                                         (this->mat)->nrows, this->num_of_segment);

    std::vector<int> result(this->num_of_segment);
    CUDA_SAFE_CALL( cudaMemcpy(&result[0], result_gpu,
                    this->num_of_segment*sizeof(int), cudaMemcpyDeviceToHost) );  
    SAFE_FREE_GPU(result_gpu);

    std::vector<int> allzerosegs;
    for(int i=0; i<this->num_of_segment; i++) if (result[i] == (this->mat)->nrows) allzerosegs.push_back(i);

    return allzerosegs;
}

template<typename T>
std::vector<int> reorderUtil<T>::locate_zero_column_in_valid_segment(InvalSegList isl)
{
    // O(n) to find missing 
    std::vector<int> temp((this->mat)->nrows, 0);
    for(int i = 0; i < (this->mat)->coo_colind_h.size(); i++){
      temp[(this->mat)->coo_colind_h[i]] = 1;
    }

    // construct invalid segment's hash table
    unordered_map<int, int> umap;
    while(!isl.empty()) {
        umap[isl.top().first] = 1; // dummy
        isl.pop();
    }

    // add and filter out those in invalid segments
    std::vector<int> result;
    for (int i = 0; i < (this->mat)->nrows; i++) {
        if (temp[i] == 0 && umap.find(i/M) == umap.end()) result.push_back(i);
    }

    return result;
}

template<typename T>
std::vector<int> reorderUtil<T>::run_old(bool verbose=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // if (init_cnt/8 > MAXITER) MAXITER = init_cnt/8; // TODO: param to be tuned

    while (inv_seg_list.size() > 1 && round <= MAXITER) {

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        // set primary seg & cnt
        primary_seg = inv_seg_list.top().first;
        primary_cnt = inv_seg_list.top().second;
        inv_seg_list.pop();

        if (verbose) { 
            std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
            ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
        }

        // as primary seg still has invalid segs & not all nodes are visited
        while (primary_cnt > 0 && inv_seg_list.size() >= 1) {

            target_seg = inv_seg_list.top().first;
            target_cnt = inv_seg_list.top().second;

            if (verbose) { 
                std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
            }

            // retrive swap list
            SwapPairList score = this->get_segment_swap_score(primary_seg, target_seg);
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }

            // if lseg's node is visited, pop top
            if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
            while (!score.empty() && visited((score.top())->v1)) { 
                if (verbose) std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl; 
                score.pop(); 
            }
            
            // otherwise, swap the toppest pair
            if (!score.empty() && !visited((score.top())->v1))
            {
                SwapPair *p = score.top();
                this->swap(p->v1, p->v2);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }
            }

            // pop target as this iter has done
            // if there's even no pair to swap in this target
            // drop the target and move on
            inv_seg_list.pop();

            // int x; std::cin >> x;
        }

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment();

        // for new round
        round += 1;
    } 


    inv_seg_list = this->locate_invalid_segment();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
bool reorderUtil<T>::seg_visited(const int seg) {
    bool flag = true;
    for(int i=0; i<M; i++) {
        // if contain unvisited, return false
        if (!visited(seg*M+i)) return false;
    }
    return flag;
}


template<typename T>
int reorderUtil<T>::locate_min_outdeg_seg()
{
    int *result_gpu;
    SAFE_ALOC_GPU(result_gpu, this->num_of_segment*sizeof(int));
    CUDA_SAFE_CALL( cudaMemset(result_gpu, 0, this->num_of_segment*sizeof(int)) );
    
    dim3 GRID(((this->mat)->nrows+MAXTHRD-1)/MAXTHRD, this->num_of_segment);
    dim3 BLOCK(MAXTHRD, 1);
    LocateAllZeroSegment<<<GRID, BLOCK>>>(result_gpu, 
                                         (this->mat)->device_ref.bsr_indptr, 
                                         (this->mat)->device_ref.bsr_indices,
                                         (this->mat)->device_ref.bsr_values,
                                         (this->mat)->nrows, this->num_of_segment);

    std::vector<int> result(this->num_of_segment);
    CUDA_SAFE_CALL( cudaMemcpy(&result[0], result_gpu,
                    this->num_of_segment*sizeof(int), cudaMemcpyDeviceToHost) );  
    SAFE_FREE_GPU(result_gpu);

    std::vector<std::pair<int,int>> minodsegs;
    for (int i=0; i<this->num_of_segment; i++) minodsegs.push_back(std::make_pair(result[i], i));

    sort(minodsegs.begin(), minodsegs.end());

    // for (int i=0; i<this->num_of_segment; i++) {
    //     std::cout << "(" << minodsegs[i].first << ", " << minodsegs[i].second << ")" << std::endl; 
    // }

    return minodsegs[this->num_of_segment-1].second;
}

template<typename T>
std::vector<int> reorderUtil<T>::run_complete(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            // SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            SwapPair *p = score.top();
            this->swap(p->v1, p->v2);
            // SwapPairs *p = score.top();
            // this->swap(p->v1, p->v2);
            // if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ = inv_seg_list;
            InvalSegList updatePQ(cmp);
            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairList score = this->get_segment_swap_score(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() && (visited((score.top())->v1) || visited((score.top())->v2))) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPair *p = score.top();
                this->swap(p->v1, p->v2);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 2) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ = inv_seg_list;
            InvalSegList updatePQ(cmp);
            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairList score = this->get_segment_swap_score(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() && (visited((score.top())->v1) || visited((score.top())->v2))) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPair *p = score.top();
                this->swap(p->v1, p->v2);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_allpairs(bool verbose=false, bool stepmsg=false)
{
    InvalSegList inv_seg_list = this->locate_invalid_segment();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            // SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            // SwapPair *p = score.top();
            // this->swap(p->v1, p->v2);
            SwapPairs *p = score.top();
            this->swap(p->v1, p->v2);
            if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ = inv_seg_list;
            InvalSegList updatePQ(cmp);
            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairsList score = this->get_segment_swap_score_all(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() 
                    && (visited((score.top())->v1) || visited((score.top())->v2))
                    && (((score.top())->v3 != -1) && visited((score.top())->v3) || 
                    ((score.top())->v4 != -1) && visited((score.top())->v4)) ) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v3 != -1) && visited((score.top())->v3))
                            std::cout << (score.top())->v3 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v4 != -1) && visited((score.top())->v4))
                            std::cout << (score.top())->v4 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPairs *p = score.top();
                this->swap(p->v1, p->v2);
                if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }
    
    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_allpairs_double(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            // SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            // SwapPair *p = score.top();
            // this->swap(p->v1, p->v2);
            SwapPairs *p = score.top();
            this->swap(p->v1, p->v2);
            if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            // the traverse queue will be reversed
            InvalSegList tempPQ(cmp_reverse);
            while (!inv_seg_list.empty()) { 
                tempPQ.push(inv_seg_list.top()); 
                inv_seg_list.pop(); 
            }

            // substitue queue will be in same order as inv_seg_list
            InvalSegList updatePQ(cmp);

            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairsList score = this->get_segment_swap_score_all(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() 
                    && (visited((score.top())->v1) || visited((score.top())->v2))
                    && (((score.top())->v3 != -1) && visited((score.top())->v3) || 
                    ((score.top())->v4 != -1) && visited((score.top())->v4)) ) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v3 != -1) && visited((score.top())->v3))
                            std::cout << (score.top())->v3 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v4 != -1) && visited((score.top())->v4))
                            std::cout << (score.top())->v4 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPairs *p = score.top();
                this->swap(p->v1, p->v2);
                if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_allpairs_reverse(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment_reverse();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment_reverse();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            // SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            // SwapPair *p = score.top();
            // this->swap(p->v1, p->v2);
            SwapPairs *p = score.top();
            this->swap(p->v1, p->v2);
            if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ = inv_seg_list;
            InvalSegList updatePQ(cmp_reverse);
            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairsList score = this->get_segment_swap_score_all(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() 
                    && (visited((score.top())->v1) || visited((score.top())->v2))
                    && (((score.top())->v3 != -1) && visited((score.top())->v3) || 
                    ((score.top())->v4 != -1) && visited((score.top())->v4)) ) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v3 != -1) && visited((score.top())->v3))
                            std::cout << (score.top())->v3 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v4 != -1) && visited((score.top())->v4))
                            std::cout << (score.top())->v4 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPairs *p = score.top();
                this->swap(p->v1, p->v2);
                if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment_reverse();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment_reverse();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment_reverse();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_allpairs_reverse_double(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment_reverse();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {
        
        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment_reverse();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            // SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            // SwapPair *p = score.top();
            // this->swap(p->v1, p->v2);
            SwapPairs *p = score.top();
            this->swap(p->v1, p->v2);
            if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ(cmp);
            while (!inv_seg_list.empty()) { 
                tempPQ.push(inv_seg_list.top()); 
                inv_seg_list.pop(); 
            }

            // substitue queue will be in same order as inv_seg_list
            InvalSegList updatePQ(cmp_reverse);

            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairsList score = this->get_segment_swap_score_all(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() 
                    && (visited((score.top())->v1) || visited((score.top())->v2))
                    && (((score.top())->v3 != -1) && visited((score.top())->v3) || 
                    ((score.top())->v4 != -1) && visited((score.top())->v4)) ) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v3 != -1) && visited((score.top())->v3))
                            std::cout << (score.top())->v3 << " has been visited in this round!" << std::endl;
                        if (((score.top())->v4 != -1) && visited((score.top())->v4))
                            std::cout << (score.top())->v4 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPairs *p = score.top();
                this->swap(p->v1, p->v2);
                if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment_reverse();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment_reverse();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment_reverse();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_onepair(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            // SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            SwapPair *p = score.top();
            this->swap(p->v1, p->v2);
            // SwapPairs *p = score.top();
            // this->swap(p->v1, p->v2);
            // if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ = inv_seg_list;
            InvalSegList updatePQ(cmp);
            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairList score = this->get_segment_swap_score(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() && (visited((score.top())->v1) || visited((score.top())->v2))) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPair *p = score.top();
                this->swap(p->v1, p->v2);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }
    
    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_onepair_double(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            // SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            SwapPair *p = score.top();
            this->swap(p->v1, p->v2);
            // SwapPairs *p = score.top();
            // this->swap(p->v1, p->v2);
            // if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            // the traverse queue will be reversed
            InvalSegList tempPQ(cmp_reverse);
            while (!inv_seg_list.empty()) { 
                tempPQ.push(inv_seg_list.top()); 
                inv_seg_list.pop(); 
            }

            // substitue queue will be in same order as inv_seg_list
            InvalSegList updatePQ(cmp);

            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairList score = this->get_segment_swap_score(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() && (visited((score.top())->v1) || visited((score.top())->v2))) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPair *p = score.top();
                this->swap(p->v1, p->v2);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_onepair_reverse(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment_reverse();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {

        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment_reverse();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            // SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            SwapPair *p = score.top();
            this->swap(p->v1, p->v2);
            // SwapPairs *p = score.top();
            // this->swap(p->v1, p->v2);
            // if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ = inv_seg_list;
            InvalSegList updatePQ(cmp_reverse);
            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairList score = this->get_segment_swap_score(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() && (visited((score.top())->v1) || visited((score.top())->v2))) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPair *p = score.top();
                this->swap(p->v1, p->v2);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment_reverse();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment_reverse();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment_reverse();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}

template<typename T>
std::vector<int> reorderUtil<T>::run_onepair_reverse_double(bool verbose=false, bool stepmsg=false) 
{
    InvalSegList inv_seg_list = this->locate_invalid_segment_reverse();
    int init_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int init_size = inv_seg_list.size();
    int primary_seg, primary_cnt, target_seg, target_cnt;
    int round = 1;
    
    // reset max iter
    // option 1: pat delta
    // if (init_cnt/48 > MAXITER) MAXITER = init_cnt/48;

    // option 2: list delta
    if (init_size > MAXITER) MAXITER = init_size;

    while (round <= MAXITER && inv_seg_list.size() >= 1) {
        
        if (stepmsg) {
            ///////////////////////////////////////////////////////
            InvalSegList l = this->locate_invalid_segment_reverse();
            int cnt = this->get_total_invalid_segment_cnt(l);
            int size = l.size();
            std::cout << round << ", " << size << ", " << cnt << std::endl;
            ///////////////////////////////////////////////////////
        }

        if (verbose) { 
            std::cout << "===== Invalid Segment List (round " + std::to_string(round) + ") =====" << std::endl;
            showpq(inv_seg_list);
        }

        if (inv_seg_list.size() == 1)
        {
            int minseg = this->locate_min_outdeg_seg();

            // retrive swap list // TODO: try aggressive all pair if still impossible
            SwapPairList score = this->get_segment_swap_score(inv_seg_list.top().first, minseg);     
            // SwapPairsList score = this->get_segment_swap_score_all(inv_seg_list.top().first, minseg);    
            if (verbose) { 
                std::cout << "===== Swap Pair List =====" << std::endl;
                showpq(score);
            }
            
            // otherwise, swap the toppest pair
            SwapPair *p = score.top();
            this->swap(p->v1, p->v2);
            // SwapPairs *p = score.top();
            // this->swap(p->v1, p->v2);
            // if (p->v3 != -1 && p->v4 != -1) this->swap(p->v3, p->v4);

            if (verbose) {
                std::cout << "===== ##Swap## =====" << std::endl; 
                p->print();
            }
        }
        
        while (inv_seg_list.size() >= 2)
        {
            // set primary seg & cnt
            primary_seg = inv_seg_list.top().first;
            primary_cnt = inv_seg_list.top().second;
            inv_seg_list.pop(); // delete it as long as it has been assigned primary
            if (verbose) { 
                std::cout << "===== primary_seg: " + std::to_string(primary_seg) + \
                ", primary_cnt: " + std::to_string(primary_cnt) + "=====" << std::endl;
            }

            // tempPQ and updatePQ for better traverse and update
            InvalSegList tempPQ(cmp);
            while (!inv_seg_list.empty()) { 
                tempPQ.push(inv_seg_list.top()); 
                inv_seg_list.pop(); 
            }

            // substitue queue will be in same order as inv_seg_list
            InvalSegList updatePQ(cmp_reverse);

            while (!tempPQ.empty()) {
                // as primary seg still has invalid segs & not all nodes are visited
                if (primary_cnt <= 0 || seg_visited(primary_seg)) break;

                // set target seg & cnt
                target_seg = (tempPQ.top()).first;
                target_cnt = (tempPQ.top()).second;
                if (verbose) { 
                    std::cout << "===== target_seg: " + std::to_string(target_seg) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // retrive swap list
                SwapPairList score = this->get_segment_swap_score(primary_seg, target_seg);      
                if (verbose) { 
                    std::cout << "===== Swap Pair List =====" << std::endl;
                    showpq(score);
                }

                // if lseg's node is visited, pop top
                if (verbose) std::cout << "===== Filter out the visited nodes in list: =====" << std::endl;
                while (!score.empty() && (visited((score.top())->v1) || visited((score.top())->v2))) { 
                    if (verbose) { 
                        if (visited((score.top())->v1))
                            std::cout << (score.top())->v1 << " has been visited in this round!" << std::endl;
                        if (visited((score.top())->v2))
                            std::cout << (score.top())->v2 << " has been visited in this round!" << std::endl;
                    }
                    score.pop(); // O(logN)
                }
                
                // otherwise, swap the toppest pair // TODO: let here do swapping for more than 1 pair
                SwapPair *p = score.top();
                this->swap(p->v1, p->v2);
                primary_cnt -= p->left_gain;
                target_cnt -= p->right_gain;

                if (verbose) {
                    std::cout << "===== ##Swap## =====" << std::endl; 
                    p->print();
                    std::cout << "===== primary_cnt: " + std::to_string(primary_cnt) + \
                    ", target_cnt: " + std::to_string(target_cnt) + "=====" << std::endl;
                }

                // preserve target in the PQ for next iter
                if (target_cnt > 0) { updatePQ.push({target_seg, target_cnt}); } 
                
                // pop current target and move on to next
                tempPQ.pop();
                
                // int x; std::cin >> x;
            } // for rhs

            // update inv_seg_list after traversing
            inv_seg_list = updatePQ;
            if (verbose) std::cout << "===== target traversal done, move on to next primary =====" << std::endl;
        } // while lhs

        // reorder
        if (verbose) std::cout << "===== ##Reorder## =====" << std::endl; 
        this->reorder();

        // locate invalid list for new round
        inv_seg_list = this->locate_invalid_segment_reverse();

        // for new round
        round += 1;
    }

    // get completion info
    inv_seg_list = this->locate_invalid_segment_reverse();
    int final_cnt = this->get_total_invalid_segment_cnt(inv_seg_list);
    int final_size = inv_seg_list.size();

    if (verbose) { 
        std::cout<< "===== Only 1 or 0 invalid segment left ==> Complete =====" << std::endl;
        showpq(inv_seg_list);
        std::cout<< "===== init_cnt: " << init_cnt << ", final_cnt: " << final_cnt \
        << ", improve_rate: " << (init_cnt-final_cnt)*1.0f/init_cnt << " =====" << std::endl;
    }

    if (stepmsg) {
        ///////////////////////////////////////////////////////
        InvalSegList l = this->locate_invalid_segment_reverse();
        int cnt = this->get_total_invalid_segment_cnt(l);
        int size = l.size();
        std::cout << round << ", " << size << ", " << cnt << std::endl;
        ///////////////////////////////////////////////////////
    }

    return {init_size, final_size, init_cnt, final_cnt, round-1};
}
#endif