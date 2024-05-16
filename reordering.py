import argparse
import numpy as np
import sys

from scipy.sparse import coo_matrix, coo_array
from scipy.io import mmread, mmwrite

parser = argparse.ArgumentParser(
    description='Extract Sparse Matrices Attributes')
parser.add_argument('--mtxfile', type=str, default="temp.mtx",
                    help="mtx file path")
parser.add_argument("--device", default='0', type=str, help='gpu id')
parser.add_argument("--v", default=32, type=int, help='the V param in VENOM pattern')
parser.add_argument("--m", default=8, type=int, help='the M param in VENOM pattern')
parser.add_argument("--maxiter", default=10, type=int, help='the MAXITER of the algorithm')
parser.add_argument("--evalonly", action='store_true', help='eval VENOM pattern only')

args = parser.parse_args()

# read mtx attributes from file
# with open(args.mtxfile, "r") as bm:
#     while True:
#         text = bm.readline().strip()
#         if text[0] != "%":
#             # print(text)
#             m, n, nnz = text.split(" ")
#             break
#         else:
#             continue
#     m, n, nnz = int(m), int(n), int(nnz)
#     row_indices = []
#     col_indices = []
#     values = []
#     lines = bm.readlines()
#     for j in range(len(lines)):
#         text = lines[j].strip().split(" ")
#         row_indices.append(int(text[0])-1)
#         col_indices.append(int(text[1])-1)
#         values.append(1 if len(text) == 2 else int(float(text[2])))

mm = mmread(args.mtxfile)
row_indices, col_indices, values = mm.row.tolist(), mm.col.tolist(), mm.data.tolist()
m, n = mm.shape
nnz = mm.nnz


M = args.m

# pad m and n
m = int((m + M - 1) // M) * M
n = int((n + M - 1) // M) * M
#print(m , " ", n , " ", nnz)

# # init a dense matrix    
# mat = ['0' * n for i in range(m)]
# for i in range(nnz):
#     tmp = list(mat[row_indices[i]])
#     tmp[col_indices[i]] = '1'
#     mat[row_indices[i]] = ''.join(tmp)

# for i in range(m):
#     print(mat[i])
        

##### grey code #####
def hamming_distance(bit_string1, bit_string2):
    # Ensure that the input bit-strings are of the same length
    if len(bit_string1) != len(bit_string2):
        raise ValueError("Bit strings must be of the same length")

    # Initialize the Hamming distance
    distance = 0

    # Compare each bit in the two strings
    for bit1, bit2 in zip(bit_string1, bit_string2):
        if bit1 != bit2:
            distance += 1

    return distance

def generate_gray_code(n):
    if n == 0:
        return ["0"]
    elif n == 1:
        return ["0", "1"]

    gray_code = generate_gray_code(n - 1)
    reflected_gray_code = gray_code[::-1]

    gray_code = ["0" + code for code in gray_code]
    reflected_gray_code = ["1" + code for code in reflected_gray_code]

    return gray_code + reflected_gray_code

def construct_bitcode_hash_tab(nb=8):
    gray_code = generate_gray_code(nb)
    bithashtab = {}
    cnt = 0
    for code in gray_code:
        if code.count('1') <= 2: 
            # print(cnt, " ", code)
            bithashtab[code] = cnt 
        else: 
            # print(-cnt, " ", code)
            bithashtab[code] = -cnt 
        cnt += 1
    return bithashtab

bithashtab = construct_bitcode_hash_tab(M)
#print(bithashtab)

# segmented mat
def generate_segmat(row_indices, col_indices):
    segmat = [['0' * M for j in range(n//M)] for i in range(m)]
    for i in range(nnz):
        tmp = list(segmat[row_indices[i]][col_indices[i]//M])
        tmp[col_indices[i]%M] = '1'
        segmat[row_indices[i]][col_indices[i]//M] = ''.join(tmp)
    return segmat

def print_segmat(segmat):
    for i in range(m):
        print(segmat[i])

def count_invalid_venom_block(segmat, V):
    inv_venom_blk = 0
    for seg in range(n//M):
        for i in range(0, int((m+V-1)//V*V), V):
            bval = 00000000
            for j in range(i, i+V):
                if j < m:
                    bval |= int(segmat[j][seg], 2)
            bval_str = bin(bval)[2:]
            if bval_str.count('1') > 4:
                inv_venom_blk += 1
    return inv_venom_blk

def count_invalid_venom_block_strict(segmat, V):
    inv_venom_blk = 0
    for seg in range(n//M):
        for i in range(0, int((m+V-1)//V*V), V):
            bval = 00000000
            invalid_nm = False
            for j in range(i, i+V):
                if j < m:
                    if segmat[j][seg].count('1') > 2:
                        invalid_nm = True
                    bval |= int(segmat[j][seg], 2)
            bval_str = bin(bval)[2:]
            if bval_str.count('1') > 4 or invalid_nm:
                inv_venom_blk += 1
    return inv_venom_blk

def count_invalid_nm_pattern(segmat, V):
    inv_nm_pattern = 0
    for seg in range(n//M):
        for i in range(m):
            if segmat[i][seg].count('1') > 2:
                inv_nm_pattern += 1
    return inv_nm_pattern

def hash_segmat(segmat, bithashtab):
    hashed_segmat = []
    for i in range(m):
        tmp = []
        for j in range(n//M):
            tmp.append(bithashtab[segmat[i][j]])
        hashed_segmat.append((i, tmp))
    return hashed_segmat

def reorder(row_indices, col_indices, new_id):
    new_row_indices = []
    new_col_indices = []
    for i in range(nnz):
        new_row_indices.append(new_id[row_indices[i]])
        new_col_indices.append(new_id[col_indices[i]])
    return new_row_indices, new_col_indices

def write_to_mtx(row_indices, col_indices):
    coo = coo_matrix((np.array(values), (np.array(row_indices), np.array(col_indices))), shape=(n, n))
    mmwrite("temp.mtx", coo, precision=1)

V = args.v
MAXITER = args.maxiter

# generate segmat for new round
segmat = generate_segmat(row_indices, col_indices)
# print_segmat(segmat)
init_inv_cnt = count_invalid_venom_block(segmat, V)
# print("invalid_venom_blocks: ", inv_cnt)
init_inv_cnt_strict = count_invalid_venom_block_strict(segmat, V)
# print("invalid_v_blocks: ", init_inv_cnt, ", invalid_vnm_blocks: ", init_inv_cnt_strict)
# init_inv_nm_pattern = count_invalid_nm_pattern(segmat, V)

if args.evalonly:
    with open("output.txt", "w") as file:
            file.write(str(init_inv_cnt)+","+str(init_inv_cnt_strict)) 
    sys.exit(0)


inv_cnt = init_inv_cnt
prev_inv_cnt = init_inv_cnt
prev_inv_cnt_strict = init_inv_cnt_strict
# prev_inv_nm_pattern = init_inv_nm_pattern
prev_row_indices = row_indices
prev_col_indices = col_indices

round = 1
if init_inv_cnt > MAXITER: 
    MAXITER = init_inv_cnt

while inv_cnt > 0 and round <= MAXITER:

    # hash segmat to bit hash table
    hashed_segmat = hash_segmat(segmat, bithashtab)
    # print_segmat(hashed_segmat)

    # sorted row by hashed bitstr
    sorted_data = sorted(hashed_segmat, key=lambda x: x[1])
    # print_segmat(sorted_data)

    # get the new_id list from sorted result
    new_id = []
    for i in range(m):
        new_id.append(sorted_data[i][0])
    # print(new_id)

    # reorder
    new_row_indices, new_col_indices = reorder(prev_row_indices, prev_col_indices, new_id)

    # for result after reordering
    segmat = generate_segmat(new_row_indices, new_col_indices)
    # print_segmat(segmat)
    inv_cnt = count_invalid_venom_block(segmat, V)
    # print("invalid_venom_blocks: ", inv_cnt, "newid: ", new_id)
    # print("invalid_venom_blocks: ", inv_cnt)
    inv_cnt_strict = count_invalid_venom_block_strict(segmat, V)
    # print("invalid_v_blocks: ", inv_cnt, ", invalid_vnm_blocks: ", inv_cnt_strict)
    # inv_nm_pattern = count_invalid_nm_pattern(segmat, V)

    # strict improve
    if inv_cnt > prev_inv_cnt:
        write_to_mtx(prev_row_indices, prev_col_indices)
        with open("output.txt", "w") as file:
            file.write(str(init_inv_cnt)+","+str(init_inv_cnt_strict)+","+str(prev_inv_cnt)+","+str(prev_inv_cnt_strict)+","+str(round)) #+","+str(inv_nm_pattern)+","+str(prev_inv_nm_pattern))
        sys.exit(0)


    # for next round
    prev_inv_cnt = inv_cnt
    prev_inv_cnt_strict = inv_cnt_strict
    # prev_inv_nm_pattern = inv_nm_pattern
    prev_row_indices = new_row_indices
    prev_col_indices = new_col_indices
    round += 1

final_cnt = prev_inv_cnt
final_cnt_strict = prev_inv_cnt_strict
# final_cnt = count_invalid_venom_block(segmat, V)
# final_cnt_strict = count_invalid_venom_block_strict(segmat, V)
# final_nm_pattern = count_invalid_nm_pattern(segmat, V)

write_to_mtx(prev_row_indices, prev_col_indices)
with open("output.txt", "w") as file:
    file.write(str(init_inv_cnt)+","+str(init_inv_cnt_strict)+","+str(final_cnt)+","+str(final_cnt_strict)+","+str(round-1)) #+","+str(init_inv_nm_pattern)+ "," +str(final_nm_pattern))