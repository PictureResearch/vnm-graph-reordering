#!/bin/bash

# dataset filenames
filename=(
cora
pubmed
citeseer
flickr
#reddit
#amazon
#yelp
ppi
#nell
#ogbn-products
#pcqm4m
)

TARGET_DIR="./eval_data"

if [ -d "$TARGET_DIR" ]; then
   echo "'$TARGET_DIR' found and now copying files, please wait ..."
else
   echo "Warning: '$TARGET_DIR' NOT found. Create it"
   mkdir $TARGET_DIR
fi

OUTPUT=output/gcn_newallpairs_avglist.csv
rm -rf $OUTPUT
rm -rf ${TARGET_DIR}/*

make clean && make spmm

for (( i=0; i<${#filename[@]}; i++ ));
do
echo "${filename[$i]}"

echo "./spmm --mtxfile /root/jchen73_ws/GCN/BiGCN_CUDA/A_HW/data/${filename[$i]}/${filename[$i]}.mtx --maxiter 10 --n 64"
printf "${filename[$i]}, " >> ${OUTPUT}
./spmm --mtxfile /root/jchen73_ws/GCN/BiGCN_CUDA/A_HW/data/${filename[$i]}/${filename[$i]}.mtx --maxiter 10 --n 64 >> ${OUTPUT}

done