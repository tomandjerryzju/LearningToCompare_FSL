#!/usr/bin/env bash

src=$1
des=$2
class_num=$3

for ((i=0; i<$class_num; i++))
do
    mkdir $des/$i
    python $image_process_utils/mv_cp_file_reg.py $src $des/$i ${i}.jpg -1 cp False
done
