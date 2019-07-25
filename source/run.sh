#!/bin/bash

# if you want to use the standard datasets, 
# set this variable to the extracted data from
# https://drive.google.com/drive/folders/1ytwtPNZGs7DfoLavsfkw5DaIa-PkhACG?usp=sharing
STD_DIR=usage_change

# set this variable to the result directory
RES_DIR=/home/gjawahar/temp/obj

# standard dataset configuration
declare -A data
data=( ["young"]="birthyear.1990_2009.lowercase" ["old"]="birthyear.1950_1969.lowercase" ["male"]="gender.male.lowercase" ["female"]="gender.female.lowercase" ["performer"]="occupation.performer.lowercase" ["sports"]="occupation.sports.lowercase" ["creator"]="occupation.creator.lowercase" ["weekday"]="yang.weekday.lowercase" ["weekend"]="yang.weekend.lowercase" ["french2014"]="french.2014.lowercase" ["french2018"]="french.2018.lowercase" ["hebrew2014"]="hebrew.2014.lowercase" ["hebrew2018"]="hebrew.2018.lowercase")

# train word2vec on custom dataset
if [ $1 == "train" ] ; then
  if [ $2 == "custom" ] ; then
    python word2vec.py --data $3 --save $RES_DIR/$4
    python word2vec.py --data $5 --save $RES_DIR/$6
  fi
fi

# detect words with usage change
if [ $1 == "detect" ] ; then
  if [ $2 == "custom" ] ; then
    python extract_from_single_split.py --data_a $3 --data_b $4 --embed_a $RES_DIR/$5.seed123 --embed_b $RES_DIR/$6.seed123 --name_split_a $5 --name_split_b $6 --out_topk $RES_DIR/detect_$5_$6_ 
  fi
  if [ $2 == "standard" ] ; then
    python extract_from_single_split.py --data_a $STD_DIR/tokdata/"${data[$3]}" --data_b $STD_DIR/tokdata/"${data[$4]}" --embed_a $STD_DIR/embeddings/"${data[$3]}".seed123.mfreq20 --embed_b $STD_DIR/embeddings/"${data[$4]}".seed123.mfreq20 --name_split_a $3 --name_split_b $4 --out_topk $RES_DIR/detect_$3_$4_
  fi
fi

# visualize neighbors for given words
if [ $1 == "visualize" ] ; then
  if [ $2 == "custom" ] ; then
    python visualize_neighbors.py --data_a $3 --data_b $4 --embed_a $RES_DIR/$5.seed123 --embed_b $RES_DIR/$6.seed123 --name_split_a $5 --name_split_b $6 --out_dir $RES_DIR/vis_$5_$6_ --words $7
  fi
  if [ $2 == "standard" ] ; then
    python visualize_neighbors.py --data_a $STD_DIR/tokdata/"${data[$3]}" --data_b $STD_DIR/tokdata/"${data[$4]}" --embed_a $STD_DIR/embeddings/"${data[$3]}".seed123.mfreq20 --embed_b $STD_DIR/embeddings/"${data[$4]}".seed123.mfreq20 --name_split_a $3 --name_split_b $4 --out_dir $RES_DIR/vis_$3_$4_ --words $5
  fi
fi



