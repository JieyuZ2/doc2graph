#!/usr/bin/env bash

data=$1

for i in "0.2" "0.4" "0.6"
do
    python3 src/main.py --dataset ${data} --model NetGen --split_ratio ${i} --teacher_forcing 1 --tag ${i}
    python3 src/main.py --dataset ${data} --model NetGenWord --split_ratio ${i} --teacher_forcing 1 --tag ${i}
    python3 src/main.py --dataset ${data} --model NetGenLink --split_ratio ${i} --teacher_forcing 1 --tag ${i}
done