#!/usr/bin/env bash

data="dblp"

python3 src/main.py --dataset ${data} --model NetGen --teacher_forcing 1
python3 src/main.py --dataset ${data} --model NetGenWord --teacher_forcing 1
python3 src/main.py --dataset ${data} --model NetGenLink --teacher_forcing 1

data="nyt"

python3 src/main.py --dataset ${data} --model NetGen --teacher_forcing 1
python3 src/main.py --dataset ${data} --model NetGenWord --teacher_forcing 1
python3 src/main.py --dataset ${data} --model NetGenLink --teacher_forcing 1

data="yelp"

python3 src/main.py --dataset ${data} --model NetGen --teacher_forcing 1
python3 src/main.py --dataset ${data} --model NetGenWord --teacher_forcing 1
python3 src/main.py --dataset ${data} --model NetGenLink --teacher_forcing 1