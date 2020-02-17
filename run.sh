#!/usr/bin/env bash

data=$1

python3 src/main.py --dataset ${data} --model NetGen --teacher_forcing 1 --recon_weight 1.0 --tag "11n"
python3 src/main.py --dataset ${data} --model NetGen --teacher_forcing 1 --recon_weight 0.0 --tag "10n"

python3 src/main.py --dataset ${data} --model NetGenWord --teacher_forcing 1 --recon_weight 1.0 --tag "11n"
python3 src/main.py --dataset ${data} --model NetGenWord --teacher_forcing 1 --recon_weight 0.0 --tag "10n"

python3 src/main.py --dataset ${data} --model NetGenS --teacher_forcing 1 --recon_weight 1.0 --tag "11n"
python3 src/main.py --dataset ${data} --model NetGenS --teacher_forcing 1 --recon_weight 0.0 --tag "10n"

python3 src/main.py --dataset ${data} --model NetGenWordS --teacher_forcing 1 --recon_weight 1.0 --tag "11n"
python3 src/main.py --dataset ${data} --model NetGenWordS --teacher_forcing 1 --recon_weight 0.0 --tag "10n"