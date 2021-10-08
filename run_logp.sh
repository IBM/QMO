#!/bin/bash
seqs=()
while read -r smiles logp; do
    seqs+=("$smiles")
done < data/logp_test.csv

if [[ ! -e logs/logp ]]
then
    mkdir -p logs/logp
fi
for s in "${seqs[@]}"
do
    python QMO/run.py -t=80 -k=1 -w=25 --beta=10 -q=100 --base_lr=0.1 --adam \
    --flip-weight -s=123456789 --sim=0.4 --score=logP $s > logs/logp/${s}.log
done
